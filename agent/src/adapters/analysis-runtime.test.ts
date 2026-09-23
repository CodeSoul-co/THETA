import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdtempSync, mkdirSync, readFileSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { DockerAnalysisRuntime, WorkerAnalysisRuntime, workspaceFile } from './analysis-runtime.js';
import { DockerSandboxWorker, type SandboxWorker } from '../workers/sandbox-worker.js';
import type { ProductSession } from '../memory/session-store.js';

const session: ProductSession={id:'session',runId:'study',title:'test',datasetRefs:['data'],messages:[],updatedAt:''};
test('null host execution quota keeps per-process isolation without limiting useful calls',async()=>{
  const {realpathSync}=await import('node:fs');
  const root=realpathSync(mkdtempSync(path.join(os.tmpdir(),'theta-unbounded-')));
  for(const dir of ['input','model','work','library'])mkdirSync(path.join(root,dir));
  const image='sha256:'+'b'.repeat(64),description=new DockerSandboxWorker({image}).describe();
  let calls=0;
  const worker:SandboxWorker={describe:()=>description,async run(){calls++;return {exitCode:0,output:'ok',diagnostics:'',seconds:0,cleanupConfirmed:true};},async recover(){return true;}};
  const runtime=new WorkerAnalysisRuntime({stateDirectory:path.join(root,'audit'),libraryDirectory:path.join(root,'library'),worker:()=>worker,
    binding:()=>({scope:'fixture',workspace:path.join(root,'work'),inputs:path.join(root,'input'),model:path.join(root,'model'),image,maxExecutions:null,maxSeconds:20,workerFingerprint:description.fingerprint})});
  try {
    for(let i=0;i<6;i++)await runtime.execute(session,{purpose:'Distinct computation',code:`print(${i})`,outputs:[],seconds:10});
    assert.equal(calls,6);assert.equal((runtime.inspect(session) as {executionsRemaining:null}).executionsRemaining,null);
  }finally{rmSync(root,{recursive:true,force:true});}
});
test('worker swaps require matching authorization, isolate receipts and retain failed outcomes during host recovery',async()=>{
  const {realpathSync}=await import('node:fs');
  const root=realpathSync(mkdtempSync(path.join(os.tmpdir(),'theta-worker-binding-')));
  for(const dir of ['input','model','work','library'])mkdirSync(path.join(root,dir));
  const image='sha256:'+'a'.repeat(64);
  let description=new DockerSandboxWorker({image}).describe(), grant=description.fingerprint, clean=false, executions=0, available=true;
  const worker:SandboxWorker={describe:()=>({...description}),async run(request){executions++;return {exitCode:1,output:'',diagnostics:'controlled failure',seconds:0,error:'controlled failure',cleanupConfirmed:clean};},async recover(){return clean;}};
  const runtime=new WorkerAnalysisRuntime({stateDirectory:path.join(root,'audit'),libraryDirectory:path.join(root,'library'),worker:()=>available?worker:undefined,
    binding:()=>({scope:'fixture',workspace:path.join(root,'work'),inputs:path.join(root,'input'),model:path.join(root,'model'),image,maxExecutions:5,maxSeconds:20,workerFingerprint:grant})});
  try {
    const first=await runtime.execute(session,{purpose:'Fixture failure',code:'print(1)',outputs:[],seconds:10}) as {id:string;completed:boolean};
    assert.equal(first.completed,false);
    for(let i=0;i<70;i++)writeFileSync(path.join(root,'work','analysis-scripts',`fixture-${i}.py`),'print(1)');
    writeFileSync(path.join(root,'work','report.md'),'Actual analysis output');
    const inventory=runtime.inspect(session) as {files:Array<{path:string}>;savedScripts:Array<{path:string}>;inventoryExcludes:string[]};
    assert.deepEqual(inventory.files.map(file=>file.path),['report.md']);
    assert.equal(inventory.savedScripts.length,1);
    assert.deepEqual(inventory.inventoryExcludes,['analysis-scripts/']);
    assert.equal((runtime.inspect(session) as {status:string}).status,'host_recovery_required');
    await assert.rejects(runtime.execute(session,{purpose:'Blocked retry',code:'print(1)',outputs:[],seconds:10}),/cleanup is unresolved/);
    assert.throws(()=>runtime.deliver(session,['report.md']),/cleanup/);
    await assert.rejects(runtime.recover(session,'another-id'),/owned/);
    assert.equal((await runtime.recover(session,first.id)).cleanupConfirmed,false);
    clean=true;await runtime.recover(session,first.id);
    const state=runtime.inspect(session) as {status:string;executions:Array<{exitCode:number;error:string}>};
    assert.equal(state.status,'ready');assert.equal(state.executions[0].exitCode,1);assert.equal(state.executions[0].error,'controlled failure');
    description=new DockerSandboxWorker({image,cpus:2}).describe();
    assert.throws(()=>runtime.inspect(session),/worker changed/);
    await assert.rejects(runtime.execute(session,{purpose:'No transferred grant',code:'print(1)',outputs:[],seconds:10}),/worker changed/);
    assert.equal(executions,1);
    grant=description.fingerprint;
    assert.equal((runtime.inspect(session) as {executionsUsed:number}).executionsUsed,0);
    assert.equal((runtime.inspect(session) as {delivered:boolean}).delivered,false);
    available=false;
    assert.throws(()=>runtime.inspect(session),/no local fallback/);
  } finally {rmSync(root,{recursive:true,force:true});}
});
test('workspace paths reject traversal and symlink escapes',()=>{
  const root=mkdtempSync(path.join(os.tmpdir(),'theta-path-'));
  try {
    for(const name of ['/tmp/x','../x','a/../../x','a\\x','a//x']) assert.throws(()=>workspaceFile(root,name));
    symlinkSync(os.tmpdir(),path.join(root,'escape'));
    assert.throws(()=>workspaceFile(root,'escape/x'),/Symlink/);
    assert.equal(workspaceFile(root,'report.md'),path.join(root,'report.md'));
  } finally {rmSync(root,{recursive:true,force:true});}
});

test('absent grant cannot execute, inspect or deliver',async()=>{
  const runtime=new DockerAnalysisRuntime({stateDirectory:'/unused',libraryDirectory:'/unused',binding:()=>undefined});
  assert.throws(()=>runtime.inspect(session),/host-authorized/);
  assert.throws(()=>runtime.deliver(session,['report']),/host-authorized/);
  await assert.rejects(runtime.execute(session,{purpose:'test',code:'print(1)',outputs:[],seconds:1}),/host-authorized/);
});

test('real offline sandbox persists scripts, verifies outputs, fails safely and archives exact bytes', {skip:!process.env.THETA_ANALYSIS_TEST_IMAGE,timeout:90000},async()=>{
  // Resolve macOS /var -> /private/var before granting strict mount paths.
  const {realpathSync}=await import('node:fs');
  const root=realpathSync(mkdtempSync(path.join(os.tmpdir(),'theta-mining-')));
  for (const name of ['input','model','work']) mkdirSync(path.join(root,name));
  const library=realpathSync(fileURLToPath(new URL('../../../workers/mining',import.meta.url)));
  const runtime=new DockerAnalysisRuntime({stateDirectory:path.join(root,'audit'),libraryDirectory:library,
    binding:current=>current.runId==='study'?{scope:'approved-fixture',workspace:path.join(root,'work'),inputs:path.join(root,'input'),model:path.join(root,'model'),image:process.env.THETA_ANALYSIS_TEST_IMAGE!,maxExecutions:4,maxSeconds:20}:undefined});
  try {
    writeFileSync(path.join(root,'input','rows.json'),JSON.stringify([{doc_id:'a',text:'explicit event',group:'a'},{doc_id:'b',text:'uncertain',group:'b'}]));
    const first=await runtime.execute(session,{purpose:'Validate corpus labels and save evidence',code:"import json, os\nfrom pathlib import Path\nfrom theta_mining import partition\nassert not any(k in os.environ for k in ('DEEPSEEK_API_KEY','OPENAI_API_KEY','THETA_TEST_SECRET'))\nrows=json.load(open('/input/rows.json'))\nPath('evidence.json').write_text(json.dumps(partition(rows,{'a':'positive','b':'unknown'})))\nprint('saved')",outputs:['evidence.json'],seconds:15}) as {completed:boolean;script:{path:string};diagnostics:string};
    assert.equal(first.completed,true,first.diagnostics);
    const second=await runtime.execute(session,{purpose:'Reuse the saved script and deliver a report',code:`import runpy\nfrom pathlib import Path\nrunpy.run_path('/work/${first.script.path}')\nPath('report.md').write_text('Synthetic plumbing check only; not a mining-quality result.')`,outputs:['report.md','evidence.json'],seconds:15}) as {completed:boolean};
    assert.equal(second.completed,true);
    assert.equal((runtime.deliver(session,['report.md','evidence.json']) as {delivered:boolean}).delivered,true);
    assert.equal((runtime.inspect(session) as {delivered:boolean}).delivered,true);
    writeFileSync(path.join(root,'work','report.md'),'changed');
    assert.equal((runtime.inspect(session) as {delivered:boolean}).delivered,false);
    const failure=await runtime.execute(session,{purpose:'Fail without claiming output',code:"raise ValueError('controlled failure')",outputs:['missing.json'],seconds:10}) as {completed:boolean;missing:string[];script:{path:string}};
    assert.equal(failure.completed,false); assert.deepEqual(failure.missing,['missing.json']);
    assert.match(readFileSync(path.join(root,'work',failure.script.path),'utf8'),/controlled failure/);
    const timeout=await runtime.execute(session,{purpose:'Timeout cleanup',code:'import time\ntime.sleep(10)',outputs:[],seconds:1}) as {completed:boolean;error:string;cleanupConfirmed:boolean};
    assert.equal(timeout.completed,false); assert.match(timeout.error,/timed out/);
    assert.equal(timeout.cleanupConfirmed,true);
    await assert.rejects(runtime.execute(session,{purpose:'Over budget',code:'print(1)',outputs:[],seconds:1}),/limit exhausted/);
    assert.throws(()=>runtime.inspect({...session,runId:'another'}),/host-authorized/);
  } finally {rmSync(root,{recursive:true,force:true});}
});
