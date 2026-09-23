import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync,rmSync,writeFileSync,mkdirSync} from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {LocalProductTools} from './local-tools.js';
import type {CapabilityWorker} from '../adapters/python-worker.js';
import type {ProductSession} from '../memory/session-store.js';
import {ResearchStore} from '../memory/research-store.js';

const plan={question:'Is x associated with y?',hypotheses:['positive conditional association'],assumptions:['independent rows'],steps:[{method:'ols',y:'y',x:['x'],params:{covariance:'HC3'}}],sensitivity:['check residuals'],stoppingRule:'One prespecified model, then report uncertainty'};

test('free statistics supports no topic run, exact approval, rejection feedback, revision and data/runtime binding',async()=>{
  const home=mkdtempSync(path.join(os.tmpdir(),'theta-statistics-tools-'));let calls=0;let version=1;
  const data={datasetRef:'data',fileName:'fixture.csv',managedPath:path.join(home,'data.csv'),sizeBytes:12,sha256:'abc'};
  const worker:CapabilityWorker={async call<T>(operation:string,input:unknown):Promise<T>{
    const payload=input as Record<string,unknown>;
    if(operation==='statistics.preview')return {version,dataset:payload.dataset,plan:payload.plan} as T;
    if(operation==='statistics.execute'){
      calls++;assert.ok(payload.authorization);const directory=path.join(home,'output');mkdirSync(directory,{recursive:true});writeFileSync(path.join(directory,'results.json'),JSON.stringify({results:[{method:'ols',metrics:{N:20},tables:{estimates:[{value:2},{value:3}]}}]}));
      return {analysisId:payload.analysisId,status:'complete',reportPath:path.join(directory,'report.html'),files:[],results:[]} as T;
    }
    return {} as T;
  }};
  try{
    const records=new ResearchStore(home);records.put('dataset','data',data);
    const tools=new LocalProductTools({runtimeDb:path.join(home,'research.sqlite'),uploadDir:path.join(home,'uploads'),worker});
    const session:ProductSession={id:'s',title:'s',datasetRefs:['data'],messages:[],updatedAt:'',analysisMode:'free'};
    const context={session,userMessage:'Analyze these data',save(){}};
    await tools.execute('statistics_plan',{datasetRef:'data',plan},context);assert.equal(calls,0);assert.equal(session.runId,undefined);
    await tools.execute('statistics_request_approval',{},context);const first=session.pendingConfirmation!;
    tools.deny(session,'Use robust standard errors');assert.equal(calls,0);assert.equal(session.pendingConfirmation,undefined);
    await assert.rejects(tools.approve(session,'确认'),/确认/);
    await tools.execute('statistics_request_approval',{},context);
    await assert.rejects(tools.approve(session,'maybe'),/确认/);
    version=2;await assert.rejects(tools.approve(session,'确认'),/environment changed/);assert.equal(calls,0);
    await tools.execute('statistics_plan',{datasetRef:'data',plan:{...plan,question:'Revised'}},context);
    await tools.execute('statistics_request_approval',{},context);assert.notEqual((session as ProductSession).pendingConfirmation?.checkpointId,first.checkpointId);
    await tools.approve(session,'确认');assert.equal(calls,1);assert.equal(session.statisticalReports?.length,1);
    await assert.rejects(tools.approve(session,'确认'));assert.equal(calls,1);
    const rows=await tools.execute('statistics_results',{step:0,table:'estimates',limit:1},context) as {rows:unknown[];nextOffset:number};assert.equal(rows.rows.length,1);assert.equal(rows.nextOffset,1);
    session.analysisMode='topic';assert.equal(tools.toolAvailable('statistics_plan',session),false);
    assert.equal(tools.toolAvailable('statistics_methods',session),true);
    await assert.rejects(tools.execute('statistics_plan',{datasetRef:'data',plan},context),/主题模式/);
  }finally{rmSync(home,{recursive:true,force:true});}
});

test('real statistics worker exports a paper report and rejects approval replay',async t=>{
  const {existsSync,readFileSync}=await import('node:fs');
  const {packageRoot}=await import('../environment.js');
  if(!existsSync(process.env.THETA_WORKER_STATISTICS_PYTHON??path.join(packageRoot,'.local/runtimes/statistics/bin/python'))){t.skip('Dedicated statistics environment not installed');return;}
  const {PythonCapabilityWorker}=await import('../adapters/python-worker.js');
  const home=mkdtempSync(path.join(os.tmpdir(),'theta-statistics-real-'));
  try{
    const worker=new PythonCapabilityWorker();const tools=new LocalProductTools({runtimeDb:path.join(home,'research.sqlite'),uploadDir:path.join(home,'uploads'),worker});
    const s:ProductSession={id:'real',title:'Synthetic evidence',analysisMode:'free',datasetRefs:[],messages:[],updatedAt:''};
    await tools.attach(path.join(packageRoot,'examples/statistics/demo.csv'),s);
    const context={session:s,userMessage:'Run the synthetic statistical acceptance',save(){}};
    await tools.execute('statistics_plan',{datasetRef:s.datasetRefs[0],plan},context);await tools.execute('statistics_request_approval',{},context);
    const pending={...s.pendingConfirmation!};const records=new ResearchStore(home);const effect=records.get<{payload:Record<string,unknown>}>('effect-approval',pending.checkpointId);
    const result=await tools.approve(s,'确认') as {status:string;reportPath:string;files:Array<{name:string;path:string}>};
    assert.equal(result.status,'complete');assert.ok(existsSync(result.reportPath));
    for(const name of ['esttab.rtf','esttab.tex','esttab.csv','results.json','reproduce.py','01-ols-diagnostics.svg'])assert.ok(result.files.some(f=>f.name===name));
    const raw=JSON.parse(readFileSync(path.join(path.dirname(result.reportPath),'results.json'),'utf8'));
    assert.equal(raw.results[0].metrics.N,180);assert.equal(raw.results[0].metrics.covariance,'HC3');
    const {execFileSync}=await import('node:child_process');
    execFileSync(process.env.THETA_WORKER_STATISTICS_PYTHON??path.join(packageRoot,'.local/runtimes/statistics/bin/python'),['-s',path.join(path.dirname(result.reportPath),'reproduce.py')],{cwd:packageRoot,timeout:30000,env:{...process.env,OMP_NUM_THREADS:'1',OPENBLAS_NUM_THREADS:'1'}});
    const reproduced=JSON.parse(readFileSync(path.join(path.dirname(result.reportPath),'reproduced/results.json'),'utf8'));
    assert.equal(reproduced.reproduction.sameSource,true);
    assert.deepEqual(reproduced.results[0].coefficients,raw.results[0].coefficients);
    assert.doesNotMatch(readFileSync(result.reportPath,'utf8'),/border="1"/);

    await assert.rejects(worker.call('statistics.execute',{...effect.payload,home,authorization:{id:pending.checkpointId,hash:pending.contentHash}}),/already been consumed/);
    const completed=s.statisticalReports![0];
    s.statisticalReports=[];s.pendingStatisticalInterpretation=undefined;s.statisticalExecution!.status='running';
    const recovered=await tools.execute('statistics_status',{},context) as {execution:{status:string};report:{analysisId:string}};
    assert.equal(recovered.execution.status,'complete');assert.equal(recovered.report.analysisId,completed.analysisId);
    assert.equal(s.pendingStatisticalInterpretation,completed.analysisId);
    tools.recordUnderstanding(s,'x has a positive conditional association; synthetic data only.');
    assert.ok(s.statisticalReports?.[0].files.some(f=>f.name==='interpretation.md'));
    await assert.rejects(tools.execute('statistics_synthesize',{analysisId:'another-study'},context),/没有该/);
    const revised=await tools.execute('statistics_synthesize',{analysisId:completed.analysisId},context) as {evidence:Array<{tables:Record<string,{rows:unknown[]}>}>};
    const tableName=Object.keys(raw.results[0].tables)[0];
    assert.deepEqual(revised.evidence[0].tables[tableName].rows,raw.results[0].tables[tableName].slice(0,20));
    assert.equal(s.pendingStatisticalInterpretation,completed.analysisId);
    tools.recordUnderstanding(s,'Revised interpretation, same computed evidence.');
    assert.equal(s.statisticalReports![0].files.filter(f=>f.name==='interpretation.md').length,1);
    assert.match(readFileSync(path.join(path.dirname(result.reportPath),'interpretation.md'),'utf8'),/Revised interpretation/);

    const math:ProductSession={id:'math',title:'Optimization without a file',analysisMode:'free',datasetRefs:[],messages:[],updatedAt:''};
    const mathPlan={...plan,question:'Minimize a bounded linear objective',steps:[{method:'linear_program',params:{c:[-1,-2],a_ub:[[1,1]],b_ub:[5]}}]};
    await tools.execute('statistics_plan',{plan:mathPlan},{session:math,userMessage:'Solve',save(){}});
    await tools.execute('statistics_request_approval',{},{session:math,userMessage:'Solve',save(){}});
    const optimum=await tools.approve(math,'确认') as {results:Array<{metrics:{objective:number}}>};
    assert.equal(optimum.results[0].metrics.objective,-10);assert.equal(math.datasetRefs.length,0);
  }finally{rmSync(home,{recursive:true,force:true});}
});
