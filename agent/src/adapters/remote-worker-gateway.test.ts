import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { mkdtempSync, writeFileSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { createHash } from 'node:crypto';
import type { AddressInfo } from 'node:net';
import { ConfiguredComputeGateway } from './remote-worker-gateway.js';
import { workerConfiguration } from './worker-configuration.js';
import { EffectApprovals } from '../domain/effect-approval.js';
import { ResearchStore } from '../memory/research-store.js';
import type { CapabilityWorker } from './python-worker.js';
import type { ComputeRequest } from './compute-gateway.js';
import { createManualServer } from '../../web/manual-server.js';
import { LocalProductTools } from '../tools/local-tools.js';
import type { ProductSession } from '../memory/session-store.js';
import { DatabaseSync } from 'node:sqlite';
const hash=(text:string)=>createHash('sha256').update(text).digest('hex');
async function fixture(){
  const home=mkdtempSync(path.join(tmpdir(),'theta-remote-contract-'));
  let status='running', wrongProject=false, corrupt=false, quoteCost=0, submits=0;
  const calls:{route:string;body:any;authorization:string|undefined}[]=[];
  const artifacts:Record<string,string>={
    'index.html':'<!doctype html><img src="native/figure.svg">',
    'training/split_results.json':JSON.stringify({groups:{all:{count:4},train:{count:2},validation:{count:1},test:{count:1}}}),
    'native/figure.svg':'<svg xmlns="http://www.w3.org/2000/svg"/>',
    'native/topic_proportions.csv':'topic,value\n1,0.6\n2,0.4',
  };
  let base='';
  const server=createServer(async(req,res)=>{
    const chunks=[];for await(const chunk of req)chunks.push(chunk);const raw=Buffer.concat(chunks).toString();
    const route=req.url!;const body=raw.startsWith('{')?JSON.parse(raw):raw;
    calls.push({route,body,authorization:req.headers.authorization});
    const json=(value:any)=>{res.setHeader('content-type','application/json');res.end(JSON.stringify(value))};
    if(route==='/v1/capabilities')return json({protocol:'theta-worker-v1',models:['lda','bertopic']});
    if(route==='/v1/preview')return json({readiness:{ready:true},quote:{id:'q1',maxCost:quoteCost,currency:'CNY',expiresAt:new Date(Date.now()+3600000).toISOString()}});
    if(route==='/v1/datasets')return json({datasetId:'data-'+calls.length,sha256:raw.match(/name="sha256"\r\n\r\n([^\r]+)/)![1],projectId:raw.match(/name="projectId"\r\n\r\n([^\r]+)/)![1]});
    if(route==='/v1/jobs'){submits++;return json({id:'remote-1',projectId:body.projectId});}
    const project=decodeURIComponent(route.split('/')[3]??'');
    if(route.endsWith('/results'))return json({schema:'theta-results-v1',status:'ready',projectId:project,jobId:'remote-1',files:Object.entries(artifacts).map(([name,data])=>({name,url:base+'/files/'+encodeURIComponent(name)+'?signature=test',sizeBytes:Buffer.byteLength(data),sha256:corrupt?'a'.repeat(64):hash(data),kind:name.endsWith('.svg')?'figure':name.endsWith('.csv')?'table':'artifact'})),evidence:{tables:[],figures:[],evidence:[],matrices:[]}});
    if(route.startsWith('/files/'))return res.end(artifacts[decodeURIComponent(route.split('?')[0].slice(7))]);
    if(route.endsWith('/cancel'))status='cancelled';
    return json({id:'remote-1',projectId:wrongProject?'another-project':project,status,phase:'training',percent:45,telemetry:{events:[{kind:'epoch',current:2,total:10}]}});
  });
  await new Promise<void>(r=>server.listen(0,'127.0.0.1',r));base=`http://127.0.0.1:${(server.address() as AddressInfo).port}`;
  const reports=new Map<string,any>();const localCalls:string[]=[];
  const worker:CapabilityWorker={async call<T>(op:string,input:any):Promise<T>{
    localCalls.push(op);
    if(op==='compute.receive'){reports.set(input.request.jobId,input.report);return input.report;}
    if(op==='compute.results')return reports.get(input.jobId);
    if(op==='compute.preview')return {readiness:{ready:true},execution:{}} as T;
    if(op==='dataset.profile')return {columns:['text']} as T;
    if(op==='runtime.config')return {embedding:{configured:false}} as T;
    throw new Error('unexpected local call '+op);
  }};
  const configuration=workerConfiguration(home);configuration.save({mode:'remote',url:base,token:'test-secret-unique',maxCost:0});
  const file=path.join(home,'data.csv');writeFileSync(file,'text\na\nb\nc\nd');
  const dataset={datasetRef:'source',sha256:hash(readFileSync(file,'utf8')),fileName:'data.csv',managedPath:file,sizeBytes:13};
  const plan={modelId:'lda',textColumn:'text',params:{num_topics:2},timeoutSeconds:300,rationale:'local',dataSplit:{enabled:true,mode:'upload' as const,sources:{train:{dataset,textColumn:'text'},validation:{dataset,textColumn:'text'},test:{dataset,textColumn:'text'}}}};
  const gateway=new ConfiguredComputeGateway(home,worker);
  const request=async():Promise<ComputeRequest>=>({jobId:'job-'+hash('job'),runId:'run',projectId:'project-a',dataset,plan,execution:(await gateway.preview({dataset,plan})).execution});
  const authorize=(payload:ComputeRequest)=>{const approvals=new EffectApprovals(new ResearchStore(home));const effect=approvals.request('session',{action:'compute.submit',target:'local',payload,summary:'test'});return approvals.decide(effect.id,'session',effect.hash,true)!;};
  return {home,base,server,worker,gateway,configuration,request,authorize,calls,localCalls,dataset,plan,artifacts,finish(){status='completed'},corrupt(){corrupt=true},wrong(){wrongProject=true},cost(){quoteCost=2},submits:()=>submits,async close(){server.closeAllConnections();await new Promise<void>(r=>server.close(()=>r()));rmSync(home,{recursive:true,force:true})}};
}
test('remote worker transfers only selected data, preserves splits and restores jobs across settings changes',async()=>{
 const f=await fixture();try{
  assert.ok(!JSON.stringify(f.configuration.publicValue()).includes('test-secret'));
  assert.ok(!readFileSync(path.join(f.home,'compute-settings.json'),'utf8').includes('test-secret'));
  const request=await f.request();const receipt=f.authorize(request);await f.gateway.submit(request,receipt);const state=await f.gateway.status(request.jobId);
  assert.equal(state.status,'running');assert.equal(state.telemetry!.events![0].current,2);
  const submission=f.calls.find(c=>c.route==='/v1/jobs')!.body;
  assert.equal(f.calls.filter(c=>c.route==='/v1/datasets').length,4);
  assert.equal(submission.projectId,'project-a');assert.ok(submission.plan.dataSplit.sources.test.dataset.datasetId);
  assert.ok(!JSON.stringify(submission).includes(f.home));assert.ok(!f.localCalls.includes('compute.submit'));
  await f.gateway.submit(request,receipt);assert.equal(f.submits(),1);
  f.configuration.save({mode:'local',url:f.base,maxCost:0});
  const restored=new ConfiguredComputeGateway(f.home,f.worker);
  f.finish();assert.equal((await restored.status(request.jobId)).status,'running');
  const report=await restored.results(request.jobId,'report');assert.equal(report.files.length,4);
  assert.equal((await restored.status(request.jobId)).status,'completed');
  assert.equal(readFileSync(path.join(report.resultDir,'native/topic_proportions.csv'),'utf8'),f.artifacts['native/topic_proportions.csv']);
  assert.ok(f.calls.every(c=>c.authorization==='Bearer test-secret-unique'));
 }finally{await f.close()}
});
test('remote quotes, changed settings, project isolation and corrupt delivery fail before completed',async()=>{
 const f=await fixture();try{
  f.cost();await assert.rejects(f.request(),/费用上限/);
  f.configuration.save({mode:'remote',url:f.base,maxCost:3});
  const request=await f.request();const receipt=f.authorize(request);
  assert.throws(()=>f.gateway.assertSelection('stale'),/配置已变化/);
  await f.gateway.submit(request,receipt);f.wrong();await assert.rejects(f.gateway.status(request.jobId),/归属/);
 }finally{await f.close()}
 const broken=await fixture();try{
  const request=await broken.request();await broken.gateway.submit(request,broken.authorize(request));broken.corrupt();broken.finish();
  await assert.rejects(broken.gateway.results(request.jobId,'report'),/校验失败/);
  assert.notEqual((await broken.gateway.status(request.jobId)).status,'completed');
 }finally{await broken.close()}
});
test('manual mode uses the configured remote gateway and receives full result catalog',async()=>{
 const f=await fixture();let manual:ReturnType<typeof createManualServer>|undefined;
 try{
  manual=createManualServer(f.home,f.worker);await new Promise<void>(r=>manual!.listen(0,'127.0.0.1',r));
  const base=`http://127.0.0.1:${(manual.address() as AddressInfo).port}`;
  const db=new DatabaseSync(path.join(f.home,'manual.sqlite'));
  const fileId=db.prepare('INSERT INTO records(kind,value) VALUES (?,?)').run('file',JSON.stringify({dataset_name:'project-a',dataset:f.dataset})).lastInsertRowid;db.close();
  const response=await fetch(base+'/api/train/start',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({file_id:Number(fileId),dataset_name:'project-a',model_type:'lda',text_column:'text',num_topics:2,worker_selection:f.configuration.publicValue().fingerprint})});
  assert.equal(response.status,201,await response.text());
  for(let i=0;i<40&&!f.submits();i++)await new Promise(r=>setTimeout(r,20));
  assert.equal(f.submits(),1);assert.ok(!f.localCalls.includes('compute.submit'));
  f.finish();
  const jobId=f.calls.find(c=>c.route==='/v1/jobs')!.body.clientJobId;
  // Reuse the persisted binding in a new gateway, as after an app restart.
  const g=new ConfiguredComputeGateway(f.home,f.worker);await g.results(jobId,'report');
  const response2=await fetch(base+'/api/results/project-a/catalog');const data=await response2.json() as any;
  assert.equal(data.results[0].status,'completed');assert.equal(data.results[0].artifacts.fileCount,4);
 }finally{if(manual){manual.closeAllConnections();await new Promise<void>(r=>manual!.close(()=>r()));}await f.close()}
});
test('new installations default to local compute without any remote requests',async()=>{
 const home=mkdtempSync(path.join(tmpdir(),'theta-local-default-'));try{
  assert.equal(workerConfiguration(home).read().mode,'local');
  const calls:string[]=[];const worker:CapabilityWorker={async call<T>(op:string){calls.push(op);return {} as T}};
  const g=new ConfiguredComputeGateway(home,worker,()=>{throw new Error('network forbidden')});
  await g.preview({dataset:{} as any,plan:{} as any});await g.status('local-job');
  assert.deepEqual(calls,['compute.preview','compute.status']);
 }finally{rmSync(home,{recursive:true,force:true})}
});

test('conversation training uses the remote worker with the original approved quote and project',async()=>{
 const f=await fixture();try{
  const records=new ResearchStore(f.home);
  records.put('dataset',f.dataset.datasetRef,f.dataset);
  records.put('run','run-a',{id:'run-a',goal:'study',datasetRef:f.dataset.datasetRef,plan:f.plan,planHash:hash('plan'),jobs:[],notes:[],computeBackend:'local'});
  const session:ProductSession={id:'session-a',title:'study',runId:'run-a',runIds:['run-a'],datasetRefs:[f.dataset.datasetRef],messages:[],updatedAt:''};
  const tools=new LocalProductTools({runtimeDb:path.join(f.home,'research.sqlite'),uploadDir:path.join(f.home,'uploads'),worker:f.worker,datasetScope:()=>({projectId:'project-a',datasetRefs:[f.dataset.datasetRef],sessionIds:['session-a']})});
  await tools.execute('training_advance',{}, {session,userMessage:'train',save(){}});
  assert.ok(session.pendingConfirmation?.summary.includes('远端'));
  await tools.approve(session,'确认');
  assert.equal(f.submits(),1);assert.ok(!f.localCalls.includes('compute.submit'));
  assert.equal(f.calls.filter(c=>c.route==='/v1/preview').length,1,'approval must retain the quote already shown');
  assert.equal(f.calls.find(c=>c.route==='/v1/jobs')!.body.projectId,'project-a');
 }finally{await f.close()}
});
