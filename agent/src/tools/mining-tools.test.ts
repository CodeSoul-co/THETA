import assert from 'node:assert/strict';
import test from 'node:test';
import { executeMining, miningPlanSchema } from './mining-tools.js';
import type { ProductSession } from '../memory/session-store.js';
import type { AnalysisRuntime } from '../adapters/analysis-runtime.js';

const ready = (): ProductSession => ({id:'session',title:'study',runId:'study',datasetRefs:['data'],messages:[],updatedAt:'',
  reports:[{jobId:'job',reportStatus:'complete',reportPath:'/report',files:[]}],
  interpretations:[{id:'interpretation',jobIds:['job'],question:'why',contextHash:'h',documentPath:'/interpretation'}]});
const plan = {question:'Does an observed condition differ between groups?',candidates:[{hypothesis:'Group composition explains the difference',topicEvidence:['topic probe receipt'],inclusion:'Explicit event',exclusion:'Explicit absence',unknown:'Ambiguous or missing evidence',test:'Full assignment counts and metadata contrasts',falsification:'Inspect discordant documents and remove largest stratum'}],requiredFiles:['report.md','evidence.json'],stoppingRule:'One tested finding or a documented unsupported hypothesis'};

test('mining requires host runtime, completed interpretation and no pending action', async()=>{
  const session=ready(); const context={session,save(){}};
  const runtime: AnalysisRuntime={inspect(){throw new Error('Unexpected access');},async execute(){throw new Error('Unexpected execution');},deliver(){throw new Error('Unexpected delivery');}};
  await assert.rejects(executeMining(undefined,'analysis_workspace',{},context),/not configured/);
  session.interpretations=[];
  await assert.rejects(executeMining(runtime,'analysis_workspace',{},context),/Complete the authorized/);
  session.interpretations=ready().interpretations;
  session.pendingConfirmation={kind:'action',checkpointId:'c',contentHash:'h',summary:'waiting',runId:'study'};
  await assert.rejects(executeMining(runtime,'analysis_workspace',{},context),/resolve pending/);
});

test('plan is not execution; delivery uses exact required files and is study-scoped',async()=>{
  const session=ready(); let runs=0,deliveries=0;
  const runtime: AnalysisRuntime={inspect(){return {};},async execute(){runs++;return {};},deliver(_session,files){deliveries++;assert.deepEqual(files,plan.requiredFiles);return {delivered:true};}};
  const context={session,save(){}};
  await assert.rejects(executeMining(runtime,'analysis_execute',{purpose:'test',code:'print(1)',outputs:[]},context),/analysis_plan/);
  await executeMining(runtime,'analysis_plan',plan,context);
  assert.equal(runs,0);assert.equal(deliveries,0);
  await executeMining(runtime,'analysis_execute',{purpose:'test',code:'print(1)',outputs:[]},context);
  await executeMining(runtime,'analysis_deliver',{},context);
  assert.equal(runs,1);assert.equal(deliveries,1);
  session.runId='another';
  await assert.rejects(executeMining(runtime,'analysis_deliver',{},context),/analysis_plan/);
});

test('plans require uncertainty/falsification and reject unsafe paths',()=>{
  assert.equal(miningPlanSchema.safeParse(plan).success,true);
  for (const name of ['/tmp/report','../report','folder/../report','folder\\report']) assert.equal(miningPlanSchema.safeParse({...plan,requiredFiles:[name]}).success,false);
  assert.equal(miningPlanSchema.safeParse({...plan,candidates:[{...plan.candidates[0],unknown:''}]}).success,false);
});
