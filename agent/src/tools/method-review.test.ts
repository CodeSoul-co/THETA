import assert from 'node:assert/strict';
import test from 'node:test';
import { executeMining, type MiningPlan } from './mining-tools.js';
import type { ProductSession } from '../memory/session-store.js';
import type { InferenceProvider } from '../providers/types.js';
import type { AnalysisRuntime } from '../adapters/analysis-runtime.js';
import { reviewMethod } from './method-review.js';

test('structured methodological feedback retains extra issues without converting rejection into approval',async()=>{
  const provider:InferenceProvider={id:'fixture',async infer(request){
    assert.deepEqual(request.options?.extra?.toolChoice,{type:'function',function:{name:'submit_method_review'}});
    assert.match((request.input as {instructions:string}).instructions,/Do not introduce a stricter/);
    return {id:'r',output:{kind:'tool_calls',toolCalls:[{name:'submit_method_review',arguments:{verdict:'revise',requestedConstruct:'time',measuredConstruct:'frequency',issues:Array.from({length:6},(_,i)=>`issue ${i}`),nextAction:'Fix the measurement mismatch'}}]}};
  }};
  const result=await reviewMethod(provider,'r','Study timing',{} as MiningPlan);
  assert.equal(result.verdict,'revise');assert.equal(result.issues.length,6);
});

test('exploration precedes hypotheses; a methodological proxy mismatch blocks testing and stale reviews cannot transfer',async()=>{
  const session:ProductSession={id:'s',runId:'r',title:'study',datasetRefs:['d'],messages:[],updatedAt:'',lastUserIntent:'Study whether an intervention changes completion time, not message frequency.',
    reports:[{jobId:'j',reportPath:'/r',reportStatus:'complete',files:[]}],interpretations:[{id:'i',jobIds:['j'],question:'q',contextHash:'h',documentPath:'/i'}]};
  const plan:MiningPlan={question:'Does intervention affect completion time?',candidates:[{hypothesis:'Intervention changes completion time',topicEvidence:['probe receipt'],inclusion:'Explicit delay',exclusion:'Explicit prompt completion',unknown:'No timing evidence',test:'Count messages mentioning intervention',falsification:'Check alternative explanations'}],requiredFiles:['report.md'],stoppingRule:'One test or limitations'};
  let requests=0,runs=0,aligned=false;
  const provider:InferenceProvider={id:'fixture',async infer(request){requests++;const input=request.input as {instructions:string;messages:Array<{content:string}>};assert.match(input.messages[0].content,/completion time/);assert.match(input.instructions,/not the answer or a benchmark score/);return {id:'response',output:{kind:'tool_calls',toolCalls:[{name:'submit_method_review',arguments:{verdict:aligned?'aligned':'revise',requestedConstruct:'completion time',measuredConstruct:aligned?'documented completion time':'message frequency',issues:aligned?[]:['The proxy does not measure the requested result.'],nextAction:aligned?'Run the planned test':'Define and measure the requested outcome.'}}]}};}};
  const runtime:AnalysisRuntime={inspect:()=>({}),async execute(){runs++;return {};},deliver:()=>({delivered:true})};
  const context={session,save(){}};
  await executeMining(runtime,'analysis_execute',{stage:'explore',purpose:'Inspect originals',code:'print(1)',outputs:[]},context,provider);assert.equal(runs,1);assert.equal(requests,0);
  await executeMining(runtime,'analysis_plan',plan,context,provider);assert.equal(requests,1);
  await executeMining(runtime,'analysis_plan',plan,context,provider);assert.equal(requests,1,'Identical critique is reused, not voted on repeatedly');
  await assert.rejects(executeMining(runtime,'analysis_execute',{purpose:'Test',code:'print(1)',outputs:[]},context,provider),/no aligned method review/);
  aligned=true;plan.candidates[0].test='Compare directly documented completion time with uncertainty and composition checks';
  await executeMining(runtime,'analysis_plan',plan,context,provider);
  await executeMining(runtime,'analysis_execute',{purpose:'Test',code:'print(1)',outputs:[]},context,provider);assert.equal(runs,2);
  const beforeArtifactChange=requests;
  plan.requiredFiles=['renamed-report.md','evidence.json'];
  await executeMining(runtime,'analysis_plan',plan,context,provider);
  assert.equal(requests,beforeArtifactChange,'Artifact inventory edits do not re-vote on an unchanged method');
  await executeMining(runtime,'analysis_deliver',{},context,provider);
  session.lastUserIntent='A different research objective';
  await assert.rejects(executeMining(runtime,'analysis_deliver',{},context,provider),/no aligned method review/);
});
