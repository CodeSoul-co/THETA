import { z } from 'zod';
import { randomUUID, createHash } from 'node:crypto';
import { readFileSync, existsSync, readdirSync, realpathSync } from 'node:fs';
import path from 'node:path';
import type { CapabilityWorker } from '../adapters/python-worker.js';
import { EffectApprovals } from '../domain/effect-approval.js';
import { contentHash, type Dataset } from '../domain/research.js';
import type { ResearchStore } from '../memory/research-store.js';
import { notebookKey, type ProductSession } from '../memory/session-store.js';
import type { ProductToolContext } from './local-tools.js';
import { analysisReady } from './mining-tools.js';

const column=z.string().min(1).max(200);
const columns=z.array(column).min(1).max(40);
const spec=z.object({method:z.string().regex(/^[a-z][a-z0-9_]{1,70}$/),name:z.string().min(1).max(100).optional(),x:columns.optional(),y:column.optional(),group:column.optional(),time:column.optional(),event:column.optional(),entity:column.optional(),weight:column.optional(),endogenous:columns.optional(),instruments:columns.optional(),factors:columns.optional(),params:z.record(z.unknown()).default({}),missing:z.enum(['drop','error']).default('drop'),alpha:z.number().min(.001).max(.2).default(.05),seed:z.number().int().min(0).max(2147483647).default(42)}).strict();
export const empiricalPlan=z.object({question:z.string().min(1).max(2000),hypotheses:z.array(z.string().max(1000)).max(6),assumptions:z.array(z.string().max(1000)).max(10),steps:z.array(spec).min(1).max(6),sensitivity:z.array(z.string().max(1000)).max(6),stoppingRule:z.string().min(1).max(1000)}).strict().refine(v=>JSON.stringify(v).length<=30000,'Plan exceeds 30000 characters');
export type EmpiricalPlan=z.infer<typeof empiricalPlan>;
export interface StatisticalReport {analysisId:string;runId:string;status:string;reportPath:string;files:Array<{name:string;path:string;kind:string}>;results:Array<Record<string,unknown>>}
export const statisticsTools=[
  {name:'statistics_synthesize',worker:'statistics',effect:'read',description:'Reinterpret an already delivered statistical batch using isolated current evidence, excluding historical assistant claims. Use this when the user requests a fresh or corrected paper interpretation, not merely a table value. No computation, no new approval; the host saves the revised interpretation. Read tables with statistics_results afterwards.',schema:z.object({analysisId:z.string()}).strict()},
  {name:'statistics_status',worker:'statistics',effect:'read',description:'Inspect the persisted current statistical execution and completed step receipts. Recover an already completed, hash-verified report after host interruption without recomputation. Failed/incomplete batches require a new plan and confirmation; this tool never reruns computation.',schema:z.object({}).strict()},
  {name:'statistics_methods',worker:'statistics',effect:'read',description:'List executable Python statistical/econometric/ML/survival/optimization methods, paginated. Only registered methods and their documented parameters are available. Available in either analysis mode.',schema:z.object({family:z.string().optional(),query:z.string().max(100).optional(),offset:z.number().int().min(0).default(0),limit:z.number().int().min(1).max(50).default(30)}).strict()},
  {name:'statistics_inspect',worker:'statistics',effect:'read',description:'Inspect exact required columns, supported parameters, limitations and executable example. Inspect every method before planning; unregistered methods must not be invented.',schema:z.object({method:z.string()}).strict()},
  {name:'statistics_plan',worker:'statistics',effect:'write',description:'Save and validate a concrete empirical analysis plan, 1..6 methods with hypotheses, identification assumptions, sensitivity checks and stopping rule. Omit datasetRef only for pure optimization or prospective power calculations with no data columns. No computation or approval. In free mode topic modeling is optional; in topic mode use after topic report interpretation. Use exact columns and explicit sample rules. Does not accept code/formulas.',schema:z.object({datasetRef:z.string().optional(),plan:empiricalPlan}).strict()},
  {name:'statistics_request_approval',worker:'statistics',effect:'effect',description:'Create an inline host confirmation card for the exact saved statistical batch and worker/data fingerprint. Includes computation, paper tables/figures and evidence interpretation. Does not execute; wait for the human. Denial is never consent. Changing the plan requires a new card.',schema:z.object({}).strict()},
  {name:'statistics_results',worker:'statistics',effect:'read',description:'Read this study’s delivered statistical evidence, full table names and paginated rows. No new estimation. Use actual estimates, intervals and denominators to explain all tables/figures at paper quality.',schema:z.object({analysisId:z.string().optional(),step:z.number().int().min(0).max(5).optional(),table:z.string().max(200).optional(),offset:z.number().int().min(0).default(0),limit:z.number().int().min(1).max(30).default(20)}).strict()},
] as const;

export class StatisticalTools {
  constructor(private records:ResearchStore,private worker:CapabilityWorker,private approvals:EffectApprovals,private home:string){}
  available(name:string,session:ProductSession):boolean {
    return ['statistics_methods','statistics_inspect','statistics_results','statistics_status','statistics_synthesize'].includes(name)||session.analysisMode==='free'||analysisReady(session);
  }
  async execute(name:string,args:Record<string,unknown>,context:ProductToolContext):Promise<unknown>{
    const s=context.session;const key=notebookKey(s);
    if(!this.available(name,s))throw new Error('主题模式需先完成主题结果解读；可在会话模式选择器切换自由分析。');
    if(name==='statistics_synthesize'){
      if(s.pendingConfirmation || s.statisticalExecution?.status==='running')throw new Error('先处理当前确认或执行任务');
      const report=s.statisticalReports?.find(r=>r.analysisId===args.analysisId && r.runId===key && r.status==='complete');
      if(!report)throw new Error('当前研究没有该已交付统计批次');
      const raw=JSON.parse(readFileSync(path.join(path.dirname(report.reportPath),'results.json'),'utf8')) as {results:Array<Record<string,unknown>>};
      let remaining=24000;
      const evidence=raw.results.map((result,step)=>({step,tables:Object.fromEntries(Object.entries(result.tables as Record<string,unknown[]>??{}).slice(0,12).map(([name,rows])=>{
        const preview=rows.slice(0,20);const size=JSON.stringify(preview).length;
        const included=size<=remaining; if(included)remaining-=size;
        return [name,{rows:included?preview:[],totalRows:rows.length,nextOffset:included&&preview.length===rows.length?null:included?preview.length:0}];
      }))}));
      s.pendingStatisticalInterpretation=report.analysisId;context.save();
      return {analysisId:report.analysisId,evidence,instruction:'Use the host-provided current report and these bounded tables to write a fresh interpretation. nextOffset=null means the table is complete; read further pages with statistics_results when necessary. Prior assistant prose is not evidence. Do not compute or create a card.'};
    }
    if(name==='statistics_status'){
      const execution=s.statisticalExecution;
      if(!execution)return {status:'none',instruction:'No statistical execution recorded. Saved plans are not execution.'};
      const directory=path.join(this.home,'statistics',execution.analysisId);
      const progressFile=path.join(this.home,'statistics-progress',`${execution.analysisId}.json`);
      const progress=existsSync(progressFile)?JSON.parse(readFileSync(progressFile,'utf8')):null;
      let workerAlive=false;
      if(progress?.status==='running' && Number.isInteger(progress.pid) && progress.pid>0){try{process.kill(progress.pid,0);workerAlive=true;}catch{}}
      const completedSteps=existsSync(directory)?readdirSync(directory).filter(f=>/^step-\d+\.json$/.test(f)).sort():[];
      if(execution.status!=='complete' && existsSync(path.join(directory,'manifest.json'))){
        const effect=this.approvals.get(execution.approvalId);
        const payload=effect.payload as {analysisId:string;studyKey:string;plan:EmpiricalPlan;dataset:Dataset|null};
        if(effect.status!=='approved'||effect.action!=='statistics.execute'||payload.analysisId!==execution.analysisId||payload.studyKey!==key)throw new Error('Recovery does not match the approved study');
        const manifest=JSON.parse(readFileSync(path.join(directory,'manifest.json'),'utf8')) as {files:Array<{name:string;path:string;kind:string;sha256:string}>};
        for(const file of manifest.files){
          if(path.dirname(realpathSync(file.path))!==realpathSync(directory)||path.basename(file.path)!==file.name||createHash('sha256').update(readFileSync(file.path)).digest('hex')!==file.sha256)throw new Error('Recovered artifact failed integrity validation');
        }
        if(!['results.json','report.html'].every(name=>manifest.files.some(f=>f.name===name)))throw new Error('Incomplete report manifest');
        const raw=JSON.parse(readFileSync(path.join(directory,'results.json'),'utf8'));
        if(raw.analysisId!==payload.analysisId||contentHash(raw.plan)!==contentHash(payload.plan)||contentHash(raw.dataset)!==contentHash(payload.dataset))throw new Error('Recovered report does not match approval');
        this.publish(s,{analysisId:execution.analysisId,runId:key,status:'complete',reportPath:path.join(directory,'report.html'),files:[...manifest.files,{name:'manifest.json',path:path.join(directory,'manifest.json'),kind:'json'}],results:raw.results.map((r:Record<string,unknown>)=>({...r,tables:Object.fromEntries(Object.entries(r.tables as Record<string,unknown[]>??{}).map(([k,v])=>[k,{rows:v.slice(0,5),totalRows:v.length}]))}))});
        context.save();
      }
      if(execution.status==='running'&&!workerAlive&&!s.pendingStatisticalInterpretation){execution.status='failed';execution.error='原统计进程未在运行；保留分步记录，未完成批次需要新计划和确认。';context.save();}
      return {execution:s.statisticalExecution,progress,workerAlive,completedSteps,report:s.statisticalReports?.find(r=>r.analysisId===execution.analysisId)??null,instruction:'Completed step files are checkpoints, not a complete report. Do not automatically rerun an interrupted or failed approval. Finished reports may be interpreted without recomputation.'};
    }
    if(name==='statistics_methods')return this.worker.call('statistics.methods',args,context.signal);
    if(name==='statistics_inspect')return this.worker.call('statistics.inspect',args,context.signal);
    if(name==='statistics_results'){
      if(s.pendingStatisticalInterpretation){if(args.analysisId&&args.analysisId!==s.pendingStatisticalInterpretation)throw new Error('Interpret only the currently approved statistical batch');args={...args,analysisId:s.pendingStatisticalInterpretation};}
      const report=args.analysisId?s.statisticalReports?.find(r=>r.analysisId===args.analysisId):s.statisticalReports?.filter(r=>r.runId===key).reverse()[0];
      if(!report)throw new Error('当前研究没有已交付的统计结果');
      const raw=JSON.parse(readFileSync(path.join(path.dirname(report.reportPath),'results.json'),'utf8')) as {results:Array<Record<string,unknown>>};
      if(args.step===undefined)return {...report,results:raw.results.map((r,i)=>({step:i,method:r.method,name:r.name,metrics:r.metrics,sample:r.sample,warnings:r.warnings,tables:Object.keys(r.tables as object??{})}))};
      const result=raw.results[Number(args.step)];if(!result)throw new Error('Unknown analysis step');
      if(!args.table)return {...result,tables:Object.entries(result.tables as Record<string,unknown[]>??{}).map(([table,rows])=>({table,totalRows:rows.length}))};
      const rows=(result.tables as Record<string,unknown[]>??{})[String(args.table)];if(!Array.isArray(rows))throw new Error('Unknown result table');
      const offset=Number(args.offset),limit=Number(args.limit);return {rows:rows.slice(offset,offset+limit),totalRows:rows.length,nextOffset:offset+limit<rows.length?offset+limit:null};
    }
    if(s.statisticalExecution?.status==='running')throw new Error('先用 statistics_status 核查当前计算，不能在状态未明时启动重复批次');
    if(s.pendingSynthesis||s.pendingStatisticalInterpretation)throw new Error('Finish the current interpretation before changing analysis plans');
    if(name==='statistics_plan'){
      if(args.datasetRef!==undefined&&!s.datasetRefs.includes(String(args.datasetRef)))throw new Error('Dataset is not attached to this session');
      const dataset=args.datasetRef===undefined?null:this.records.get<Dataset>('dataset',String(args.datasetRef));
      const preview=await this.worker.call('statistics.preview',{dataset,plan:args.plan},context.signal);
      if(s.pendingConfirmation){this.approvals.decide(s.pendingConfirmation.checkpointId,s.id,s.pendingConfirmation.contentHash,false);s.pendingConfirmation=undefined;}
      s.statisticalPlans??={};s.statisticalPlans[key]={datasetRef:dataset?.datasetRef,plan:args.plan as EmpiricalPlan};context.save();
      return {saved:true,preview,instruction:'Plan validated, not executed. Request the real confirmation card when ready. Revisions invalidate previous approval.'};
    }
    if(name==='statistics_request_approval'){
      if(s.pendingConfirmation)return {needsUser:true,summary:s.pendingConfirmation.summary,instruction:'An existing action awaits a human decision. Do not replace it implicitly.'};
      const saved=s.statisticalPlans?.[key];if(!saved)throw new Error('先保存明确的实证分析计划');
      const dataset=saved.datasetRef?this.records.get<Dataset>('dataset',saved.datasetRef):null;
      const preview=await this.worker.call('statistics.preview',{dataset,plan:saved.plan},context.signal);
      const payload={analysisId:`stats-${randomUUID()}`,studyKey:key,dataset,plan:saved.plan,preview};
      const summary=`确认执行统计分析？\n问题：${saved.plan.question}\n数据：${dataset?.fileName??'无需观测数据（使用已声明的数学参数）'}\n方法：${saved.plan.steps.map(step=>step.method).join('、')}\n完整规格：${JSON.stringify(saved.plan.steps)}\n缺失值规则：${saved.plan.steps.map(step=>`${step.method}: ${step.missing}`).join('；')}\n本地统计 worker，最长 120 秒；不调用外部服务、不下载模型。\n包含本批计算、论文表格（CSV/LaTeX/RTF）、图表、可复现脚本与结果解释。后续新实验需重新确认。`;
      const approval=this.approvals.request(s.id,{action:'statistics.execute',target:'statistics-local',payload,summary});
      s.pendingConfirmation={kind:'action',runId:s.runId??s.id,checkpointId:approval.id,contentHash:approval.hash,summary};context.save();return {needsUser:true,summary};
    }
    throw new Error('Unknown statistical tool');
  }
  async approve(s:ProductSession,signal?:AbortSignal,save:()=>void=()=>{}):Promise<unknown>{
    const pending=s.pendingConfirmation!;const effect=this.approvals.get(pending.checkpointId);
    const payload=effect.payload as {analysisId:string;studyKey:string;dataset:Dataset|null;plan:EmpiricalPlan;preview:unknown};
    if(payload.studyKey!==notebookKey(s)||pending.runId!==(s.runId??s.id)||(payload.dataset&&!s.datasetRefs.includes(payload.dataset.datasetRef))||!this.available('statistics_request_approval',{...s,pendingConfirmation:undefined}))throw new Error('Study, dataset or analysis mode changed; request a new approval');
    const saved=s.statisticalPlans?.[payload.studyKey];
    if(!saved||contentHash(saved)!==contentHash({datasetRef:payload.dataset?.datasetRef,plan:payload.plan}))throw new Error('Statistical plan changed');
    const preview=await this.worker.call('statistics.preview',{dataset:payload.dataset,plan:payload.plan},signal);
    if(contentHash(preview)!==contentHash(payload.preview))throw new Error('Dataset or worker environment changed; request a new approval');
    const receipt=this.approvals.decide(effect.id,s.id,pending.contentHash,true);this.approvals.assert(receipt,effect.action,'statistics-local',payload);s.pendingConfirmation=undefined;
    s.statisticalExecution={analysisId:payload.analysisId,approvalId:effect.id,studyKey:payload.studyKey,status:'running',startedAt:new Date().toISOString()};save();
    try {
      const result=await this.worker.call<StatisticalReport>('statistics.execute',{...payload,home:this.home,authorization:receipt},signal);
      this.publish(s,{...result,runId:payload.studyKey});save();return result;
    } catch(error){s.statisticalExecution.status='failed';s.statisticalExecution.error=error instanceof Error?error.message:String(error);save();throw error;}
  }
  private publish(s:ProductSession,result:StatisticalReport){
    s.statisticalReports=[...(s.statisticalReports??[]).filter(r=>r.analysisId!==result.analysisId),result].slice(-30);
    if(s.statisticalExecution)s.statisticalExecution.status='complete';
    s.pendingStatisticalInterpretation = result.analysisId;
    s.pendingArtifacts=[...(s.pendingArtifacts??[]),...result.files.filter(f=>['report.html','report.md','esttab.csv','esttab.tex','esttab.rtf','results.json','reproduce.py','manifest.json'].includes(f.name)||f.kind==='pdf').map(f=>({name:f.name,path:f.path}))];
  }
}
