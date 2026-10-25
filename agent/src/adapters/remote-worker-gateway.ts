import { openAsBlob, mkdirSync, existsSync, readFileSync, createWriteStream } from 'node:fs';
import { rm, rename } from 'node:fs/promises';
import { createHash, randomUUID } from 'node:crypto';
import { Readable, Transform } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import path from 'node:path';
import { ResearchStore } from '../memory/research-store.js';
import { EffectApprovals, type ApprovalReceipt } from '../domain/effect-approval.js';
import { contentHash, type ComputeJob, type Dataset, type TrainingPlan } from '../domain/research.js';
import { LocalComputeGateway, type ComputeGateway, type ComputeRequest, type ResultView } from './compute-gateway.js';
import type { CapabilityWorker } from './python-worker.js';
import { workerConfiguration, safeWorkerUrl, type WorkerConfiguration } from './worker-configuration.js';
class WorkerApiError extends Error { constructor(readonly status:number,message:string){super(message);} }
interface RemoteJob { config:WorkerConfiguration; projectId:string; remoteId?:string; requestHash:string; request:ComputeRequest; report?:any }
interface RemoteFile { name:string; url:string; sha256:string; sizeBytes:number; kind:string }
const metadata=(d:Dataset)=>({datasetRef:d.datasetRef,sha256:d.sha256,fileName:d.fileName,sizeBytes:d.sizeBytes});
function portablePlan(plan:TrainingPlan): any {
  const value=structuredClone(plan) as any;
  if(value.dataSplit?.sources) for(const source of Object.values(value.dataSplit.sources) as any[])source.dataset=metadata(source.dataset);
  if(Object.keys(value.params).some(k=>/model_path|api_key(?!_env)|token|password/i.test(k)))throw new Error('远程训练不能携带本机模型路径或密钥；请在计算服务配置资源');
  return value;
}
/** Only training crosses this boundary. Dataset previews, conversations and credentials stay local. */
export class ConfiguredComputeGateway implements ComputeGateway {
  private local:LocalComputeGateway; private store:ResearchStore; private config:ReturnType<typeof workerConfiguration>;
  private downloads=new Map<string,Promise<void>>(); private downloadErrors=new Map<string,string>();
  constructor(private home:string, private worker:CapabilityWorker,private fetcher:typeof fetch=fetch) {
    this.local=new LocalComputeGateway(home,worker);this.store=new ResearchStore(home);this.config=workerConfiguration(home);
  }
  private mapping(id:string):RemoteJob|undefined { try{return this.store.get('worker-job',id)}catch(e){if(e instanceof Error&&e.message.startsWith('记录不存在'))return;throw e;} }
  private async api(config:WorkerConfiguration,route:string,body?:unknown,method?:string):Promise<any> {
    const r=await this.fetcher(config.url+route,{method:method??(body===undefined?'GET':'POST'),redirect:'error',signal:AbortSignal.timeout(300000),headers:{authorization:`Bearer ${this.config.token(config)}`,...(body instanceof FormData?{}:{'content-type':'application/json'})},body:body===undefined?undefined:body instanceof FormData?body:JSON.stringify(body)}).catch(()=>{throw new Error('无法连接远程计算服务，请检查服务地址、网络和服务运行状态；未自动重试训练提交');});
    const value=await r.json();if(!r.ok)throw new WorkerApiError(r.status,`远程计算 HTTP ${r.status}：${String(value.error?.message??value.error??'请求失败').slice(0,1000)}`);return value;
  }
  isRemote(){return this.config.read().mode==='remote';}
  assertSelection(fingerprint?:string){if(this.isRemote() && fingerprint!==this.config.fingerprint(this.config.read()))throw new Error('远程计算配置已变化，请重新打开训练配置并确认上传与费用上限');}
  async checkRuntime(input:any){if(!this.isRemote())return this.worker.call('runtime.check',input);const catalog=await this.catalog();return {ready:catalog.models.includes(input.modelId),issues:catalog.models.includes(input.modelId)?[]:['远程服务未提供此模型'],compute:'remote',note:'实际资源和费用在提交前检查'};}
  async runtime(){const runtime=await this.worker.call<any>('runtime.config',{});return {...runtime,compute:this.config.publicValue()};}
  async catalog(){const c=this.config.read();if(c.mode!=='remote')return {protocol:'theta-worker-v1',mode:'local',models:[]};const v=await this.api(c,'/v1/capabilities');if(v.protocol!=='theta-worker-v1'||!Array.isArray(v.models))throw new Error('该服务未实现 THETA Worker v1 协议');return v;}
  async preview(input:{dataset:Dataset;plan:TrainingPlan}):Promise<any>{
    const c=this.config.read();if(c.mode==='local')return this.worker.call('compute.preview',input);
    const catalog=await this.catalog();if(!catalog.models.includes(input.plan.modelId))throw new Error('远程服务不支持模型 '+input.plan.modelId);
    const plan=portablePlan(input.plan);
    const value=await this.api(c,'/v1/preview',{dataset:metadata(input.dataset),plan});
    if(!value.readiness||typeof value.readiness.ready!=='boolean')throw new Error('远程服务缺少资源检查结果');
    const quote=value.quote;
    if(!quote||!Number.isFinite(quote.maxCost)||quote.maxCost<0||quote.currency!==c.currency||quote.maxCost>c.maxCost||!quote.id||!Number.isFinite(Date.parse(quote.expiresAt))||Date.parse(quote.expiresAt)<=Date.now())throw new Error('远程报价无效、已过期或超过设置中的单任务费用上限');
    return {...value,execution:{remoteWorker:{fingerprint:this.config.fingerprint(c),quote},embedding:{mode:input.plan.params.embedding_provider==='cloud'?'cloud':'local'},remoteEndpoint:c.url},rowCount:value.rowCount??0};
  }
  private async upload(c:WorkerConfiguration,projectId:string,d:Dataset){
    const digest=createHash('sha256');const blob=await openAsBlob(d.managedPath);for await(const chunk of Readable.fromWeb(blob.stream() as any))digest.update(chunk);
    if(digest.digest('hex')!==d.sha256)throw new Error('本地数据已变更，请重新确认训练');
    const form=new FormData();form.append('file',blob,d.fileName);form.append('projectId',projectId);form.append('sha256',d.sha256);
    const value=await this.api(c,'/v1/datasets',form);
    if(!value.datasetId||value.sha256!==d.sha256||value.projectId!==projectId)throw new Error('远程数据校验或项目归属不一致');
    return {datasetId:value.datasetId,sha256:d.sha256};
  }
  async submit(request:ComputeRequest,receipt?:ApprovalReceipt):Promise<ComputeJob>{
    if(!/^job-[a-f0-9]{64}$/.test(request.jobId))throw new Error('无效任务标识');
    const remote=request.execution?.remoteWorker as any;
    if(!remote)return this.local.submit(request,receipt);
    new EffectApprovals(this.store).assert(receipt,'compute.submit','local',request);
    const c=this.config.read();if(c.mode!=='remote'||remote.fingerprint!==this.config.fingerprint(c)||Date.parse(remote.quote.expiresAt)<=Date.now())throw new Error('计算配置或报价已变化，请重新确认');
    const prior=this.mapping(request.jobId);if(prior){if(prior.requestHash!==contentHash(request))throw new Error('任务内容变化');if(prior.remoteId)return this.status(request.jobId);}
    if(!request.projectId)throw new Error('远程训练缺少项目归属');
    // Independent installations may have identically named manual projects.
    this.store.putIfAbsent('worker-namespace','home',{id:randomUUID()});
    const namespace=this.store.get<{id:string}>('worker-namespace','home').id;
    const projectId='theta-'+contentHash({namespace,project:request.projectId});
    const value:RemoteJob=prior??{config:c,projectId,requestHash:contentHash(request),request};
    if(!prior&&!this.store.putIfAbsent('worker-job',request.jobId,value))throw new Error('任务正在提交，请查询状态');
    const dataset=await this.upload(c,projectId,request.dataset);const plan=portablePlan(request.plan);
    if(plan.dataSplit?.sources)for(const [key,source] of Object.entries(request.plan.dataSplit!.sources!))plan.dataSplit.sources[key]={...plan.dataSplit.sources[key],dataset:await this.upload(c,projectId,source.dataset)};
    let job;
    try { job=await this.api(c,'/v1/jobs',{clientJobId:request.jobId,idempotencyKey:request.jobId,projectId,dataset,plan,quoteId:remote.quote.id,maxCost:remote.quote.maxCost,currency:c.currency}); } catch(error) {
      if(error instanceof WorkerApiError && error.status<500)throw error;
      return {id:request.jobId,status:'queued',phase:'confirming_submission',percent:0,message:'提交响应中断，正在核对原任务；不会自动重复训练或计费'};
    }
    if(!job.id||job.projectId!==projectId)throw new Error('远程任务归属无效；请使用原任务 ID 查询，勿重复计费');
    value.remoteId=String(job.id);this.store.put('worker-job',request.jobId,value);return {id:request.jobId,status:'queued',phase:'queued',percent:0};
  }
  private route(m:RemoteJob,suffix=''){return `/v1/projects/${encodeURIComponent(m.projectId)}/jobs/${encodeURIComponent(m.remoteId!)}`+suffix;}
  async status(id:string):Promise<ComputeJob>{
    const m=this.mapping(id);if(!m)return this.local.status(id);
    if(m.report&&existsSync(m.report.resultDir))return {id,status:'completed',phase:'completed',percent:100,resultDir:m.report.resultDir,resultHash:m.report.resultHash};
    if(!m.remoteId){
      let found;
      try {found=await this.api(m.config,`/v1/projects/${encodeURIComponent(m.projectId)}/jobs/by-client-id/${encodeURIComponent(id)}`);}
      catch(error){return {id,status:'queued',phase:'confirming_submission',percent:0,message:'原任务提交结果尚待服务方核对，不会重复提交',error:error instanceof Error?error.message:String(error)};}
      if(!found.id||found.projectId!==m.projectId)throw new Error('远程提交结果尚待核对');m.remoteId=String(found.id);this.store.put('worker-job',id,m);
    }
    const s=await this.api(m.config,this.route(m));
    if(String(s.id)!==m.remoteId||s.projectId!==m.projectId||!['queued','running','completed','failed','cancelled'].includes(s.status))throw new Error('远程任务状态或项目归属无效');
    const state:ComputeJob={id,phaseHistory:s.phaseHistory,diagnostics:s.diagnostics,status:s.status,phase:String(s.phase??s.status),percent:Math.max(0,Math.min(99,Number(s.percent)||0)),error:s.error,message:s.message,telemetry:s.telemetry};
    if(s.status==='completed'){
      if(m.report&&existsSync(m.report.resultDir))return {...state,percent:100,resultDir:m.report.resultDir,resultHash:m.report.resultHash};
      if(this.downloadErrors.has(id))return {...state,status:'failed',phase:'result_download_failed',error:this.downloadErrors.get(id)};
      if(!this.downloads.has(id)){const work=this.hydrate(id,m).catch(e=>{this.downloadErrors.set(id,String(e instanceof Error?e.message:e))}).finally(()=>this.downloads.delete(id));this.downloads.set(id,work);}
      return {...state,status:'running',phase:'downloading_results',message:'训练已结束，正在接收并校验结果与图表',percent:99};
    }
    return state;
  }
  async cancel(id:string){const m=this.mapping(id);if(!m)return this.local.cancel(id);if(!m.remoteId)await this.status(id);const current=this.mapping(id)!;if(!current.remoteId)throw new Error('原任务提交结果尚待核对，请先恢复连接或联系计算服务，暂未发送取消请求');await this.api(current.config,this.route(current,'/cancel'),{});return this.status(id);}
  cachedStatus(_id:string):ComputeJob|undefined {return undefined;}
  private async hydrate(id:string,m:RemoteJob){
    const manifest=await this.api(m.config,this.route(m,'/results'));
    if(manifest.schema!=='theta-results-v1'||manifest.projectId!==m.projectId||manifest.jobId!==m.remoteId||manifest.status!=='ready'||!Array.isArray(manifest.files)||!manifest.files.length)throw new Error('完整结果尚未就绪或清单归属不一致');
    const names=new Set<string>();
    for(const f of manifest.files as RemoteFile[]){if(typeof f.name!=='string'||/[\x00-\x1f<>"|?*]/.test(f.name)||f.name.length>240||f.name.split('/').some(p=>/[. ]$/.test(p)||/^(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)/i.test(p))||f.name.includes('\\')||f.name.includes(':')||f.name.split('/').some(p=>!p||p==='.'||p==='..')||[...names].some(name=>name.normalize('NFC').toLowerCase()===f.name.normalize('NFC').toLowerCase())||!['figure','table','matrix','report','artifact'].includes(f.kind)||!Number.isSafeInteger(f.sizeBytes)||f.sizeBytes<0||!/^[a-f0-9]{64}$/.test(f.sha256))throw new Error('结果清单包含无效路径或校验信息');names.add(f.name);const artifactUrl=new URL(f.url);const base=new URL(f.url);base.search='';safeWorkerUrl(base.href);if(artifactUrl.hash)throw new Error('结果链接包含无效片段');}
    if([...names].filter(name=>/(^|\/)split_results\.json$/.test(name)).length!==1||!names.has('index.html'))throw new Error('结果缺少数据划分清单或完整图表报告');
    const root=path.join(this.home,'compute',id,'delivery');const stage=root+'-'+randomUUID();mkdirSync(stage,{recursive:true});
    try{
      for(const f of manifest.files as RemoteFile[]){
        const target=path.join(stage,f.name);mkdirSync(path.dirname(target),{recursive:true});
        const url=new URL(f.url);const sameOrigin=url.origin===new URL(m.config.url).origin;
        const r=await this.fetcher(url,{redirect:'error',signal:AbortSignal.timeout(300000),headers:sameOrigin?{authorization:`Bearer ${this.config.token(m.config)}`}:{}});if(!r.ok||!r.body)throw new Error(`结果下载失败：${f.name}`);
        let bytes=0;const digest=createHash('sha256');await pipeline(Readable.fromWeb(r.body as any),new Transform({transform(chunk,_enc,done){bytes+=chunk.length;digest.update(chunk);done(bytes>f.sizeBytes?new Error('结果大小超出清单'):null,chunk)}}),createWriteStream(target,{flags:'wx',mode:0o600}));
        if(bytes!==f.sizeBytes||digest.digest('hex')!==f.sha256)throw new Error(`结果校验失败：${f.name}`);
      }
      const splits=JSON.parse(readFileSync(path.join(stage,[...names].find(name=>/(^|\/)split_results\.json$/.test(name))!),'utf8'));
      if(!splits?.groups?.all||!splits.groups.train||!splits.groups.validation||!splits.groups.test)throw new Error('数据划分清单无效');
      await rm(root,{recursive:true,force:true});await rename(stage,root);
      const files=manifest.files.map((f:RemoteFile)=>({...f,path:path.join(root,f.name)}));
      m.report={modelId:m.request.plan.modelId,quality:manifest.quality,summary:manifest.summary,diagnostics:manifest.diagnostics,jobId:id,reportStatus:'complete',reportPath:path.join(root,'index.html'),resultDir:root,resultHash:contentHash(manifest.files),files,trainingPlan:m.request.plan,sourceData:manifest.sourceData,evidence:this.localEvidence(manifest.evidence??{},files),missingEvidence:[]};
      m.report=await this.worker.call('compute.receive',{home:this.home,request:m.request,report:m.report});
      this.store.put('worker-job',id,m);
    }catch(e){await rm(stage,{recursive:true,force:true});throw e;}
  }
  private localEvidence(value:any,files:any[]):any {
    if(Array.isArray(value))return value.map(item=>this.localEvidence(item,files));
    if(value&&typeof value==='object')return Object.fromEntries(Object.entries(value).map(([key,item])=>[key,key==='path'&&typeof item==='string' ? files.find(f=>f.name===item)?.path ?? '' : this.localEvidence(item,files)]));
    return value;
  }
  async results(id:string,view:ResultView,offset=0):Promise<any>{
    let m=this.mapping(id);if(!m)return this.local.results(id,view,offset);
    if(!m.report){await this.status(id);await this.downloads.get(id);m=this.mapping(id)!;}
    if(!m.report)throw new Error(this.downloadErrors.get(id)??'结果仍在准备中');
    return this.local.results(id,view,offset);
  }
}
