import path from 'node:path';
import { BundledSkills, skillTools } from './bundled-skills.js';
import { KnowledgeBase } from '../knowledge/knowledge-base.js';
import { mkdirSync, writeFileSync } from 'node:fs';
import { randomUUID } from 'node:crypto';
import type { InferenceProvider } from '../providers/types.js';
import { contentHash, type Dataset, type ResearchRun, type TrainingPlan, type ComputeJob } from '../domain/research.js';
import { ResearchStore } from '../memory/research-store.js';
import { notebookKey, type AnalysisNotebook, type ProductSession } from '../memory/session-store.js';
import { PythonCapabilityWorker, type CapabilityWorker } from '../adapters/python-worker.js';
import { LocalComputeGateway, type ComputeGateway } from '../adapters/compute-gateway.js';
import { GoComputeGateway, goConfigurationFingerprint } from '../adapters/go-compute-gateway.js';
import { BusinessApiComputeGateway, businessApiConfigurationFingerprint } from '../adapters/business-api-gateway.js';
import { EffectApprovals } from '../domain/effect-approval.js';
import type { ComputeRequest, ResultView } from '../adapters/compute-gateway.js';
import { productTools, trainingPlanSchema } from './tool-catalog.js';
import { TrainingBatches } from './training-batches.js';
import { ResearchContexts } from '../memory/research-context.js';
import { analysisReady, executeMining, miningTools } from './mining-tools.js';
import type { AnalysisRuntime } from '../adapters/analysis-runtime.js';
import { StatisticalTools, statisticsTools } from './statistics-tools.js';

export interface ProductToolContext { session: ProductSession; userMessage: string; signal?: AbortSignal; save(): void }
export interface ProductToolExecutor {
  execute(name: string, input: unknown, context: ProductToolContext): Promise<unknown>;
  readContext?(session: ProductSession): unknown;
  readState?(session: ProductSession): unknown;
  knowledgeCatalog?(): unknown;
  toolAvailable?(name: string, session: ProductSession): boolean;
  analysisProgress?(session: ProductSession): unknown;
  decorateAnswer?(session: ProductSession, answer: string): string;
  recordUnderstanding?(session: ProductSession, answer: string): void;
}

export type ResultSelectionRequest = { mode: 'manual'; jobIds: string[] }
  | { mode: 'automatic'; strategy: 'latest_completed' | 'all_completed' };

interface FigureCitationTarget { jobId: string; figure: string; chartName?: string }

/** Parse the host-owned result references appended to the current user turn. */
function figureCitationTargets(message: string): FigureCitationTarget[] {
  const targets: FigureCitationTarget[] = [];
  for (const match of message.matchAll(/\[\[cite\]\]([\s\S]*?)\[\[\/cite\]\]/gu)) {
    const raw = match[1].trim();
    try {
      const parsed: unknown = JSON.parse(raw);
      const values = Array.isArray(parsed) ? parsed : [parsed];
      for (const value of values) {
        if (!value || typeof value !== 'object') continue;
        const item = value as Record<string, unknown>;
        if (typeof item.jobId === 'string' && item.jobId && typeof item.figure === 'string' && item.figure) {
          targets.push({ jobId: item.jobId, figure: item.figure, ...(typeof item.chartName === 'string' ? { chartName: item.chartName } : {}) });
        }
      }
      continue;
    } catch { /* Accept citations written by older clients below. */ }
    for (const item of raw.split('；')) {
      const legacy = /^任务\s+(\S+)\s+(.+)$/u.exec(item.trim());
      if (legacy) targets.push({ jobId: legacy[1], figure: legacy[2] });
    }
  }
  return targets;
}

/** Research domain orchestration. No CLI imports, workflow FSM or training process code. */
export class LocalProductTools implements ProductToolExecutor {
  private readonly records: ResearchStore;
  private readonly worker: CapabilityWorker;
  private readonly compute: ComputeGateway;
  private readonly backend: string;
  private readonly approvals: EffectApprovals;
  private readonly contexts: ResearchContexts;
  private readonly statistics: StatisticalTools;
  private readonly batches: TrainingBatches;
  private readonly knowledge = new KnowledgeBase();
  /** Last observed job snapshot per job id; terminal jobs are never re-queried. */
  private readonly statusCache = new Map<string, ComputeJob>();
  private static readonly TERMINAL = new Set(['completed', 'failed', 'cancelled']);
  knowledgeCatalog(): unknown { return this.knowledge.catalog(); }
  toolAvailable(name: string, session: ProductSession): boolean {
    if (statisticsTools.some(tool=>tool.name===name)) return this.statistics.available(name,session);
    return !miningTools.some(tool=>tool.name===name) || (!!this.options.analysis && analysisReady(session));
  }
  analysisProgress(session: ProductSession): unknown {
    if (!this.options.analysis || !analysisReady(session)) return null;
    try {
      const plan=session.miningPlans?.[notebookKey(session)] ?? null;
      const workspace=this.options.analysis.inspect(session) as {delivered?:boolean;deliveredFiles?:Array<{path:string}>};
      if (plan && !plan.requiredFiles.every(name=>workspace.deliveredFiles?.some(file=>file.path===name))) workspace.delivered=false;
      return {plan,workspace,methodReview:session.methodReviews?.[notebookKey(session)] ?? null};
    }
    catch (error) { return {unavailable:error instanceof Error ? error.message : String(error)}; }
  }
  constructor(readonly options: { runtimeDb: string; uploadDir: string; inferenceFactory?: () => InferenceProvider | undefined; worker?: CapabilityWorker; compute?: ComputeGateway; analysis?: AnalysisRuntime; onJobObserved?: (job: ComputeJob) => void; onDatasetDerived?: (dataset: Dataset, source: Dataset, session: ProductSession) => void }) {
    const home = path.dirname(options.runtimeDb);
    this.records = new ResearchStore(home);
    this.contexts = new ResearchContexts(this.records);
    this.approvals = new EffectApprovals(this.records);
    this.worker = options.worker ?? new PythonCapabilityWorker();
    this.statistics = new StatisticalTools(this.records,this.worker,this.approvals,home);
    const businessUrl = (process.env.THETA_BUSINESS_API_URL ?? '').replace(/\/+$/u, '');
    this.backend = options.compute ? 'custom' : businessUrl || process.env.THETA_COMPUTE_URL || 'local';
    this.compute = options.compute ?? (businessUrl
      ? new BusinessApiComputeGateway(this.records, fetch, { baseUrl: businessUrl })
      : process.env.THETA_COMPUTE_URL ? new GoComputeGateway(process.env.THETA_COMPUTE_URL, this.records, fetch, home, this.worker) : new LocalComputeGateway(home, this.worker));
    this.batches = new TrainingBatches(this.records, this.compute, this.backend);
  }
  readContext(session: ProductSession): unknown {
    if (!session.contextId) return undefined;
    const { id, markdown, documentPath } = this.contexts.read(session.contextId);
    return { id, markdown, documentPath };
  }
  readState(session: ProductSession): unknown {
    if (!session.runId) return null;
    const run = this.run(session);
    return { goal: run.goal, plan: run.plan, lastObservedJob: this.batches.view(session)?.job ?? run.lastObservedJob ?? null,
      datasetRef: run.datasetRef,
      instruction: 'Last observed host state overrides historical dialogue; use run_status if fresh status is needed. Do not claim an old running job is still running after observed completion.' };
  }
  decorateAnswer(session: ProductSession, answer: string): string {
    const artifacts = session.pendingArtifacts ?? [];
    session.pendingArtifacts = undefined;
    for (const item of artifacts) answer = answer.replaceAll(`](file://${item.path})`, `](<${item.path}>)`);
    const remaining = artifacts.filter(item => !answer.includes(`](${item.path})`) && !answer.includes(`](<${item.path}>)`));
    return remaining.length ? `${answer}\n\n本次可用文件：\n\n${remaining.map(item => `- [${item.name}](<${item.path}>)`).join('\n')}` : answer;
  }
  recordUnderstanding(session: ProductSession, answer: string): void {
    if (session.pendingStatisticalInterpretation) {
      const report = session.statisticalReports?.find(r=>r.analysisId===session.pendingStatisticalInterpretation);
      if (report) {
        const documentPath = path.join(path.dirname(report.reportPath), 'interpretation.md');
        writeFileSync(documentPath, `# 统计结果研究解读\n\n分析：${report.analysisId}\n\n以下为 Agent 对已计算证据的解释；估计与原始精度以 results.json 为准。\n\n${answer}\n`, {mode:0o600});
        report.files = [...report.files.filter(f=>f.name!=='interpretation.md'), {name:'interpretation.md',path:documentPath,kind:'md'}];
        session.pendingArtifacts = [...(session.pendingArtifacts ?? []), {name:'统计结果研究解读',path:documentPath}];
      }
      session.pendingStatisticalInterpretation = undefined;
    }
    if (session.pendingSynthesis) {
      const request = session.pendingSynthesis;
      const directory = path.join(path.dirname(this.options.runtimeDb), 'interpretations');
      mkdirSync(directory, { recursive: true });
      const documentPath = path.join(directory, `${request.id}.md`);
      writeFileSync(documentPath, `# 研究结果综合解读\n\n问题：${request.question}\n\n来源：${request.jobIds.join(', ')}\n\n上下文版本：${request.contextHash}\n\n${answer}\n`, { mode: 0o600 });
      session.interpretations ??= [];
      session.interpretations.push({ ...request, documentPath });
      session.pendingSynthesis = undefined;
    }
    if (!session.pendingUnderstanding) return;
    const previous = session.contextId ? this.contexts.read(session.contextId) : undefined;
    const pending = session.pendingUnderstanding;
    const value = this.contexts.save(session.id, { title: previous?.title ?? '数据理解与研究目标',
      goal: session.runId ? this.run(session).goal : previous?.goal ?? pending.goal, understanding: answer.slice(0, 14000),
      questions: previous?.questions ?? ['研究或分析目标、数据来源和代表性仍需结合用户反馈确认。'], datasetRefs: pending.datasetRefs }, session.contextId);
    session.contextId = value.id; session.pendingUnderstanding = undefined;
  }
  async attach(filePath: string, session: ProductSession): Promise<unknown> {
    const dataset = await this.worker.call<Dataset>('dataset.import', { filePath: path.resolve(filePath), uploadDir: this.options.uploadDir });
    return this.registerDataset(dataset, session);
  }
  private registerDataset(dataset: Dataset, session: ProductSession): unknown {
    this.records.put('dataset', dataset.datasetRef, dataset);
    if (!session.datasetRefs.includes(dataset.datasetRef)) session.datasetRefs.push(dataset.datasetRef);
    if (session.runId) {
      const run = this.run(session);
      if (!run.datasetRef) { run.datasetRef = dataset.datasetRef; this.saveRun(run); }
    }
    return { datasetRef: dataset.datasetRef, fileName: dataset.fileName, sizeBytes: dataset.sizeBytes, sha256: dataset.sha256 };
  }
  private run(session: ProductSession): ResearchRun {
    if (!session.runId) throw new Error('尚未选择研究任务。可以先讨论或读取数据，需要记录研究时调用 run_create。');
    return this.records.get('run', session.runId);
  }
  private saveRun(run: ResearchRun): void { this.records.put('run', run.id, run); }
  /**
   * One status lookup per job, reusing the local store and the last observation.
   * A single unavailable job must not fail the whole catalog: the UI keeps its
   * previous view instead of showing an empty drawer.
   */
  private async jobStatus(jobId: string): Promise<ComputeJob> {
    const known = this.statusCache.get(jobId);
    if (known && LocalProductTools.TERMINAL.has(String(known.status))) return known;
    const local = this.compute.cachedStatus?.(jobId);
    if (local) { this.statusCache.set(jobId, local); return local; }
    try {
      const job = await this.compute.status(jobId);
      this.statusCache.set(jobId, job);
      return job;
    } catch (error) {
      if (known) return known;
      throw error;
    }
  }
  async resultCatalog(session: ProductSession): Promise<{ results: Array<Record<string, unknown>>; selection: ProductSession['resultSelection'] | null }> {
    const runs = (session.runIds ?? (session.runId ? [session.runId] : [])).map(id => this.records.get<ResearchRun>('run', id));
    const entries = runs.flatMap(run => run.jobs.map((jobId, order) => ({ run, jobId, order })));
    const results = await Promise.all(entries.map(async ({run, jobId, order}) => {
      this.requireBackend(run);
      const job = await this.jobStatus(jobId).catch((): ComputeJob => {
        const cached = this.statusCache.get(jobId);
        if (cached) return cached;
        const observed = run.lastObservedJob;
        return { id: jobId, status: observed?.status ?? 'queued', phase: observed?.phase ?? '状态未知', percent: observed?.percent ?? 0 };
      });
      const report = session.reports?.find(item => item.jobId === jobId);
      return { jobId, runId: run.id, modelId: run.plan?.modelId ?? 'unknown', status: job.status, phase: job.phase,
        percent: job.percent, selected: session.resultSelection?.resolvedJobIds.includes(jobId) ?? false,
        reportStatus: report?.reportStatus === 'incomplete' ? 'incomplete' : report ? 'ready' : 'not_requested', order };
    }));
    return { results: results.map(({order: _order, ...item}) => item), selection: session.resultSelection ?? null };
  }
  async selectResults(session: ProductSession, input: ResultSelectionRequest): Promise<NonNullable<ProductSession['resultSelection']>> {
    if (!input || !['manual','automatic'].includes(input.mode)) throw new Error('结果选择模式无效');
    const catalog = await this.resultCatalog(session); const completed = catalog.results.filter(item => item.status === 'completed');
    let ids: string[];
    if (input.mode === 'manual') {
      if (!Array.isArray(input.jobIds) || input.jobIds.length < 1 || input.jobIds.length > 12 || new Set(input.jobIds).size !== input.jobIds.length || input.jobIds.some(id => typeof id !== 'string')) throw new Error('手动选择需要 1–12 个不重复的任务 ID');
      const completedIds = new Set(completed.map(item => String(item.jobId)));
      if (input.jobIds.some(id => !completedIds.has(id))) throw new Error('只能选择当前对话中已完成的训练结果');
      ids = [...input.jobIds];
    } else {
      if (!['latest_completed','all_completed'].includes(input.strategy)) throw new Error('自动选择策略无效');
      if (!completed.length) throw new Error('当前对话还没有已完成的训练结果');
      ids = input.strategy === 'latest_completed' ? [String(completed.at(-1)!.jobId)] : completed.slice(-12).map(item => String(item.jobId));
    }
    session.resultSelection = { mode: input.mode, ...(input.mode === 'automatic' ? { strategy: input.strategy } : {}), resolvedJobIds: ids, resolvedAt: new Date().toISOString() };
    return session.resultSelection;
  }
  private requireBackend(run: ResearchRun): void {
    if (run.computeBackend && run.computeBackend !== this.backend) throw new Error('本研究绑定了另一计算端点。请恢复原端点查询任务；不能把相同批准转到其他计算服务。');
  }
  /** 远端配置指纹：把批准绑定到端点、账号与模型映射；本地模式不需要。 */
  private configurationFingerprint(): string | undefined {
    if (this.backend === 'local' || this.backend === 'custom') return undefined;
    return process.env.THETA_BUSINESS_API_URL ? businessApiConfigurationFingerprint({ baseUrl: this.backend }) : goConfigurationFingerprint();
  }
  private dataset(session: ProductSession, ref?: string): Dataset {
    const id = ref ?? (session.runId ? this.run(session).datasetRef : undefined) ?? (session.datasetRefs.length === 1 ? session.datasetRefs[0] : undefined);
    if (!id || !session.datasetRefs.includes(id)) throw new Error('请先在本会话中提供数据文件。');
    return this.records.get('dataset', id);
  }
  private async preview(run: ResearchRun, session: ProductSession): Promise<{ request: ComputeRequest; summary: string; ready: boolean; readiness: { ready: boolean; issues?: string[] } }> {
    this.requireBackend(run);
    if (!run.plan || !run.planHash) throw new Error('请先提出具体模型、数据列和参数方案。');
    const dataset = this.dataset(session);
    if (this.needsEmbeddingChoice(session, run, run.plan)) throw new Error('请先选择本地或云端嵌入，再生成新的训练确认卡。');
    const preview = await this.worker.call<{ execution: Record<string, unknown>; readiness: { ready: boolean; issues?: string[] }; rowCount: number }>('compute.preview', { plan: run.plan, dataset });
    const remote = this.backend !== 'local' && this.backend !== 'custom';
    const request = { jobId: `job-${run.planHash}`, runId: run.id, dataset, plan: run.plan, execution: { ...preview.execution, ...(remote ? { computeConfigurationFingerprint: this.configurationFingerprint() } : {}) } };
    const embedding = preview.execution.embedding as { mode: string; endpoint?: string; model?: string };
    const runtime = preview.execution.runtime as { profile: string; revision: string; python: string; mode: string } | undefined;
    const environment = !remote && runtime ? `\n运行环境：${runtime.profile} · ${runtime.revision}（${runtime.mode === 'shared' ? '共享兼容环境' : '独立配置'}）\nPython：${runtime.python}` : '';
    const external = embedding.mode === 'cloud'
      ? `\n外部 embedding：${embedding.endpoint}\n外部模型：${embedding.model}\n将发送：选定文本列的全文与派生词表（${preview.rowCount} 行输入）。\n最多 ${preview.execution.maxExternalRequests} 次 HTTP 请求，包括失败请求；没有自动重试。\n费用按供应商计费，当前没有可靠金额报价。`
      : '\nEmbedding 使用本地资源；本操作不授权外部 embedding 或模型下载。';
    return { request, readiness: preview.readiness, ready: remote || preview.readiness.ready,
      summary: `允许执行这次${remote ? '远端' : `本地 ${run.plan.device ?? 'cpu'}`}计算？\n${this.planSummary(run, dataset)}${remote ? `\n计算端点：${this.backend}\n远端运行资源/费用以已登记 runtime 为准；确认包含该任务的状态查询和结果链接读取。` : ''}${environment}${external}\n这是一次具体操作授权，不授权后续新实验。` };
  }
  private requestEffect(session: ProductSession, run: ResearchRun, action: string, payload: unknown, summary: string): unknown {
    const approval = this.approvals.request(session.id, { action, target: this.backend, payload, summary });
    session.pendingConfirmation = { kind: 'action', runId: run.id, checkpointId: approval.id, contentHash: approval.hash, summary };
    return { needsUser: true, summary, instruction: '暂停当前工具调用，等待用户确认。用户也可以拒绝、修改或继续讨论，不要求完成固定步骤。' };
  }
  hasTrainingBatch(session: ProductSession): boolean { return this.batches.active(session); }
  trainingConfiguration(session: ProductSession): { plans: TrainingPlan[]; datasetName: string; datasetRef: string; goal: string } | undefined {
    const pending = session.pendingConfirmation;
    if (!pending || pending.runId !== session.runId) return;
    const plans = pending.kind === 'training_config' ? pending.trainingPlans
      : pending.kind === 'action' && this.approvals.get(pending.checkpointId).action === 'compute.submit'
        ? [(this.approvals.get(pending.checkpointId).payload as ComputeRequest).plan] : undefined;
    if (!plans?.length) return;
    const dataset = this.dataset(session);
    return { plans, datasetName: dataset.fileName, datasetRef: dataset.datasetRef, goal: this.run(session).goal };
  }
  async trainingEditor(session: ProductSession): Promise<unknown> {
    const configuration = this.trainingConfiguration(session);
    if (!configuration) throw new Error('当前没有可编辑的训练确认卡。');
    const [profile, runtime] = await Promise.all([
      this.worker.call<{ columns: string[] }>('dataset.profile', { dataset: this.dataset(session) }),
      this.worker.call('runtime.config', {}),
    ]);
    const supplied = configuration.plans.find(plan => typeof plan.params['text.stopwords'] === 'string')?.params['text.stopwords'];
    let stopwords: { id: string; name: string; count: number } | undefined;
    if (typeof supplied === 'string' && supplied.trim()) {
      const words = supplied.split(/\r?\n/u).map(word => word.trim()).filter(Boolean);
      const id = `stopwords-${contentHash({ sessionId: session.id, words })}`;
      stopwords = { id, name: 'Agent 预填停用词表', count: words.length };
      this.records.put('training-stopwords', id, { sessionId: session.id, words, name: stopwords.name });
    }
    return { ...configuration, inputKind: ['.txt', '.md', '.pdf', '.docx'].includes(path.extname(this.dataset(session).fileName).toLowerCase()) ? 'text' : 'table', datasetSizeBytes: this.dataset(session).sizeBytes, columns: profile.columns, runtime, stopwords };
  }
  private async proposeTrainingConfiguration(context: ProductToolContext, plans: TrainingPlan[]): Promise<unknown> {
    if (new Set(plans.map(plan => plan.modelId)).size !== plans.length) throw new Error('每个模型只需配置一次。');
    const run = this.run(context.session); const dataset = this.dataset(context.session);
    for (const plan of plans) await this.worker.call('plan.validate', { plan, dataset }, context.signal);
    const summary = `允许执行这次多模型分析？\n数据：${dataset.fileName}\n模型：${plans.map(plan => plan.modelId).join('、')}\n点击配置卡可调整模型与各自参数，最终确认后按顺序执行。\n${plans.map(plan => this.planSummary({ ...run, plan }, dataset)).join('\n\n')}\n云端嵌入会发送选定正文与派生词表并按供应商计费；未确认前不执行。`;
    context.session.pendingConfirmation = { kind: 'training_config', runId: run.id, checkpointId: `training-${randomUUID()}`,
      contentHash: contentHash({ runId: run.id, dataset, plans, backend: this.backend }), trainingPlans: plans, summary };
    context.save(); return { needsUser: true, summary, instruction: '等待用户在训练配置弹窗里修改并确认。不得替用户提交。' };
  }
  /** Final host/UI confirmation: validate ALL edited models before queuing ANY compute. */
  async submitTrainingConfiguration(session: ProductSession, input: {
    checkpointId: string; expectedContentHash: string; plans: unknown[]; cloudConfirmed?: boolean;
    cloudSelection?: { provider: string; model: string; endpoint: string }; stopwords?: string[];
  }, save: () => void, signal?: AbortSignal): Promise<unknown> {
    const batchId = `batch-${input.checkpointId}`; const configurationHash = contentHash(input);
    const existing = this.batches.get(batchId);
    if (existing) {
      if (existing.sessionId !== session.id || existing.sourceHash !== input.expectedContentHash || existing.configurationHash !== configurationHash) throw new Error('这张卡已提交，不能使用旧卡提交另一份配置。');
      const restored = this.batches.restore(session, batchId); save();
      if (restored) await this.batches.advance(session, save);
      return { reused: true, ...this.batches.view(session) };
    }
    const pending = session.pendingConfirmation; const configuration = this.trainingConfiguration(session);
    if (!pending || !configuration || pending.checkpointId !== input.checkpointId || pending.contentHash !== input.expectedContentHash) throw new Error('训练卡已变化，请重新查看当前方案。');
    const sourceRun = this.run(session); this.requireBackend(sourceRun);
    const dataset = this.dataset(session);
    if (pending.kind === 'action') {
      const effect = this.approvals.get(pending.checkpointId);
      if (effect.status !== 'pending' || effect.expiresAt < Date.now() || contentHash((effect.payload as ComputeRequest).dataset) !== contentHash(dataset)) throw new Error('训练确认已过期或数据变化，请重新生成卡片。');
    } else if (pending.contentHash !== contentHash({ runId: sourceRun.id, dataset, plans: pending.trainingPlans, backend: this.backend })) throw new Error('数据或方案已变化，请重新生成卡片。');
    if (!Array.isArray(input.plans) || input.plans.length < 1 || input.plans.length > 12) throw new Error('请选择 1–12 个模型。');
    const plans = input.plans.map(plan => trainingPlanSchema.parse(plan));
    if (new Set(plans.map(plan => plan.modelId)).size !== plans.length) throw new Error('模型不能重复。');
    const requests: ComputeRequest[] = [];
    for (const plan of plans) {
      // Credentials/endpoints are host configuration, never editable parameter overrides.
      for (const key of ['embedding_cloud_provider', 'embedding_model', 'embedding_api_base', 'embedding_api_key_env', 'text.stopwords']) delete plan.params[key];
      if (input.stopwords) plan.params['text.stopwords'] = input.stopwords.join('\n');
      const cloud = plan.modelId === 'theta' && plan.params.embedding_provider === 'cloud';
      if (cloud && (input.cloudConfirmed !== true || (plan.params.mode ?? 'zero_shot') !== 'zero_shot' || !input.cloudSelection)) throw new Error('云端嵌入仅支持 THETA 零样本，请明确勾选发送文本与费用确认。');
      const preview = await this.worker.call<{ execution: Record<string, unknown>; readiness: { ready: boolean; issues?: string[] } }>('compute.preview', { plan, dataset }, signal);
      const actual = preview.execution.embedding as { mode: string; provider?: string; endpoint?: string; model?: string };
      if (actual.mode !== (cloud ? 'cloud' : 'local')) throw new Error(`${plan.modelId} 的实际嵌入方式与选择不一致。`);
      if (cloud) {
        const shown = input.cloudSelection!;
        if (shown.provider !== actual.provider || shown.model !== actual.model || shown.endpoint.replace(/\/$/u, '').replace(/\/embeddings$/u, '') + '/embeddings' !== actual.endpoint) throw new Error('云端服务已变化，请重新打开配置并确认。');
      }
      if (['local', 'custom'].includes(this.backend) && !preview.readiness.ready) throw new Error(preview.readiness.issues?.join(" ") || `${plan.modelId.toUpperCase()} 的运行环境或模型资源未就绪；未启动任何模型，请调整配置后重试。`);
      // The execution contract uses runId as dataset.project_id (at most 36 chars).
      const runId = `run-${contentHash({ batchId, model: plan.modelId }).slice(0, 32)}`;
      requests.push({ runId, jobId: `job-${contentHash({ batchId, dataset, plan })}`, dataset, plan,
        execution: { ...preview.execution, ...(!['local', 'custom'].includes(this.backend) ? { computeConfigurationFingerprint: this.configurationFingerprint() } : {}) } });
    }
    signal?.throwIfAborted();
    if (pending.kind === 'action') this.approvals.decide(pending.checkpointId, session.id, pending.contentHash, false);
    this.batches.create(session, { id: batchId, sourceHash: input.expectedContentHash, configurationHash, requests, goal: sourceRun.goal });
    save();
    await this.batches.advance(session, save);
    return { batchId, ...this.batches.view(session), instruction: '用户已在配置弹窗确认了这些模型和最终参数。队列已持久化，按顺序执行；不能另行提交新任务。' };
  }
  private embeddingScope(session: ProductSession, run: ResearchRun, plan: TrainingPlan): string {
    return contentHash({ runId: run.id, dataset: this.dataset(session), plan, backend: this.backend });
  }
  private needsEmbeddingChoice(session: ProductSession, run: ResearchRun, plan: TrainingPlan): boolean {
    return plan.modelId === 'theta' && (plan.params.mode ?? 'zero_shot') === 'zero_shot'
      && session.embeddingSelections?.[run.id]?.scope !== this.embeddingScope(session, run, plan);
  }
  private async requestEmbeddingChoice(context: ProductToolContext, run: ResearchRun, plan: TrainingPlan): Promise<unknown> {
    const { session } = context;
    const config = await this.worker.call<{ embedding: NonNullable<NonNullable<ProductSession['pendingConfirmation']>['embeddingChoice']>['cloud'] }>('runtime.config', {}, context.signal);
    const cloud = config.embedding;
    const choice = { plan, datasetHash: this.dataset(session).sha256, backend: this.backend,
      cloud: { configured: cloud.configured === true && !cloud.configurationIssues?.length, provider: cloud.provider, endpoint: cloud.endpoint, model: cloud.model, configurationIssues: cloud.configurationIssues } };
    const summary = `请选择嵌入方式\nTHETA 零样本分析需要文本嵌入，请先选择计算方式，再查看训练确认卡。\n本地：Qwen3-Embedding ${plan.params.model_size ?? '0.6B'}，使用本机资源，不向云端发送嵌入文本；资源是否就绪会在下一步检查，不会自动下载。\n云端：${cloud.provider || '未配置'} · ${cloud.model || '未配置'}\n云端地址：${cloud.endpoint || '未配置'}\n${choice.cloud.configured ? '云端会发送选定正文和派生词表，按供应商计费。' : '云端配置未就绪，请选择本地或先完善服务配置。'}\n本次外部请求上限：${plan.externalRequestLimit ?? 100} 次。\n选择方式不会开始训练或发送嵌入请求；训练卡仍需单独确认。也可以回复“本地嵌入”或“云端嵌入”。`;
    session.pendingConfirmation = { kind: 'embedding_choice', runId: run.id, checkpointId: `embedding-${randomUUID()}`, contentHash: contentHash(choice), summary, embeddingChoice: choice };
    context.save();
    return { needsUser: true, summary, instruction: '等待用户选择本地或云端嵌入。模型不能代选；选择不等于训练授权。' };
  }
  /** Host-only preference decision. Never exposed as an inference tool or a compute approval. */
  async selectEmbedding(session: ProductSession, provider: 'local' | 'cloud', signal?: AbortSignal, save: () => void = () => {}): Promise<{ summary: string }> {
    const pending = session.pendingConfirmation;
    if (pending?.kind !== 'embedding_choice' || !pending.embeddingChoice) throw new Error('当前没有待选择的嵌入方式。');
    const choice = pending.embeddingChoice;
    const run = this.run(session);
    if (pending.runId !== run.id || choice.backend !== this.backend || choice.datasetHash !== this.dataset(session).sha256 || pending.contentHash !== contentHash(choice)) throw new Error('研究或数据已变化，请重新生成嵌入选择卡。');
    if (!['local', 'cloud'].includes(provider)) throw new Error('请选择本地或云端嵌入。');
    const plan = structuredClone(choice.plan);
    // The choice shown to the user is the configured service, not an LLM-supplied endpoint.
    for (const key of ['embedding_cloud_provider', 'embedding_model', 'embedding_api_base', 'embedding_api_key_env', 'embedding_dimensions']) delete plan.params[key];
    if (provider === 'cloud') {
      const { embedding } = await this.worker.call<{ embedding: typeof choice.cloud }>('runtime.config', {}, signal);
      if (!embedding.configured || embedding.configurationIssues?.length) throw new Error('云端嵌入配置未就绪，请选择本地或完善配置。');
      if (embedding.provider !== choice.cloud.provider || embedding.model !== choice.cloud.model || embedding.endpoint !== choice.cloud.endpoint) throw new Error('云端服务配置已变化，请重新生成选择卡。');
      plan.params.embedding_cloud_provider = embedding.provider;
      plan.params.embedding_model = embedding.model;
      plan.params.embedding_api_base = embedding.endpoint;
    }
    plan.params.embedding_provider = provider;
    await this.worker.call('plan.validate', { plan, dataset: this.dataset(session) }, signal);
    session.embeddingSelections ??= {};
    session.embeddingSelections[run.id] = { scope: this.embeddingScope(session, run, plan) };
    const context = { session, userMessage: provider === 'local' ? '本地嵌入' : '云端嵌入', signal, save };
    if (choice.plans) {
      const result = await this.proposeTrainingConfiguration(context, choice.plans.map(item => item.modelId === 'theta' ? plan : item)) as { summary: string };
      return { summary: result.summary };
    }
    await this.execute('plan_propose', plan, context);
    const result = await this.execute('training_advance', {}, context) as { summary?: string; readiness?: unknown };
    return { summary: result.summary ?? `已保存${provider === 'local' ? '本地' : '云端'}嵌入选择，尚未启动训练。计算环境尚未就绪：${JSON.stringify(result.readiness)}。修复环境后可重新请求训练确认卡。` };
  }
  private async resultScope(session: ProductSession, jobIds: string[]): Promise<unknown[]> {
    const runs = (session.runIds ?? [session.runId!]).map(id => this.records.get<ResearchRun>('run', id));
    return Promise.all(jobIds.map(async jobId => {
      const owner = runs.find(run => run.jobs.includes(jobId));
      if (!owner) throw new Error('只能读取本会话已关联任务的结果');
      this.requireBackend(owner);
      const job = await this.compute.status(jobId);
      if (job.status !== 'completed') throw new Error('任务尚未成功完成，不能解读结果');
      return { jobId, runId: owner.id, resultHash: job.resultHash ?? null, planHash: owner.planHash, datasetRef: owner.datasetRef };
    }));
  }
  private async analysisPayload(session: ProductSession, jobIds: string[], question?: string): Promise<Record<string, unknown>> {
    return { jobs: await this.resultScope(session, jobIds), jobIds,
      ...(question !== undefined ? { question, contextHash: contentHash(this.readContext(session) ?? null) } : {}) };
  }
  private planSummary(run: ResearchRun, dataset: Dataset): string {
    const plan = run.plan!;
    const remote = this.backend !== 'local' && this.backend !== 'custom';
    const seconds = plan.timeoutSeconds;
    const durationText = [[Math.floor(seconds / 3600), '小时'], [Math.floor(seconds % 3600 / 60), '分钟'], [seconds % 60, '秒']]
      .filter(([value]) => value).map(([value, unit]) => `${value} ${unit}`).join(' ');
    const duration = remote ? `请求时长：${durationText}；现有远端 API 不接受此上限，实际时限由已登记 runtime 决定。` : `最长运行：${durationText}（上限，不是预计耗时）`;
    return `研究目标：${run.goal}\n数据：${dataset.fileName}\n模型：${plan.modelId}\n文本列：${plan.textColumn}${plan.timeColumn ? `\n时间列：${plan.timeColumn}` : ''}${plan.labelColumn ? `\n标签列：${plan.labelColumn}` : ''}\n设备：${plan.device ?? 'cpu'}${plan.covariates?.length ? `\n协变量：${plan.covariates.join('、')}` : ''}\n参数：${JSON.stringify(plan.params)}\n理由：${plan.rationale}\n${duration}\n输入将规范化为独立 UTF-8 CSV，原文件保留。`;
  }
  /** Read-only observation can run while the conversation holds a session lease. */
  async observeTraining(session: ProductSession) {
    if (!session.runId) return undefined;
    const run = this.run(session); this.requireBackend(run);
    if (!run.activeJob || !run.jobs.includes(run.activeJob)) return undefined;
    return this.compute.status(run.activeJob);
  }
  private async job(run: ResearchRun, session: ProductSession, id = run.activeJob): Promise<unknown> {
    this.requireBackend(run);
    if (!id || !run.jobs.includes(id)) throw new Error('尚无关联的训练任务。');
    const job = await this.compute.status(id);
    run.lastObservedJob = { id: job.id, status: job.status, percent: job.percent, phase: job.phase, telemetry: job.telemetry, resultDir: job.resultDir, resultWarning: job.resultWarning }; this.saveRun(run);
    session.monitorTraining = ['queued', 'running'].includes(job.status);
    this.options.onJobObserved?.(job);
    const error = job.diagnostics?.available ? job.diagnostics.message : job.error;
    return { job, snapshot: { currentState: job.status, trainingPercent: job.percent }, summary: `训练${({ queued: '已排队', running: '进行中', completed: '已完成', failed: '失败', cancelled: '已取消' })[job.status]}${error ? `：${error}` : ''}${job.resultWarning ? `\n${job.resultWarning}` : ''}${job.resultDir ? `\n原始结果目录：${job.resultDir}` : ''}` };
  }
  async execute(name: string, input: unknown, context: ProductToolContext): Promise<unknown> {
    if (name === 'training_request_approval') name = 'training_advance';
    const tool = productTools.find((item) => item.name === name);
    if (!tool) throw new Error(`Unknown tool: ${name}`);
    const args = tool.schema.parse(input) as Record<string, unknown>;
    context.signal?.throwIfAborted();
    const session = context.session;
    if (skillTools.some(tool=>tool.name===name)) return new BundledSkills(path.dirname(this.options.runtimeDb)).execute(name,args,context);
    if (statisticsTools.some(tool=>tool.name===name)) return this.statistics.execute(name,args,context);
    if (miningTools.some(tool=>tool.name===name)) return executeMining(this.options.analysis,name,input,context,this.options.inferenceFactory?.());
    if (name === 'dataset_preprocess') {
      if (session.pendingConfirmation || session.pendingSynthesis || this.batches.active(session)) throw new Error('请先完成或撤回待确认方案；正在训练时不能切换数据。');
      const source = this.dataset(session, args.datasetRef as string | undefined);
      const previous = session.runId ? this.run(session) : undefined;
      if (previous?.lastObservedJob && ['running', 'queued'].includes(previous.lastObservedJob.status)) throw new Error('当前训练尚未完成，请勿切换数据。');
      const result = await this.worker.call<{ dataset: Dataset; summary: Record<string, unknown>; profile: Record<string, unknown>; artifacts: Array<{name:string;path:string}> }>('dataset.preprocess', {
        ...args, dataset: source, home: path.dirname(this.options.runtimeDb), uploadDir: this.options.uploadDir,
      }, context.signal);
      this.options.onDatasetDerived?.(result.dataset, source, session);
      this.registerDataset(result.dataset, session);
      const run: ResearchRun = { id: `run-${randomUUID()}`, goal: previous?.goal ?? session.lastUserIntent ?? String(args.purpose), datasetRef: result.dataset.datasetRef, profile: result.profile, jobs: [], notes: [`预处理来源：${source.datasetRef}`, String(args.purpose)] };
      this.saveRun(run);
      session.runIds ??= previous ? [previous.id] : [];
      session.runIds.push(run.id); session.runId = run.id;
      session.skillArtifacts = [...(session.skillArtifacts ?? []), ...result.artifacts];
      session.pendingArtifacts = [...(session.pendingArtifacts ?? []), ...result.artifacts];
      this.records.put('dataset-lineage', run.id, { ...result.summary, runId: run.id, artifacts: result.artifacts });
      context.save();
      return { ...result.summary, runId: run.id, profile: result.profile, artifacts: result.artifacts,
        instruction: '派生数据已保存并选为当前研究输入，原数据与历史任务不变。检查校验结果并说明删改行数，再根据派生列和模型实际参数生成 training_configure 确认卡；尚未启动训练。' };
    }
    if (name === 'analysis_history') {
      const receipts = session.messages.flatMap((message,index) => {
        const call = session.messages[index-1]?.metadata?.toolCalls as Array<{id:string;name:string;arguments:unknown}> | undefined;
        if (message.role !== 'tool' || !call?.[0] || call[0].id !== message.metadata?.toolCallId || call[0].name === 'analysis_history') return [];
        return [{callId:call[0].id, tool:call[0].name, arguments:call[0].arguments, receipt:message.content, runId:message.metadata?.runId ?? null}];
      }).reverse();
      const selected = args.callId ? receipts.filter(item=>item.callId===args.callId) : receipts.slice(Number(args.offset),Number(args.offset)+Number(args.limit));
      return {receipts:selected, nextOffset:!args.callId && Number(args.offset)+selected.length<receipts.length ? Number(args.offset)+selected.length : null,
        instruction:'These are historical original receipts, not new execution. A missing runId means older provenance is unspecified, not that it belongs to the selected study. Match file/data/job identity before use.'};
    }
    if (name === 'analysis_checkpoint') {
      if (session.pendingSynthesis) throw new Error('The approved interpretation is saved separately; do not change the research notebook during synthesis.');
      session.analysisNotebooks ??= {};
      const key = notebookKey(session);
      if (!session.analysisNotebooks[key] && Object.keys(session.analysisNotebooks).length >= 30) throw new Error('Notebook study limit reached; use a new session.');
      session.analysisNotebooks[key] = args as unknown as AnalysisNotebook;
      context.save();
      return {saved:true, instruction:'Agent-authored working notes saved, not independently verified. Now perform the recorded next action using available tools; do not repeat completed inspections. This receipt does not prove that any listed artifact exists or grant execution approval.'};
    }
    if (name === 'knowledge_list') return this.knowledge.list(args);
    if (name === 'knowledge_search') return this.knowledge.search(String(args.query), args.documentId as string | undefined, args.limit as number);
    if (name === 'knowledge_read') return this.knowledge.read(args as { documentId: string; sectionId?: string; offset?: number; limit?: number });
    if (name === 'reports_list') return [...(session.reports ?? []), ...(session.statisticalReports ?? []).map(({analysisId,runId,status,reportPath,files})=>({analysisId,runId,status,reportPath,files}))];
    if (name === 'runs_list') return (session.runIds ?? []).map(id => { const run = this.records.get<ResearchRun>('run', id); return { id, goal: run.goal, datasetRef: run.datasetRef, modelId: run.plan?.modelId, jobs: run.jobs }; });
    if (name === 'contexts_list') return this.contexts.list();
    if (name === 'context_read') return this.contexts.read(String(args.contextId ?? session.contextId ?? ''));
    if (name === 'context_select') {
      const value = this.contexts.select(session.id, String(args.contextId)); session.contextId = value.id; context.save();
      return { ...value, instruction: '上下文已复制到本会话，不会自动附加其中的数据，也不继承任何计算授权。' };
    }
    if (name === 'context_save') {
      if (session.pendingSynthesis) throw new Error('综合解读由宿主保存为独立 Markdown；本次解读不要覆盖数据理解上下文。');
      const value = this.contexts.save(session.id, { title: String(args.title), goal: String(args.goal),
        understanding: String(args.understanding), questions: args.questions as string[],
        datasetRefs: session.datasetRefs.length ? [...session.datasetRefs] : session.contextId ? this.contexts.read(session.contextId).datasetRefs : [] }, session.contextId);
      session.contextId = value.id; session.pendingUnderstanding = undefined; context.save(); return value;
    }
    if (name === 'datasets_discover') {
      const catalog = await this.worker.call<Record<string,unknown>>('dataset.discover', args, context.signal);
      const uploaded = this.records.list<Dataset>('dataset'); const offset = Number(args.offset ?? 0);
      return {...catalog, uploadedDatasets:uploaded.slice(offset,offset+50).map(d=>({catalogId:'catalog-'+contentHash({registeredDataset:d.datasetRef,sha256:d.sha256}),name:d.fileName,sizeBytes:d.sizeBytes,source:'managed upload'})),uploadedNextOffset:offset+50<uploaded.length?offset+50:null,
        instruction:'目录候选与已注册上传均可选择。用户明确指定文件后使用其 catalogId；不要因为文件不在预上传目录就断言上传失败。发现不会自动附加或计算。'};
    }
    if (name === 'dataset_use') {
      const uploaded = this.records.list<Dataset>('dataset').find(d=>'catalog-'+contentHash({registeredDataset:d.datasetRef,sha256:d.sha256})===args.catalogId);
      const dataset = uploaded ?? await this.worker.call<Dataset>('dataset.use', { ...args, uploadDir: this.options.uploadDir }, context.signal);
      const attached = this.registerDataset(dataset, session); context.save();
      return { attached, instruction: '用户选择的数据已附加；可用 dataset_understand 理解业务。若当前研究已绑定其他数据，先新建研究，不要静默替换已有实验数据。' };
    }
    if (name === 'dataset_understand') {
      const dataset = this.dataset(session, args.datasetRef as string | undefined);
      const result = await this.worker.call('dataset.understand', { ...args, dataset }, context.signal);
      session.pendingUnderstanding = { datasetRefs: [dataset.datasetRef], goal: session.runId ? this.run(session).goal : session.lastUserIntent || context.userMessage || '待确认用户研究或分析目标' };
      context.save(); return result;
    }
    if (name === 'models_list') return this.worker.call('models.list', {}, context.signal);
    if (name === 'models_inspect') return this.worker.call('models.inspect', args, context.signal);
    if (name === 'runtime_config') return this.worker.call('runtime.config', {}, context.signal);
    if (name === 'runtime_check') return this.worker.call('runtime.check', args, context.signal);
    if (name === 'dataset_read') return this.worker.call('dataset.profile', { ...args, dataset: this.dataset(session, args.datasetRef as string | undefined) }, context.signal);
    if (name === 'run_create') {
      const ref = args.datasetRef as string | undefined ?? (session.datasetRefs.length === 1 ? session.datasetRefs[0] : undefined);
      if (ref) this.dataset(session, ref);
      const run: ResearchRun = { id: `run-${randomUUID()}`, goal: String(args.goal ?? context.userMessage), datasetRef: ref, jobs: [], notes: [] };
      this.saveRun(run); session.runId = run.id; session.runIds ??= []; session.runIds.push(run.id);
      session.pendingConfirmation = undefined; session.monitorTraining = false; context.save();
      return run;
    }
    if (name === 'run_select') {
      if (!session.runIds?.includes(String(args.runId))) throw new Error('研究任务不属于当前会话');
      session.runId = String(args.runId); session.pendingConfirmation = undefined; session.monitorTraining = false; context.save();
      return this.run(session);
    }
    const run = this.run(session);
    if (name === 'training_configure') {
      const plans = args.plans as TrainingPlan[];
      const theta = plans.find(plan => plan.modelId === 'theta' && (plan.params.mode ?? 'zero_shot') === 'zero_shot');
      if (theta) {
        const result = await this.requestEmbeddingChoice(context, run, theta);
        const choice = session.pendingConfirmation!.embeddingChoice!; choice.plans = plans;
        session.pendingConfirmation!.contentHash = contentHash(choice); context.save(); return result;
      }
      return this.proposeTrainingConfiguration(context, plans);
    }
    if (name === 'run_update') {
      if (run.jobs.length) throw new Error('已有计算任务的研究保留原目标；新实验请 run_create，后续业务理解可单独更新 context_save。');
      if (session.pendingConfirmation) this.deny(session);
      run.goal = String(args.goal); run.notes.push(context.userMessage); this.saveRun(run);
      if (session.contextId) {
        const previous = this.contexts.read(session.contextId);
        this.contexts.save(session.id, { title: previous.title, goal: run.goal, understanding: previous.understanding,
          questions: previous.questions, datasetRefs: previous.datasetRefs }, session.contextId);
      }
      context.save(); return run;
    }
    if (name === 'run_status') {
      if (this.batches.view(session)) { await this.batches.advance(session, context.save); return this.batches.view(session); }
      const result = run.activeJob ? await this.job(run, session) : { goal: run.goal, plan: run.plan, pendingConfirmation: session.pendingConfirmation }; context.save(); return result;
    }
    if (name === 'research_read') return run;
    if (name === 'research_answer' || name === 'checkpoint_revise') {
      if (!context.userMessage.trim()) throw new Error('需要实际用户消息');
      run.notes.push(context.userMessage);
      if (name === 'checkpoint_revise') { if (session.pendingConfirmation) this.deny(session); }
      this.saveRun(run); context.save(); return { recorded: true, instruction: '根据用户反馈决定是否需要重新提案；这不是批准。', run };
    }
    if (name === 'research_continue') {
      run.profile = await this.worker.call('dataset.understand', { dataset: this.dataset(session) }, context.signal);
      session.pendingUnderstanding = { datasetRefs: [this.dataset(session).datasetRef], goal: run.goal }; context.save();
      this.saveRun(run); return { run, instruction: '结合内容证据与用户研究或业务目标解释：分析对象、关心的现象、可回答的问题和局限。区分观察/推测/待确认。首次理解后若用户尚未明确下一步，提供“快速描述性统计＋一个推荐模型”或“先明确目的”两种选择；用户选定快速分析且证据足够时准备具体训练方案与宿主确认卡。不要把格式检查当内容理解，也不要强制进入训练。' };
    }
    if (name === 'plan_propose') {
      if (run.jobs.length && run.plan?.modelId !== args.modelId) throw new Error('切换模型请先 run_create 建立独立实验，保留已有训练的目标、方案和结果。');
      if (run.activeJob) {
        this.requireBackend(run);
        const job = await this.compute.status(run.activeJob);
        if (['queued', 'running'].includes(job.status)) throw new Error('当前实验仍在执行。可以另建研究进行比较，或明确取消后修改。');
      }
      const dataset = this.dataset(session);
      const plan = args as unknown as TrainingPlan;
      if (this.needsEmbeddingChoice(session, run, plan)) return this.requestEmbeddingChoice(context, run, plan);
      await this.worker.call('plan.validate', { plan, dataset }, context.signal);
      run.plan = plan; run.revision = (run.revision ?? 0) + 1;
      run.computeBackend = this.backend;
      run.planHash = contentHash({ runId: run.id, revision: run.revision, datasetHash: dataset.sha256, plan, backend: this.backend });
      run.activeJob = undefined;
      this.saveRun(run);
      session.pendingConfirmation = undefined; context.save();
      return { plan: run.plan, summary: this.planSummary(run, dataset), instruction: '方案已记录，尚未授权任何开销。可以讨论、修改、检查环境，或调用 training_advance 提出本次计算授权。' };
    }
    if (name === 'checkpoint_review') {
      if (!session.pendingConfirmation) throw new Error('没有待确认内容，请先提出具体方案。');
      return { needsUser: true, ...session.pendingConfirmation };
    }
    if (name === 'training_prepare') {
      if (session.pendingConfirmation?.kind === 'embedding_choice') return { needsUser: true, summary: session.pendingConfirmation.summary };
      if (run.plan && this.needsEmbeddingChoice(session, run, run.plan)) return this.requestEmbeddingChoice(context, run, run.plan);
      const preview = await this.preview(run, session);
      return { ready: preview.ready, readiness: preview.readiness, summary: preview.summary, instruction: '此工具只检查准备情况，不消耗外部 API 额度，也不改变批准状态。' };
    }
    if (name === 'training_advance') {
      this.requireBackend(run);
      if (run.activeJob) { const result = await this.job(run, session); context.save(); return { ...(result as Record<string, unknown>), reusedExistingJob: true, instruction: '当前方案已有任务，本次仅返回原任务状态，没有创建新的确认卡，也没有启动新计算。不要反复申请同一操作。若用户明确要求新实验，先 plan_propose 保存新一版方案，再请求训练确认；否则说明当前真实状态即可。' }; }
      if (session.pendingConfirmation?.kind === 'embedding_choice') return { needsUser: true, summary: session.pendingConfirmation.summary };
      if (run.plan && this.needsEmbeddingChoice(session, run, run.plan)) return this.requestEmbeddingChoice(context, run, run.plan);
      const preview = await this.preview(run, session);
      if (!preview.ready) return { ready: false, readiness: preview.readiness, instruction: preview.readiness.issues?.join(' ') || '计算环境尚未就绪，不得声称已训练或静默下载模型。' };
      const result = this.requestEffect(session, run, 'compute.submit', preview.request, preview.summary); context.save(); return result;
    }
    if (name === 'training_cancel') {
      if (this.batches.active(session)) {
        const result = this.requestEffect(session, run, 'compute.batch.cancel', { batchIds: session.trainingBatchIds }, '确认停止当前训练队列？正在执行的模型会取消，后续模型不再启动，已有结果保留。'); context.save(); return result;
      }
      this.requireBackend(run);
      if (!run.activeJob) throw new Error('没有可取消的任务');
      const job = await this.compute.status(run.activeJob);
      if (!['queued', 'running'].includes(job.status)) throw new Error('任务已经结束');
      const result = this.requestEffect(session, run, 'compute.cancel', { jobId: run.activeJob, runId: run.id }, '确认停止当前训练？已生成的状态与产物将保留。'); context.save(); return result;
    }
    if (name === 'figure_adjust') {
      let jobId = String(args.jobId);
      let figure = String(args.figure);
      const cited = figureCitationTargets(context.userMessage);
      if (cited.length) {
        const selected = cited.length === 1 ? cited[0] : cited.find(item => item.jobId === jobId && item.figure === figure);
        if (!selected) throw new Error('本轮只能调整已引用的图；请从当前引用中选择一个精确目标。');
        // The cited result identity is authoritative. Model-generated tool arguments
        // may describe the requested edit, but cannot redirect it to another figure.
        jobId = selected.jobId;
        figure = selected.figure;
      }
      const report = (session.reports ?? []).find(item => item.jobId === jobId);
      if (!report) throw new Error('该任务还没有整理过结果。先调用 results_read 生成图表与数据表，再调整其中的图。');
      const target = path.basename(figure);
      const exact = report.files.find(file => file.name === figure || file.path === figure);
      const byBasename = report.files.filter(file => file.kind === 'figure' && path.basename(file.path) === target);
      const figureFile = exact ?? (byBasename.length === 1 ? byBasename[0] : undefined);
      if (!figureFile || figureFile.kind !== 'figure') throw new Error('找不到这张图的交付文件；请使用结果回执里列出的图形名字。');
      const reportDir = path.dirname(report.reportPath);
      const outcome = await this.worker.call<{ path: string; paths?: string[]; name: string; kind: string; note: string }>('figure.adjust', {
        reportDir, figure: path.relative(reportDir, figureFile.path), spec: args.spec,
      }, context.signal);
      const stored = (outcome.paths ?? [outcome.path]).map(file => ({
        name: path.relative(reportDir, file).split(path.sep).join('/'), path: file, kind: 'figure',
      }));
      report.files = [...report.files.filter(item => !stored.some(file => file.path === item.path)), ...stored];
      context.save();
      return { ...outcome, figure: path.relative(reportDir, outcome.path),
        instruction: '已通过所选图的原生绘图代码修改并另存，原图未改动。请用 markdown 图片展示：![调整版图](<' + outcome.path + '>)，说明标题变化，不要展示内部 job ID。' };
    }
    if (name === 'results_read') {
      this.requireBackend(run);
      const selected = session.resultSelection?.resolvedJobIds;
      const jobId = String(args.jobId ?? (selected?.length === 1 ? selected[0] : run.activeJob) ?? '');
      if (!(session.runIds ?? [run.id]).some(id => this.records.get<ResearchRun>('run', id).jobs.includes(jobId))) throw new Error('只能读取本会话已关联任务的结果');
      const payload = await this.analysisPayload(session, [jobId]);
      const key = contentHash({ backend: this.backend, payload });
      const grant = session.resultGrants?.[key];
      if (grant) {
        try { this.approvals.assert(grant, 'results.read', this.backend, payload); }
        catch { delete session.resultGrants![key]; }
        if (session.resultGrants![key] && args.view !== 'report') return this.compute.results(jobId, args.view as ResultView, args.offset as number | undefined);
      }
      const result = this.requestEffect(session, run, 'results.read', payload,
        `确认整理并解读本次模型产出？\n任务：${jobId}\n将读取已完成结果与原始数据标签，调用仓库原生可视化，交付全部可用图表、表格和原始矩阵，解释指标和图表含义。不会重新训练或调用外部 embedding。\n本次不包含结合原始文本与研究或业务目标的深入解读，该操作单独确认。`);
      context.save(); return result;
    }
    if (name === 'results_synthesize') {
      const jobIds = args.jobIds as string[] | undefined ?? session.resultSelection?.resolvedJobIds ?? (run.activeJob ? [run.activeJob] : []);
      if (!jobIds.length) throw new Error('尚无可解读结果');
      const payload = await this.analysisPayload(session, jobIds, String(args.question));
      const result = this.requestEffect(session, run, 'results.synthesize', payload,
        `确认结合原始文本、图表、结果表和既有数据理解进行研究解读？\n问题：${args.question}\n任务：${jobIds.join('、')}\n上下文：${session.contextId ?? '尚未保存，将明确缺失的业务信息'}\n将单独调用研究解读 Agent，使用当前对话模型分析上述证据并保存独立 Markdown，不重训，不调用外部 embedding。`);
      context.save(); return result;
    }
    throw new Error(`未实现工具：${name}`);
  }
  /** Host only: resume the exact suspended action, without another planning/model call. */
  async approve(session: ProductSession, userMessage: string, signal?: AbortSignal, save: () => void = () => {}): Promise<unknown> {
    if (!isExplicitApproval(userMessage) || !session.pendingConfirmation) throw new Error('需要先展示确认内容，并由用户明确回复“确认”。');
    const pending = session.pendingConfirmation;
    if (pending.kind === 'training_config') {
      const theta = pending.trainingPlans?.find(plan => plan.modelId === 'theta' && plan.params.embedding_provider === 'cloud');
      return this.submitTrainingConfiguration(session, { checkpointId: pending.checkpointId, expectedContentHash: pending.contentHash, plans: pending.trainingPlans!, cloudConfirmed: !!theta,
        ...(theta ? { cloudSelection: { provider: String(theta.params.embedding_cloud_provider), model: String(theta.params.embedding_model), endpoint: String(theta.params.embedding_api_base) } } : {}) }, save, signal);
    }
    if (pending.kind !== 'action') throw new Error('旧版本确认已经失效，请重新查看本次计算操作。');
    if (this.approvals.get(pending.checkpointId).action === 'statistics.execute') return this.statistics.approve(session, signal, save);
    const run = this.run(session); this.requireBackend(run);
    if (pending.runId !== run.id) throw new Error('当前研究已变化，请重新确认。');
    const effect = this.approvals.get(pending.checkpointId);
    if (effect.action === 'compute.batch.cancel') {
      if (contentHash(effect.payload) !== contentHash({ batchIds: session.trainingBatchIds })) throw new Error('队列已变化，请重新确认。');
      this.approvals.decide(effect.id, session.id, pending.contentHash, true);
      await this.batches.cancel(session); session.pendingConfirmation = undefined; save(); return { cancelled: true };
    }
    if (effect.action === 'compute.cancel') {
      if (contentHash(effect.payload) !== contentHash({ jobId: run.activeJob, runId: run.id })) throw new Error('当前任务已变化。');
      const receipt = this.approvals.decide(effect.id, session.id, pending.contentHash, true);
      this.approvals.assert(receipt, effect.action, this.backend, effect.payload);
      session.pendingConfirmation = undefined;
      return this.compute.cancel(run.activeJob!);
    }
    if (effect.action === 'results.read' || effect.action === 'results.synthesize') {
      const original = effect.payload as { jobIds: string[]; question?: string };
      const payload = await this.analysisPayload(session, original.jobIds, original.question);
      if (contentHash(payload) !== contentHash(effect.payload)) throw new Error('结果或业务上下文已变化，请重新查看并确认。');
      const receipt = this.approvals.decide(effect.id, session.id, pending.contentHash, true);
      this.approvals.assert(receipt, effect.action, this.backend, payload);
      session.pendingConfirmation = undefined;
      const reports = await Promise.all(original.jobIds.map(async jobId => {
        try { return await this.compute.results(jobId, 'report'); }
        catch (error) {
          const job = await this.compute.status(jobId).catch(() => undefined);
          throw new Error(`结果整理失败：${error instanceof Error ? error.message : String(error)}${job?.resultDir ? `\n原始结果目录：${job.resultDir}` : ''}\n已有训练产物保留；这不是训练重试。修复后可以重新申请结果整理。`);
        }
      }));
      const incomplete = reports.some(value => (value as { reportStatus?: string }).reportStatus === 'incomplete');
      for (const value of reports) {
        const report = value as { jobId?: string; reportPath?: string; manifestPath?: string; reportStatus?: 'complete' | 'incomplete'; files?: Array<{ name: string; path: string; kind: string }> };
        if (report.jobId && report.reportPath && Array.isArray(report.files)) {
          session.reports = [...(session.reports ?? []).filter(old => old.jobId !== report.jobId), { jobId: report.jobId, reportPath: report.reportPath, manifestPath: report.manifestPath, reportStatus: report.reportStatus, files: report.files }].slice(-30);
          session.pendingArtifacts = [...(session.pendingArtifacts ?? []), ...report.files.map(file => ({ name: `${report.jobId!.slice(0, 12)} · ${file.name}`, path: file.path }))];
        }
      }
      if (effect.action === 'results.read') {
        session.resultGrants ??= {};
        session.resultGrants[contentHash({ backend: this.backend, payload })] = receipt;
      } else if (!incomplete) {
        session.pendingSynthesis = { id: `interpretation-${randomUUID()}`, question: original.question!, jobIds: original.jobIds, contextHash: String(payload.contextHash) };
      }
      const analysisReports = reports.map(value => {
        const report = value as Record<string, unknown>;
        if (report.schemaVersion !== 'theta.result-report.v2') return report;
        const rawEvidence = report.evidence as Record<string, unknown>;
        // Per-file hashes and repeated reading advice stay in the manifest. Keep the
        // actual tables and source rows inside the host's bounded inference receipt.
        const evidence = { ...rawEvidence, figures: (rawEvidence.figures as Array<Record<string, unknown>> | undefined)?.map(({ relativePath, format }) => ({ relativePath, format })) };
        return { jobId: report.jobId, modelId: report.modelId, resultHash: report.resultHash,
          trainingPlan: report.trainingPlan,
          ...(effect.action === 'results.synthesize' ? { sourceData: report.sourceData } : {}),
          reportPath: report.reportPath, reportStatus: report.reportStatus, resultDir: report.resultDir, logPath: report.logPath, diagnostics: report.diagnostics, trainingLogPath: report.trainingLogPath, quality: report.quality, summary: report.summary, evidence, missingEvidence: report.missingEvidence,
          artifactRoots: { native: path.join(path.dirname(String(report.reportPath)), 'native'), training: report.resultDir },
          availableArtifacts: (report.files as Array<{ name: string; kind: string }>).map(({ name, kind }) => ({ name, kind })),
          instruction: '本次全部文件链接由宿主追加。使用当前原生表格和矩阵证据作答；报告入口只使用本回执的 reportPath，历史路径可能已过时。无需重复文件清单。' };
      });
      return { resumedAction: effect.action, reports: analysisReports,
        ...(effect.action === 'results.synthesize' ? { researchContext: this.readContext(session) } : {}),
        instruction: incomplete
          ? '结果整理未完成，仅生成了已有文件与诊断入口。明确说明缺失证据和错误，给出 reportPath、resultDir、logPath；availableArtifacts 列出的原始矩阵/文件确实存在，只是分析证据未通过校验，不要说它们不存在。不能声称图表已完整生成，也不能据此做主题占比或深入研究解读。diagnostics 已给出时直接解释，不要再问是否允许读取已有诊断。没有已保存模型的证据时不要承诺可重新导出。未重新训练。'
          : effect.action === 'results.read'
          ? '用户已确认结果整理与基础解读。给出报告路径、所有生成图表和表格链接，逐项解释含义及缺失项；综合业务解读需另用 results_synthesize 征求确认。'
          : '用户已确认独立研究解读。依据 sourceData 原始文本、已保存研究目标、原生图表与原始 theta/beta 回答实质研究问题，引用具体 sourceRow 和原生产物。不要把解释文件格式或图表用途当作结果分析。只有 matrixRowsAligned=true 才能关联文本与该行主题权重。完整回答由宿主保存为独立 Markdown。' };
    }
    if (effect.action !== 'compute.submit') throw new Error('未注册的外部操作，不能执行。');
    const preview = await this.preview(run, session);
    if (!preview.ready || contentHash(preview.request) !== contentHash(effect.payload)) throw new Error('操作、环境配置或数据已变化，请重新查看并确认。');
    const receipt = this.approvals.decide(effect.id, session.id, pending.contentHash, true);
    this.approvals.assert(receipt, effect.action, this.backend, preview.request);
    session.pendingConfirmation = undefined;
    const job = await this.compute.submit(preview.request, receipt);
    run.activeJob = job.id; run.lastObservedJob = { id: job.id, status: job.status, percent: job.percent, phase: job.phase, telemetry: job.telemetry, resultDir: job.resultDir, resultWarning: job.resultWarning }; if (!run.jobs.includes(job.id)) run.jobs.push(job.id);
    this.saveRun(run); session.monitorTraining = ['queued', 'running'].includes(job.status);
    this.options.onJobObserved?.(job);
    return { resumedAction: effect.action, job, instruction: '这是主机实际提交回执。请按真实任务状态回应用户，不得宣称已完成训练。' };
  }
  deny(session: ProductSession, feedback?: string): unknown {
    const pending = session.pendingConfirmation;
    if (!pending) throw new Error('当前没有待确认操作。');
    if (pending.kind === 'action') this.approvals.decide(pending.checkpointId, session.id, pending.contentHash, false);
    session.pendingConfirmation = undefined;
    if (feedback?.trim() && session.runId && pending.runId === session.runId) {
      const run = this.records.get<ResearchRun>('run', pending.runId);
      run.notes.push(feedback.trim()); this.saveRun(run);
    }
    return { denied: true, feedback: feedback?.trim(), instruction: '用户拒绝了该操作，没有启动新计算或外部请求。继续讨论；不要自行再次请求相同操作。' };
  }

}
export const isExplicitApproval = (text: string): boolean => /^(?:确认|确认执行|确认开始训练|批准|同意|yes|approve|\/approve)[。.!！]?$/iu.test(text.trim());
export const explicitEmbeddingChoice = (text: string): 'local' | 'cloud' | undefined => {
  const value = text.trim().replace(/[。.!！]$/u, '');
  if (/^(?:选择|使用|用|我选)?(?:本地|local)(?:嵌入|embedding|emb)?$/iu.test(value)) return 'local';
  if (/^(?:选择|使用|用|我选)?(?:云端|云|cloud)(?:嵌入|embedding|emb)?$/iu.test(value)) return 'cloud';
  return undefined;
};

/** Only unambiguous refusal commands are host shortcuts; other wording stays with the Agent. */
export const isExplicitDenial = (text: string): boolean => /^(?:拒绝|不执行|取消本次操作|no|deny|\/deny)(?:$|[\s，,。.!！:：;；、]|因为|原因是)/iu.test(text.trim());
