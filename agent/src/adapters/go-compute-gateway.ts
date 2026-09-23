import { createWriteStream, lstatSync, mkdirSync, openAsBlob, readFileSync, readdirSync, rmSync } from 'node:fs';
import { createHash, randomUUID } from 'node:crypto';
import { Readable, Transform } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import path from 'node:path';
import { t as listTar, x as extractTar, type ReadEntry } from 'tar';
import type { ComputeJob } from '../domain/research.js';
import { contentHash } from '../domain/research.js';
import { ResearchStore } from '../memory/research-store.js';
import type { ComputeGateway, ComputeRequest, ResultView } from './compute-gateway.js';
import { EffectApprovals, type ApprovalReceipt } from '../domain/effect-approval.js';
import type { CapabilityWorker } from './python-worker.js';

interface RemoteJobRecord { hash: string; taskId?: string; request?: ComputeRequest }

interface RemoteBinding {
  userId: string; projectId: string;
  datasets: Record<string, { ref: string; sha256: string; textColumn: string; timeColumn?: string; covariates?: string[] }>;
  models: Record<string, { modelId: number; runtimeId?: number }>;
}
const goConfiguration = () => ({
  bindings: process.env.THETA_COMPUTE_BINDINGS ? readFileSync(process.env.THETA_COMPUTE_BINDINGS, 'utf8') : null,
  credential: process.env.THETA_COMPUTE_TOKEN ?? null,
});
export const goConfigurationFingerprint = (): string => contentHash(goConfiguration());
/** Adapter for the Go v1 API. Statically bound datasets remain host-controlled;
 * unbound managed uploads are transferred through the additive dataset endpoint.
 * The current server has no submit idempotency contract: uncertain requests are fenced
 * locally and never retried automatically. Go owns all remote task lifecycle state. */
export class GoComputeGateway implements ComputeGateway {
  constructor(private readonly endpoint: string, private readonly store: ResearchStore, private readonly fetcher: typeof fetch = fetch,
    private readonly home?: string, private readonly worker?: CapabilityWorker) {
    const url = new URL(endpoint);
    if (url.protocol !== 'https:' && !(url.protocol === 'http:' && ['127.0.0.1', 'localhost', '[::1]'].includes(url.hostname))) throw new Error('远端计算端点必须使用 HTTPS；本机允许 HTTP。');
    if (url.username || url.password || url.search || url.hash) throw new Error('计算端点不可包含凭据、查询或片段。');
  }
  private async request(route: string, body?: unknown): Promise<Record<string, unknown>> {
    const response = await this.fetcher(`${this.endpoint.replace(/\/$/u, '')}${route}`, { method: body === undefined ? 'GET' : 'POST', headers: { 'content-type': 'application/json', ...(process.env.THETA_COMPUTE_TOKEN ? { authorization: `Bearer ${process.env.THETA_COMPUTE_TOKEN}` } : {}) }, ...(body === undefined ? {} : { body: JSON.stringify(body) }), signal: AbortSignal.timeout(15000), redirect: 'error' });
    if (!response.ok) throw new Error(`计算服务返回 HTTP ${response.status}`);
    return await response.json() as Record<string, unknown>;
  }
  private async uploadDataset(request: ComputeRequest, bindings: RemoteBinding): Promise<string> {
    const file = await openAsBlob(request.dataset.managedPath);
    const form = new FormData();
    form.append('file', file, request.dataset.fileName);
    form.append('dataset_ref', request.dataset.datasetRef);
    form.append('project_id', bindings.projectId);
    form.append('user_id', bindings.userId);
    form.append('display_name', request.dataset.fileName);
    form.append('sha256', request.dataset.sha256);
    const response = await this.fetcher(`${this.endpoint.replace(/\/$/u, '')}/api/v1/datasets`, {
      method: 'POST', body: form, redirect: 'error', signal: AbortSignal.timeout(300000),
      headers: process.env.THETA_COMPUTE_TOKEN ? { authorization: `Bearer ${process.env.THETA_COMPUTE_TOKEN}` } : {},
    });
    const body = await response.json().catch(() => null) as { dataset_ref?: string; sha256?: string; error?: { message?: string } } | null;
    if (!response.ok) throw new Error(`计算服务数据上传返回 HTTP ${response.status}${body?.error?.message ? `：${body.error.message}` : ''}`);
    if (!body?.dataset_ref || body.sha256 !== request.dataset.sha256) throw new Error('计算服务返回的数据集引用或 SHA-256 与本地托管副本不一致。');
    return body.dataset_ref;
  }
  async submit(request: ComputeRequest, authorization?: ApprovalReceipt): Promise<ComputeJob> {
    new EffectApprovals(this.store).assert(authorization, 'compute.submit', this.endpoint, request);
    const configuration = goConfiguration();
    if (request.execution?.computeConfigurationFingerprint !== contentHash(configuration)) throw new Error('远端数据/runtime 映射或凭据已变化，请重新确认。');
    if ((request.execution?.embedding as { mode?: string } | undefined)?.mode === 'cloud') throw new Error('当前 Go worker 协议尚不能验证云 embedding 调用预算，暂不能远端执行此方案。');
    let existing: RemoteJobRecord | undefined;
    try { existing = this.store.get('remote-job', request.jobId); } catch (error) { if (!(error instanceof Error) || !error.message.startsWith('记录不存在')) throw error; }
    const hash = contentHash(request);
    if (existing) {
      if (existing.hash !== hash) throw new Error('提交内容与原请求不一致');
      if (existing.taskId) return this.status(request.jobId);
      throw new Error('上次远端提交结果不确定，请先在 Go 控制面核对任务；不会自动重复提交。');
    }
    const file = process.env.THETA_COMPUTE_BINDINGS;
    if (!file) throw new Error('远端模式需要 THETA_COMPUTE_BINDINGS：宿主登记的用户、数据版本和模型 runtime 映射。');
    const bindings = JSON.parse(configuration.bindings!) as RemoteBinding;
    const dataset = bindings.datasets[request.dataset.datasetRef];
    const model = bindings.models[request.plan.modelId];
    if (dataset && (dataset.sha256 !== request.dataset.sha256 || dataset.textColumn !== request.plan.textColumn || dataset.timeColumn !== request.plan.timeColumn || contentHash(dataset.covariates ?? []) !== contentHash(request.plan.covariates ?? []))) throw new Error('远端数据及规范化列映射与本次方案不匹配。');
    if (!model || !bindings.userId || !bindings.projectId) throw new Error('缺少远端身份/模型映射');
    const remoteDatasetRef = dataset?.ref ?? await this.uploadDataset(request, bindings);
    if (!this.store.putIfAbsent('remote-job', request.jobId, { hash, request })) throw new Error('另一个宿主正在提交此任务，请先查询状态，不会重复发送请求。');
    if (Object.hasOwn(request.plan.params, 'embedding.model_path')) throw new Error('远端任务不能提交宿主本地 embedding 路径。');
    const params = {
      ...request.plan.params,
      'input.text_column': request.plan.textColumn,
      ...(request.plan.timeColumn ? { 'input.time_column': request.plan.timeColumn } : {}),
      ...(request.plan.labelColumn ? { 'input.label_column': request.plan.labelColumn } : {}),
      ...(request.plan.covariates?.length ? { 'input.covariates': request.plan.covariates } : {}),
      ...(request.plan.modelId === 'theta' && request.plan.params.embedding_provider === undefined ? { embedding_provider: 'local' } : {}),
    };
    const task = await this.request('/api/v1/tasks', { user_id: bindings.userId, project_id: bindings.projectId, dataset_ref: remoteDatasetRef, model_id: model.modelId, ...(model.runtimeId ? { runtime_id: model.runtimeId } : {}), job_name: request.jobId, params, priority: 5 });
    if (typeof task.id !== 'number' && typeof task.id !== 'string') throw new Error('计算服务未返回有效 task ID，提交结果待核对');
    this.store.put('remote-job', request.jobId, { hash, taskId: String(task.id), request });
    return this.map(request.jobId, task);
  }
  private id(jobId: string): string {
    const mapping = this.store.get<{ taskId?: string }>('remote-job', jobId);
    if (!mapping.taskId || !/^\d+$/u.test(mapping.taskId)) throw new Error('缺少已确认的远端任务 ID');
    return mapping.taskId;
  }
  private map(id: string, task: Record<string, unknown>): ComputeJob {
    const state = String(task.status).toLowerCase();
    const status = ({ pending: 'queued', queued: 'queued', running: 'running', succeeded: 'completed', success: 'completed', completed: 'completed', failed: 'failed', cancelled: 'cancelled', canceled: 'cancelled', cancelling: 'running' } as const)[state];
    if (!status) throw new Error(`未知远端任务状态：${state}`);
    return { id, status, phase: String(task.phase ?? state), percent: Number(task.progress ?? 0) };
  }
  async status(jobId: string): Promise<ComputeJob> { return this.map(jobId, await this.request(`/api/v1/tasks/${this.id(jobId)}`)); }
  async cancel(jobId: string): Promise<ComputeJob> { await this.request(`/api/v1/tasks/${this.id(jobId)}/cancel`, {}); return this.status(jobId); }
  private cachedReport(jobId: string): Record<string, unknown> | undefined {
    try { return this.store.get<Record<string, unknown>>('remote-report', jobId); }
    catch (error) { if (!(error instanceof Error) || !error.message.startsWith('记录不存在')) throw error; return undefined; }
  }
  private reportView(report: Record<string, unknown>, view: ResultView, offset = 0): unknown {
    if (view === 'report') return report;
    if (!Number.isInteger(offset) || offset < 0) throw new Error('无效结果分页位置');
    const evidence = structuredClone(report.evidence as Record<string, unknown>);
    Object.assign(evidence, { reportPath: report.reportPath, reportStatus: report.reportStatus,
      resultDir: report.resultDir, missingEvidence: report.missingEvidence });
    if (view === 'artifacts') return { jobId: report.jobId, sha256: report.resultHash, resultDir: report.resultDir,
      reportPath: report.reportPath, reportStatus: report.reportStatus, missingEvidence: report.missingEvidence,
      availableArtifacts: (report.files as Array<Record<string, unknown>>).map(({ name, kind, path }) => ({ name, kind, path })) };
    const key = ({ tables: 'tables', figures: 'figures', distribution: 'matrices', metrics: 'evidence' } as const)[view as 'tables'|'figures'|'distribution'|'metrics'];
    if (key) {
      const values = (evidence[key] as unknown[] | undefined) ?? []; const size = view === 'tables' || view === 'metrics' ? 2 : 10;
      return { trainingRunId: report.jobId, [key]: values.slice(offset, offset + size), itemCount: values.length,
        nextOffset: offset + size < values.length ? offset + size : null, limitations: evidence.limitations,
        reportPath: report.reportPath, reportStatus: report.reportStatus, missingEvidence: report.missingEvidence };
    }
    const counts = Object.fromEntries(['evidence','tables','figures','matrices'].map(key => [key, ((evidence[key] as unknown[] | undefined) ?? []).length]));
    return { ...evidence, evidence: ((evidence.evidence as unknown[]) ?? []).slice(0, 3),
      tables: ((evidence.tables as unknown[]) ?? []).slice(0, 1), figures: ((evidence.figures as unknown[]) ?? []).slice(0, 5),
      summaryCounts: counts, plan: report.trainingPlan, instruction: '概览有界；用 metrics/tables/figures/distribution 分页读取剩余证据。' };
  }
  private async hydrate(jobId: string, result: Record<string, unknown>, download: Record<string, unknown>): Promise<Record<string, unknown>> {
    if (!this.home || !this.worker) throw new Error('Agent 未配置远端结果校验目录或报告 Worker');
    const mapping = this.store.get<RemoteJobRecord>('remote-job', jobId);
    if (!mapping.request) throw new Error('此任务由旧版 Agent 提交，缺少数据与训练方案绑定，不能安全关联原始文本行');
    const output = (result.output_files ?? {}) as Record<string, unknown>;
    const integrity = (output.integrity ?? {}) as Record<string, unknown>;
    const sha256 = String(download.sha256 ?? integrity.archive_sha256 ?? '');
    const size = Number(download.size_bytes ?? integrity.archive_size_bytes);
    const normalized = String(integrity.normalized_input_sha256 ?? '');
    if (!/^[a-f0-9]{64}$/u.test(sha256) || !Number.isSafeInteger(size) || size <= 0 || size > 5 * 1024 ** 3 || !/^[a-f0-9]{64}$/u.test(normalized)) {
      throw new Error('Worker 未返回完整的归档 SHA-256、长度或规范化输入校验信息，拒绝深入读取');
    }
    const url = new URL(String(download.download_url ?? ''));
    if (url.protocol !== 'https:' && !(url.protocol === 'http:' && ['127.0.0.1','localhost','[::1]'].includes(url.hostname))) throw new Error('模型下载链接必须使用 HTTPS；本机允许 HTTP');
    const base = path.join(this.home, 'remote-results', jobId); mkdirSync(base, { recursive: true });
    const bundle = path.join(base, `bundle-${randomUUID()}`); const archive = path.join(base, `download-${randomUUID()}.tar.gz`);
    try {
      const response = await this.fetcher(url, { redirect: 'error', signal: AbortSignal.timeout(120000) });
      if (!response.ok || !response.body) throw new Error(`结果包下载返回 HTTP ${response.status}`);
      const lengthHeader = response.headers.get('content-length'); const declared = lengthHeader === null ? undefined : Number(lengthHeader);
      if (declared !== undefined && Number.isFinite(declared) && declared !== size) throw new Error('结果包 Content-Length 与 Worker 元数据不一致');
      const digest = createHash('sha256'); let received = 0;
      const meter = new Transform({ transform(chunk: Buffer, _encoding, callback) {
        received += chunk.length; digest.update(chunk);
        callback(received > size ? new Error('结果包超过声明长度') : null, chunk);
      } });
      await pipeline(Readable.fromWeb(response.body as never), meter, createWriteStream(archive, { mode: 0o600 }));
      if (received !== size || digest.digest('hex') !== sha256) throw new Error('结果包长度或 SHA-256 校验失败');
      mkdirSync(bundle, { recursive: false, mode: 0o700 });
      let entries = 0; let expanded = 0;
      const safeName = (name: string) => { const clean = name.endsWith('/') ? name.slice(0, -1) : name; return Boolean(clean) && clean !== '.' && !path.isAbsolute(clean) && !clean.includes('\\') && clean.split('/').every(part => part && part !== '.' && part !== '..'); };
      let invalid = false;
      await listTar({ file: archive, strict: true, onReadEntry: (entry: ReadEntry) => {
        entries++; expanded += Number(entry.size ?? 0);
        if (entries > 100000 || expanded > 10 * 1024 ** 3 || !safeName(entry.path) || !['File','Directory'].includes(entry.type)) invalid = true;
        entry.resume();
      } });
      if (invalid) throw new Error('结果包包含不安全或超出预算的条目');
      await extractTar({ file: archive, cwd: bundle, preservePaths: false, strict: true,
        filter: (name, entry) => { const type = 'type' in entry ? String(entry.type) : entry.isDirectory() ? 'Directory' : entry.isFile() ? 'File' : 'Other'; return safeName(name) && ['File','Directory'].includes(type); } });
      const inspect = (directory: string): void => { for (const name of readdirSync(directory)) { const item = path.join(directory, name); const stat = lstatSync(item); if (stat.isSymbolicLink() || !stat.isFile() && !stat.isDirectory()) throw new Error('结果包解压后包含不安全条目'); if (stat.isDirectory()) inspect(item); } };
      inspect(bundle);
      mkdirSync(path.join(bundle, 'workspace'), { recursive: true, mode: 0o700 });
      const report = await this.worker.call<Record<string, unknown>>('compute.remote_report', { home: this.home, jobId, bundleRoot: bundle,
        normalizedInputSha256: normalized, dataset: mapping.request.dataset, plan: mapping.request.plan });
      this.store.put('remote-report', jobId, report); return report;
    } catch (error) { rmSync(bundle, { recursive: true, force: true }); throw error; }
    finally { rmSync(archive, { force: true }); }
  }
  async results(jobId: string, view: ResultView, offset?: number): Promise<unknown> {
    if ((await this.status(jobId)).status !== 'completed') throw new Error('远端任务尚未完成');
    const cached = this.cachedReport(jobId);
    if (cached) return this.reportView(cached, view, offset);
    const taskId = this.id(jobId);
    const [result, download] = await Promise.all([
      this.request(`/api/v1/tasks/${taskId}/result`),
      this.request(`/api/v1/tasks/${taskId}/model/download`),
    ]);
    if (view === 'report') return this.hydrate(jobId, result, download);
    return {
      jobId,
      taskId,
      metrics: result.metrics ?? {},
      outputFiles: result.output_files ?? {},
      modelWeightsPath: result.model_weights_path,
      download,
      evidence: [],
      limitations: ['分布式控制面返回已验证的指标与对象清单；深入主题解读仍需读取并校验下载包中的矩阵、表格和原文行。'],
    };
  }
}
