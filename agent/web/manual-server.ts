import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { randomUUID } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync, readdirSync, statSync, realpathSync, rmSync, createReadStream } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { DatabaseSync } from 'node:sqlite';
import { PythonCapabilityWorker, type CapabilityWorker } from '../src/adapters/python-worker.js';
import { LocalComputeGateway, type ComputeRequest } from '../src/adapters/compute-gateway.js';
import { EffectApprovals } from '../src/domain/effect-approval.js';
import { ResearchStore } from '../src/memory/research-store.js';
import type { ComputeJob, Dataset, TrainingPlan } from '../src/domain/research.js';
import { contentHash } from '../src/domain/research.js';
import { loadThetaProjectEnvironment, repositoryRoot } from '../src/environment.js';
import { assertRequestOrigin, AgentAccessError } from './deployment.js';
import { exportBuiltinStopwords, parseStopwords, STOPWORD_BYTES } from './stopwords.js';
import { stageDelivery, createDeliveryZip } from './delivery-archive.js';

// 手动工作台是独立宿主：没有 ConversationAgent、对话会话或确认卡映射。
// 只复用底层数据导入、模型计算和结果文件能力；存储也与 .theta_agent 分开。
type RecordValue = Record<string, any>;
class HttpError extends Error { constructor(readonly status: number, message: string) { super(message); } }
const supported = new Set(['theta', 'lda', 'hdp', 'btm', 'stm', 'dtm', 'nvdm', 'gsm', 'prodlda', 'ctm', 'etm', 'bertopic']);
const user = { id: 1, username: '本地开发', full_name: '本地手动工作台', role: 'local', is_active: true };
const json = (res: ServerResponse, value: unknown, status = 200) => { res.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' }); res.end(JSON.stringify(value)); };
async function body(req: IncomingMessage, limit = 1024 * 1024) {
  const chunks: Buffer[] = []; let size = 0;
  for await (const chunk of req) { const bytes = Buffer.from(chunk); size += bytes.length; if (size > limit) throw new HttpError(413, '文件超过大小限制'); chunks.push(bytes); }
  return Buffer.concat(chunks);
}
const safeName = (name: string, maxLength = 160) => { if (typeof name !== 'string' || !name.trim() || name.length > maxLength || /[/\\\x00-\x1f]/u.test(name) || name.trim() === '..' || name.trim() === '.') throw new HttpError(400, '名称包含不允许的字符'); return name.trim(); };
const projectNameKey = (name: string) => name.trim().normalize('NFC').toLowerCase();

export function createManualServer(home: string, worker: CapabilityWorker = new PythonCapabilityWorker(), options: { localToken?: string } = {}) {
  mkdirSync(home, { recursive: true });
  const db = new DatabaseSync(path.join(home, 'manual.sqlite'));
  db.exec('PRAGMA journal_mode=WAL; CREATE TABLE IF NOT EXISTS records (id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, value TEXT NOT NULL)');
  const list = (kind: string): RecordValue[] => db.prepare('SELECT id,value FROM records WHERE kind=? ORDER BY id DESC').all(kind).map(row => ({ ...JSON.parse(String(row.value)), id: Number(row.id) }));
  const get = (kind: string, id: string | number) => { const row = db.prepare('SELECT value FROM records WHERE kind=? AND id=?').get(kind, Number(id)); if (!row) throw new HttpError(404, '记录不存在'); return { ...JSON.parse(String(row.value)), id: Number(id) } as RecordValue; };
  const insert = (kind: string, value: RecordValue): RecordValue => { const id = Number(db.prepare('INSERT INTO records(kind,value) VALUES (?,?)').run(kind, JSON.stringify(value)).lastInsertRowid); return { ...value, id }; };
  const save = (kind: string, value: RecordValue) => db.prepare('UPDATE records SET value=? WHERE kind=? AND id=?').run(JSON.stringify(value), kind, value.id);
  const assertProjectNameAvailable = (name: string, exceptId?: number) => {
    if (list('project').some(project => !project.archived && project.id !== exceptId && projectNameKey(project.name) === projectNameKey(name))) {
      throw new HttpError(409, '已存在同名项目，请打开已有项目，或使用其他名称。');
    }
  };
  const compute = new LocalComputeGateway(home, worker);
  const approvals = new EffectApprovals(new ResearchStore(home));
  const observations = new Map<number, { at: number; pending: Promise<RecordValue> }>();
  const detailedStates = new Map<string, { at: number; value: ComputeJob }>();
  const statusReads = new Map<string, Promise<ComputeJob>>();
  const readStatus = async (id: string, includeTelemetry = true): Promise<ComputeJob> => {
    const cached = compute.cachedStatus(id);
    const observed = detailedStates.get(id);
    if (cached && !includeTelemetry && (['completed', 'failed', 'cancelled'].includes(cached.status) ||
        observed && Date.now() - observed.at < 15_000)) return cached;
    if (observed && (!cached || cached.status === observed.value.status) &&
        (['completed', 'failed', 'cancelled'].includes(observed.value.status) || Date.now() - observed.at < 2_000)) return observed.value;
    const pending = statusReads.get(id);
    if (pending) return pending;
    const read = compute.status(id).then(value => {
      detailedStates.set(id, { at: Date.now(), value });
      return value;
    }).finally(() => statusReads.delete(id));
    statusReads.set(id, read);
    return read;
  };
  const dispatching = new Map<number, Promise<void>>();
  const preparing = new Set<Promise<void>>();
  let closing = false;
  for (const job of list('job')) {
    if (job.preflightIncomplete && !job.queue && ['pending', 'running'].includes(job.status)) {
      save('job', { ...job, status: 'failed', submission_error: true, error_message: '服务重启中断了任务检查，请重新提交。' });
    }
  }
  const advance = (id: number): Promise<void> => {
    const current = dispatching.get(id);
    if (current) return current;
    const work = (async () => {
      let job = get('job', id);
      if (closing || job.cancel_requested || !job.queue?.length) return;
      const states: ComputeJob[] = await Promise.all(job.workers.map((item: RecordValue) => readStatus(item.id, false)));
      if (states.some(state => ['running', 'queued'].includes(state.status))) return;
      job = get('job', id);
      if (closing || job.cancel_requested || !job.queue?.length) return;
      const { model, request } = job.queue[0];
      const approval = approvals.request(request.runId, { action: 'compute.submit', target: 'local', payload: request, summary: `手动训练 ${model}` });
      const receipt = approvals.decide(approval.id, request.runId, approval.hash, true);
      await compute.submit(request, receipt);
      job = get('job', id);
      if (!job.workers.some((item: RecordValue) => item.id === request.jobId)) job.workers.push({ id: request.jobId, model });
      job.queue.shift();
      save('job', job);
      observations.delete(id);
      if (job.cancel_requested) await compute.cancel(request.jobId);
    })().catch(error => {
      const job = get('job', id);
      save('job', { ...job, queue: [], submission_error: true, status: job.cancel_requested ? 'cancelled' : 'failed', error_message: error instanceof Error ? error.message : '排队任务启动失败' });
    }).finally(() => { dispatching.delete(id); });
    dispatching.set(id, work);
    return work;
  };
  const observe = (job: RecordValue, includeTelemetry = true): Promise<RecordValue> => {
    const cached = observations.get(job.id);
    if (includeTelemetry && cached && Date.now() - cached.at < 1500) return cached.pending;
    const pending = (async () => {
      const states: ComputeJob[] = await Promise.all((job.workers as { id: string }[]).map(item => readStatus(item.id, includeTelemetry)));
      // A submission or cancellation may have updated this row while we waited.
      job = get('job', job.id);
      const failed = states.find(state => state.status === 'failed' || state.status === 'cancelled');
      const done = states.length === job.models.length && states.every(state => state.status === 'completed');
      const active = states.find(state => ['running', 'queued'].includes(state.status));
      const status = job.cancel_requested ? (active || dispatching.has(job.id) ? 'cancelling' : 'cancelled') : active || job.queue?.length ? 'running' : failed || job.submission_error ? 'failed' : done ? 'succeeded' : job.status;
      const progress = states.length ? Math.round(states.reduce((sum, state) => sum + state.percent, 0) / job.models.length) : 0;
      Object.assign(job, { status, progress, states, error_message: job.cancel_requested ? '任务已由用户取消' : failed?.error ?? job.error_message }); save('job', job);
      for (const project of list('project')) {
        if (String(project.task_id) !== String(job.id)) continue;
        const pipelineStatus = status === 'succeeded' ? 'completed' : ['failed', 'cancelled'].includes(status) ? 'error' : 'running';
        if (project.pipeline_status !== pipelineStatus) save('project', { ...project, pipeline_status: pipelineStatus, status: pipelineStatus === 'completed' ? 'completed' : pipelineStatus === 'error' ? 'no_result' : 'running' });
      }
      const { queue, ...publicJob } = job;
      return { ...publicJob, queued_models: (queue ?? []).map((item: RecordValue) => item.model), job_id: job.id, task_id: String(job.id), dataset: job.dataset_name, current_step: active?.phase, message: status === 'cancelling' ? '正在停止计算进程' : failed?.error ?? (done ? '训练与结果生成完成' : active ? active.message ?? (active.phase.includes('prepar') ? '正在预处理数据' : '本地模型训练中') : job.error_message ?? '等待执行') };
    })();
    if (includeTelemetry) observations.set(job.id, { at: Date.now(), pending }); return pending;
  };
  const latest = async (dataset: string, model?: string) => {
    for (const job of list('job').filter(j => j.dataset_name === dataset)) {
      const current = await observe(job);
      const selected = current.workers.find((w: RecordValue) => !model || w.model === model);
      const state = current.states?.find((s: ComputeJob) => s.id === selected?.id && s.status === 'completed');
      if (state?.resultDir) return { root: realpathSync(state.resultDir), model: selected.model };
    }
    throw new HttpError(404, '该模型还没有已完成的结果');
  };
  const files = (root: string): string[] => readdirSync(root, { withFileTypes: true }).flatMap(entry => entry.isSymbolicLink() ? [] : entry.isDirectory() ? files(path.join(root, entry.name)) : [path.join(root, entry.name)]);
  const readResult = (root: string, pattern: RegExp) => { const file = files(root).find(file => pattern.test(path.basename(file))); return file ? JSON.parse(readFileSync(file, 'utf8')) : {}; };
  const topics = (root: string): Record<string, [string, number | null][]> => {
    const raw = readResult(root, /^topic_words.*\.json$/u); const values = raw.topics ?? raw.topic_words ?? raw;
    return Object.fromEntries(Object.entries(values).filter(([, value]) => Array.isArray(value)).map(([key, value]) => [key, (value as unknown[]).map(word => Array.isArray(word) ? [String(word[0]), typeof word[1] === 'number' ? word[1] : null] : typeof word === 'object' && word ? [String((word as RecordValue).word), (word as RecordValue).weight ?? null] : [String(word), null])]));
  };
  const server = createServer(async (req, res) => {
    try {
      assertRequestOrigin(req, { mode: 'local', localToken: options.localToken });
      if (req.headers['sec-fetch-site'] === 'cross-site') throw new HttpError(403, '本地接口不接受跨站请求');
      const url = new URL(req.url ?? '/', `http://${req.headers.host}`);
      const parts = url.pathname.split('/').filter(Boolean).map(decodeURIComponent);
      const method = req.method ?? 'GET';
      if (url.pathname === '/health') return json(res, { status: 'ok', service: 'theta-manual', upload_mode: 'direct', local_only: true });
      if (url.pathname === '/api/auth/me') return json(res, user);
      if (url.pathname === '/api/auth/verify') return json(res, { valid: true, user_id: user.id, username: user.username });
      if (url.pathname === '/api/auth/logout') return json(res, { message: '本地开发无需登录' });
      if (url.pathname === '/config') return json(res, { supported_models: [...supported], supported_modes: ['zero_shot', 'unsupervised', 'supervised'], supported_model_sizes: ['0.6B', '4B', '8B'], default_num_topics: 20, default_epochs: 100 });
      if (url.pathname === '/api/runtime/config' && method === 'GET') return json(res, await worker.call('runtime.config', {}));
      if (url.pathname.startsWith('/api/models/') && method === 'GET') {
        const modelId = url.pathname.split('/')[3];
        if (!supported.has(modelId)) throw new HttpError(404, '模型不存在');
        return json(res, await worker.call('models.inspect', { modelId }));
      }
      if (url.pathname === '/api/projects') {
        if (method === 'GET') return json(res, list('project').filter(project => !project.archived));
        if (method === 'POST') {
          const input = JSON.parse((await body(req)).toString());
          const name = safeName(input.name);
          // Check and allocate under one write lock, including concurrent API clients.
          db.exec('BEGIN IMMEDIATE');
          let project: RecordValue;
          try {
            assertProjectNameAvailable(name);
            const base = input.dataset_name ? safeName(input.dataset_name) : name.replace(/\s+/gu, '_').replace(/[^\w\u4e00-\u9fa5-]/gu, '').toLowerCase() || 'dataset';
            // Archived projects and orphaned uploads/results still own their data.
            const reserved = new Set(['project', 'file', 'job'].flatMap(kind => list(kind).map(item => item.dataset_name)));
            let datasetName = base;
            while (reserved.has(datasetName)) datasetName = `${base.slice(0, 120)}-${randomUUID()}`;
            project = insert('project', { name, dataset_name: datasetName, mode: input.mode ?? 'zero_shot', num_topics: input.num_topics ?? 20, created_at: new Date().toISOString(), status: 'draft', pipeline_status: 'draft' });
            db.exec('COMMIT');
          } catch (error) { db.exec('ROLLBACK'); throw error; }
          return json(res, project, 201);
        }
      }
      if (parts[1] === 'projects' && parts.length === 3 && method === 'PATCH') {
        const input = JSON.parse((await body(req)).toString());
        db.exec('BEGIN IMMEDIATE');
        let project: RecordValue;
        try {
          project = get('project', parts[2]);
          if ('name' in input) {
            input.name = safeName(input.name);
            if (projectNameKey(input.name) !== projectNameKey(project.name)) assertProjectNameAvailable(input.name, project.id);
          }
          if ('dataset_name' in input && input.dataset_name !== project.dataset_name) throw new HttpError(409, '项目数据集标识不可更改，请新建项目上传数据。');
          for (const key of ['name', 'mode', 'num_topics', 'status', 'pipeline_status', 'task_id']) if (key in input) project[key] = input[key];
          save('project', project);
          db.exec('COMMIT');
        } catch (error) { db.exec('ROLLBACK'); throw error; }
        return json(res, project);
      }
      if (parts[1] === 'projects' && parts.length === 3 && method === 'DELETE') { const project = get('project', parts[2]); project.archived = true; save('project', project); return json(res, { message: '项目已归档' }); }
      if (url.pathname === '/api/files') return json(res, list('file').map(({ dataset, ...file }) => file));
      if (url.pathname === '/api/stopwords/default' && method === 'GET') {
        res.writeHead(200, { 'Content-Type': 'text/plain; charset=utf-8', 'Content-Disposition': "attachment; filename=THETA-stopwords.txt; filename*=UTF-8''THETA-%E5%86%85%E7%BD%AE%E5%81%9C%E7%94%A8%E8%AF%8D%E8%A1%A8.txt", 'Cache-Control': 'no-store' });
        return res.end(exportBuiltinStopwords(repositoryRoot()));
      }
      if (url.pathname === '/api/stopwords' && method === 'POST') {
        const name = safeName(url.searchParams.get('filename') ?? '');
        if (!/\.txt$/iu.test(name)) throw new HttpError(400, '停用词表请使用 UTF-8 编码的 TXT 文件');
        const words = parseStopwords(await body(req, STOPWORD_BYTES));
        const record = insert('stopwords', { name, words, created_at: new Date().toISOString() });
        return json(res, { id: record.id, name, count: words.length }, 201);
      }
      if (url.pathname === '/api/upload' && method === 'POST') {
        const name = safeName(url.searchParams.get('filename') ?? '', 255); const datasetName = safeName(url.searchParams.get('dataset_name') ?? '');
        if (!/\.(csv|tsv|txt|md|json|jsonl|ndjson|xlsx|xls|parquet|pdf|docx)$/iu.test(name)) throw new HttpError(400, '暂不支持此文件类型');
        const bytes = await body(req, 200 * 1024 * 1024); if (!bytes.length) throw new HttpError(400, '不能上传空文件');
        const expectedSize = req.headers['x-theta-file-size'];
        if (expectedSize !== undefined && (!/^\d+$/.test(String(expectedSize)) || Number(expectedSize) !== bytes.length)) throw new HttpError(400, '文件传输不完整，请重新上传原始文件。');
        const folder = path.join(home, 'incoming', randomUUID()); mkdirSync(folder, { recursive: true });
        const source = path.join(folder, 'data' + path.extname(name).toLowerCase()); writeFileSync(source, bytes, { flag: 'wx', mode: 0o600 });
        const dataset = { ...await worker.call<Dataset>('dataset.import', { filePath: source, uploadDir: path.join(home, 'uploads') }), fileName: name };
        const file = insert('file', { filename: name, dataset_name: datasetName, dataset, file_path: name, size: bytes.length, created_at: new Date().toISOString() });
        return json(res, { id: file.id, filename: name, dataset_name: datasetName, file_path: name }, 201);
      }
      if (parts[1] === 'datasets' && parts[3] === 'combine' && method === 'POST') {
        const input = JSON.parse((await body(req)).toString());
        if (!Array.isArray(input.fileIds) || !input.fileIds.length || input.fileIds.length > 500) throw new HttpError(400, '请选择 1–500 个正文文件');
        const files = input.fileIds.map((id: unknown) => get('file', String(id)));
        if (files.some((file: RecordValue) => !file || file.dataset_name !== parts[2])) throw new HttpError(400, '文件不属于当前项目');
        const dataset = await worker.call<Dataset>('dataset.combine', { datasets: files.map((file: RecordValue, index: number) => ({ ...file.dataset, ...(typeof input.sourceNames?.[index] === 'string' ? { fileName: input.sourceNames[index].slice(0, 512) } : {}) })), uploadDir: path.join(home, 'uploads') });
        const file = insert('file', { filename: dataset.fileName, dataset_name: parts[2], dataset, input_kind: 'text', file_path: dataset.fileName, size: dataset.sizeBytes, created_at: new Date().toISOString() });
        return json(res, { file_id: file.id, name: dataset.fileName, size: dataset.sizeBytes, inputKind: 'text' }, 201);
      }
      if (parts[1] === 'datasets' && parts[3] === 'preview') {
        const file = url.searchParams.get('file_id') ? get('file', url.searchParams.get('file_id')!) : list('file').find(file => file.dataset_name === parts[2]);
        if (!file || file.dataset_name !== parts[2]) throw new HttpError(404, '数据集不存在');
        return json(res, await worker.call('dataset.preview', { dataset: file.dataset }));
      }
      if (parts[1] === 'preprocessing' && parts[2] === 'check') return json(res, { has_bow: false, has_embeddings: false, ready_for_training: false, managed_by_training: true });
      if (url.pathname === '/api/train/start' && method === 'POST') {
        const input = JSON.parse((await body(req)).toString()); const file = get('file', input.file_id);
        if (file.dataset_name !== input.dataset_name) throw new HttpError(400, '上传文件与数据集不匹配');
        const plotLanguage = input.plot_language ?? input.language ?? 'zh';
        if (!['zh', 'en', 'chinese', 'english'].includes(plotLanguage)) throw new HttpError(400, '请选择中文或英文绘图语言');
        const stopwords = input.stopwords_id == null ? undefined : get('stopwords', input.stopwords_id).words;
        const models = [...new Set(String(input.model_type ?? 'theta').split(','))];
        if (!models.length || models.some(model => !supported.has(model))) throw new HttpError(400, '模型选择无效');
        const embeddingProvider = input.embedding_provider ?? 'local';
        if (!['local', 'cloud'].includes(embeddingProvider)) throw new HttpError(400, '请选择本地或云端嵌入');
        const cloud = models.includes('theta') && embeddingProvider === 'cloud';
        if (cloud && (input.cloud_confirmed !== true || (input.mode ?? 'zero_shot') !== 'zero_shot')) throw new HttpError(400, '云端嵌入需要明确同意发送文本，且仅支持零样本模式');
        const externalRequestLimit = input.external_request_limit ?? 200;
        if (cloud && (!Number.isInteger(externalRequestLimit) || externalRequestLimit < 1 || externalRequestLimit > 1000)) throw new HttpError(400, '云端请求上限必须为 1–1000 的整数');
        const requestHash = contentHash(input);
        const existing = list('job').find(job => job.requestHash === requestHash && ['pending', 'running'].includes(job.status));
        if (existing) return json(res, { id: existing.id, status: existing.status, created_at: existing.created_at }, 200);
        const job = insert('job', { requestHash, preflightIncomplete: true, dataset_name: file.dataset_name, file_id: file.id, models, mode: input.mode ?? 'zero_shot', num_topics: input.num_topics, status: 'pending', workers: [], created_at: new Date().toISOString() });
        // Large Excel profiles and twelve preflights exceed an HTTP timeout. Return
        // the durable task immediately; all errors remain visible through polling.
        const preflight = (async () => {
        const profile = await worker.call<{ columns: string[] }>('dataset.profile', { dataset: file.dataset });
        const textColumn = input.text_column || profile.columns.find(column => /^(text|content|cleaned_content|body)$/iu.test(column)) || (profile.columns.length === 1 ? profile.columns[0] : undefined);
        if (!textColumn) throw new HttpError(400, '请选择文本列后再开始分析');
        const plans: { model: string; plan: TrainingPlan; execution: Record<string, unknown> }[] = [];
        for (const model of models) {
          const custom = input.model_params?.[model] ?? {};
          const params: TrainingPlan['params'] = { ...(model === 'hdp' ? { max_topics: input.num_topics ?? 20 } : { num_topics: input.num_topics ?? 20 }), vocab_size: input.vocab_size ?? 5000, ...custom, language: plotLanguage };
          // Text preprocessing is automatic and independent from chart language.
          delete params['text.stopwords'];
          if (stopwords) params['text.stopwords'] = stopwords.join('\n');
          if (model === 'theta') Object.assign(params, { mode: input.mode ?? 'zero_shot', model_size: input.model_size ?? '0.6B', embedding_provider: embeddingProvider });
          const plan: TrainingPlan = { modelId: model, textColumn, params, timeoutSeconds: 43200, device: process.platform === 'win32' ? 'auto' : 'cpu', rationale: '用户在手动工作台明确提交的参数', ...(model === 'theta' && cloud ? { externalRequestLimit } : {}), ...(input.time_column ? { timeColumn: input.time_column } : {}), ...(input.label_column ? { labelColumn: input.label_column } : {}), ...(model === 'stm' ? { covariates: input.meta_columns ?? [] } : {}) };
          const preview = await worker.call<{ execution: Record<string, unknown>; readiness: { ready: boolean; missing?: unknown } }>('compute.preview', { dataset: file.dataset, plan });
          if (!preview.readiness.ready) throw new HttpError(400, `${model.toUpperCase()} 的本地运行环境或模型权重未就绪，请先配置计算环境。`);
          if ((preview.execution.embedding as RecordValue)?.mode !== (model === 'theta' && cloud ? 'cloud' : 'local')) throw new HttpError(400, '实际嵌入策略与用户选择不一致，请重新确认');
          if (model === 'theta' && cloud) {
            const actual = preview.execution.embedding as RecordValue;
            const shown = input.cloud_selection;
            if (!shown || shown.provider !== actual.provider || shown.model !== actual.model || String(shown.endpoint).replace(/\/$/u, '').replace(/\/embeddings$/u, '') + '/embeddings' !== actual.endpoint) throw new HttpError(400, '云端接收服务或模型与界面确认内容不一致，请重新打开配置并确认');
          }
          plans.push({ model, plan, execution: preview.execution });
        }
        const current = get('job', job.id);
        if (current.cancel_requested || closing) return;
        current.queue = plans.map(({ model, plan, execution }) => ({
          model,
          request: { jobId: `job-${contentHash({ manualJobId: job.id, model, dataset: file.dataset, plan })}`, runId: `manual-project-${job.id}`, dataset: file.dataset, plan, execution } satisfies ComputeRequest,
        }));
        current.status = 'running'; current.preflightIncomplete = false; save('job', current);
        await advance(job.id);
        })().catch(error => {
          const current = get('job', job.id);
          save('job', { ...current, status: current.cancel_requested ? 'cancelled' : 'failed', submission_error: true, error_message: error instanceof Error ? error.message : '任务检查失败' });
        });
        preparing.add(preflight);
        void preflight.finally(() => preparing.delete(preflight));
        return json(res, { id: job.id, status: 'pending', created_at: job.created_at }, 201);
      }
      if (url.pathname === '/api/train/jobs') return json(res, await Promise.all(list('job').map(job => observe(job, false))));
      if (parts[1] === 'train' && parts[3] === 'cancel' && parts.length === 4 && method === 'POST') {
        const job = get('job', parts[2]);
        if (!['pending', 'running', 'cancelling', 'cancelled'].includes(job.status)) throw new HttpError(409, '任务已结束');
        job.cancel_requested = true; job.status = 'cancelling'; job.error_message = '正在停止计算进程'; save('job', job);
        await Promise.all(job.workers.map((item: RecordValue) => compute.cancel(item.id)));
        observations.delete(job.id);
        return json(res, await observe(job));
      }
      if (parts[1] === 'train' && parts.length === 4) {
        const job = await observe(get('job', parts[2]));
        if (parts[3] === 'status') return json(res, job);
        const result = await latest(job.dataset_name, job.models[0]);
        if (parts[3] === 'metrics') return json(res, { job_id: job.id, metrics: readResult(result.root, /^(?:evaluation_)?metrics.*\.json$/u) });
        if (parts[3] === 'summary') return json(res, { job_id: job.id, summary: { num_topics: job.num_topics, top_words: Object.values(topics(result.root)).map(words => words.map(word => word[0])) } });
      }
      if (url.pathname === '/api/data/oss-datasets') {
        const names = new Set<string>(); for (const job of list('job')) if ((await observe(job)).states?.some((s: ComputeJob) => s.status === 'completed')) names.add(job.dataset_name);
        return json(res, { datasets: [...names].map(name => ({ name, chart_count: 0 })), storage: 'local' });
      }
      if (parts[1] === 'results') {
        const dataset = parts[2]; const model = url.searchParams.get('model') ?? undefined;
        if (parts[3] === 'catalog' && method === 'GET') {
          const results: RecordValue[] = [];
          const seen = new Set<string>();
          for (const job of list('job').filter(j => j.dataset_name === dataset)) {
            const current = await observe(job);
            for (const state of (current.states ?? []) as ComputeJob[]) {
              if (seen.has(state.id)) continue;
              seen.add(state.id);
              const modelId = current.workers.find((item: RecordValue) => item.id === state.id)?.model;
              const base = `/api/backend/api/results/${encodeURIComponent(dataset)}`;
              const query = `model=${encodeURIComponent(modelId)}&job_id=${encodeURIComponent(state.id)}`;
              const resultRoot = state.status === 'completed' && state.resultDir ? realpathSync(state.resultDir) : undefined;
              const entries = resultRoot ? files(resultRoot).map(file => {
                const name = path.relative(resultRoot, file).split(path.sep).join('/');
                const ext = path.extname(file).toLowerCase();
                const kind = /\.(png|jpe?g|svg|pdf|webp)$/u.test(ext) ? 'figure' : /\.(csv|tsv)$/u.test(ext) ? 'table' : ext === '.npy' ? 'matrix' : ext === '.html' ? 'report' : 'artifact';
                return { name, kind, url: `${base}/visualizations/file?${query}&path=${encodeURIComponent(name)}` };
              }) : [];
              results.push({ jobId: state.id, runId: String(job.id), modelId, status: state.status, phase: state.phase, percent: state.percent, selected: false,
                reportStatus: entries.length ? 'ready' : 'not_requested',
                execution: { id: state.id, status: state.status, phase: state.phase, percent: state.percent, phaseHistory: state.phaseHistory, telemetry: state.telemetry },
                ...(entries.length ? { artifacts: { reportUrl: entries.find(file => /(?:^|\/)index\.html$/u.test(file.name))?.url ?? '', archiveUrl: `${base}/archive?${query}`, fileCount: entries.length, figureCount: entries.filter(file => file.kind === 'figure').length, tableCount: entries.filter(file => file.kind === 'table').length, matrixCount: entries.filter(file => file.kind === 'matrix').length, files: entries } } : {}),
              });
            }
          }
          return json(res, { results, selection: null });
        }
        if (parts[3] === 'models') {
          const models = new Set<string>();
          for (const job of list('job').filter(j => j.dataset_name === dataset)) { const state = await observe(job); for (const item of state.workers) if (state.states?.some((s: ComputeJob) => s.id === item.id && s.status === 'completed')) models.add(item.model); }
          return json(res, { models: [...models] });
        }
        const selectedJobId = url.searchParams.get('job_id');
        const owner = selectedJobId ? list('job').find(job => job.dataset_name === dataset && job.workers?.some((item: RecordValue) => item.id === selectedJobId && (!model || item.model === model))) : undefined;
        if (selectedJobId && !owner) throw new HttpError(404, '该项目没有此训练结果');
        const selectedState = owner ? ((await observe(owner)).states as ComputeJob[]).find(state => state.id === selectedJobId && state.status === 'completed') : undefined;
        if (selectedJobId && !selectedState?.resultDir) throw new HttpError(404, '该训练结果尚未完成');
        const result = selectedState?.resultDir ? { root: realpathSync(selectedState.resultDir), model: owner!.workers.find((item: RecordValue) => item.id === selectedJobId).model } : await latest(dataset, model);
        if (parts[3] === 'archive' && method === 'GET') {
          const stage = path.join(home, 'incoming', `delivery-${randomUUID()}`);
          const archive = stage + '.zip';
          mkdirSync(stage, { recursive: true });
          const cleanup = () => { rmSync(stage, { recursive: true, force: true }); rmSync(archive, { force: true }); };
          try {
            stageDelivery(stage, files(result.root).map(file => ({ path: file, name: '训练结果/' + path.relative(result.root, file).split(path.sep).join('/') })));
            await createDeliveryZip(stage, archive);
            res.writeHead(200, { 'Content-Type': 'application/zip', 'Content-Disposition': `attachment; filename="THETA-${result.model}-complete.zip"`, 'Cache-Control': 'no-store' });
            const stream = createReadStream(archive);
            res.on('close', () => { stream.destroy(); cleanup(); });
            stream.on('error', () => { res.destroy(); cleanup(); });
            return stream.pipe(res);
          } catch (error) { cleanup(); throw error; }
        }
        if (parts[3] === 'topic-words') return json(res, { dataset, model: result.model, topics: topics(result.root) });
        if (parts[3] === 'metrics') return json(res, { dataset, model: result.model, metrics: readResult(result.root, /^(?:evaluation_)?metrics.*\.json$/u) });
        if (parts[3] === 'visualizations') {
          const artifacts = files(result.root);
          if (parts[4] === 'file') {
            const relative = url.searchParams.get('path') ?? ''; const file = path.resolve(result.root, relative);
            if (!artifacts.includes(file) || !realpathSync(file).startsWith(result.root + path.sep)) throw new HttpError(404, '结果文件不存在');
            const mime: Record<string, string> = { '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.svg': 'image/svg+xml', '.html': 'text/html', '.csv': 'text/csv; charset=utf-8', '.json': 'application/json', '.pdf': 'application/pdf', '.txt': 'text/plain; charset=utf-8' };
            res.writeHead(200, { 'Content-Type': mime[path.extname(file)] ?? 'application/octet-stream', 'X-Content-Type-Options': 'nosniff', 'Content-Security-Policy': "sandbox allow-scripts; default-src 'self' data: https: 'unsafe-inline'" });
            const stream = createReadStream(file);
            res.on('close', () => stream.destroy());
            stream.on('error', () => res.destroy());
            return stream.pipe(res);
          }
          const global_files: RecordValue[] = []; const topic_files: Record<string, RecordValue[]> = {};
          for (const file of artifacts) {
            const relative = path.relative(result.root, file).split(path.sep).join('/'); const item = { name: path.basename(file), path: relative, size: statSync(file).size, type: path.extname(file).slice(1), url: `/api/backend/api/results/${encodeURIComponent(dataset)}/visualizations/file?model=${encodeURIComponent(result.model)}&path=${encodeURIComponent(relative)}` };
            const topic = relative.match(/(?:^|\/)topic_(\d+)\//u)?.[1]; if (topic) (topic_files[topic] ??= []).push(item); else global_files.push(item);
          }
          return json(res, { dataset, model: result.model, global_files, topic_files });
        }
      }
      throw new HttpError(404, '接口不存在');
    } catch (error) { json(res, { detail: error instanceof Error ? error.message : '本地服务出错' }, error instanceof HttpError || error instanceof AgentAccessError ? error.status : 400); }
  });
  const queueTimer = setInterval(() => {
    for (const job of list('job')) if (job.queue?.length && !job.cancel_requested) void advance(job.id);
  }, 3000);
  queueTimer.unref();
  let resourcesClosed: Promise<void> = Promise.resolve();
  server.once('close', () => {
    closing = true; clearInterval(queueTimer);
    resourcesClosed = Promise.allSettled([...preparing, ...dispatching.values(), ...[...observations.values()].map(item => item.pending)]).then(() => db.close());
  });
  const closeHttp = server.close.bind(server);
  // HTTP close alone does not wait for background work or release SQLite's Windows lock.
  server.close = (callback) => closeHttp(error => {
    void resourcesClosed.then(() => callback?.(error), cause => callback?.(cause));
  });
  return server;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (process.env.NODE_ENV === 'production') throw new Error('手动免登录服务仅用于本机开发，不能在生产环境启动。');
  loadThetaProjectEnvironment();
  const home = path.join(repositoryRoot(), '.local', 'manual-workbench');
  const port = Number(process.env.THETA_MANUAL_PORT ?? 4321);
  createManualServer(home).listen(port, '127.0.0.1', () => console.log(`THETA 手动工作台 http://127.0.0.1:${port} · ${home}`));
}
