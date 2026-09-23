import { createHash } from 'node:crypto';
import { openAsBlob, readFileSync } from 'node:fs';
import type { ComputeJob, Dataset } from '../domain/research.js';
import { contentHash } from '../domain/research.js';
import { ResearchStore } from '../memory/research-store.js';
import type { ComputeGateway, ComputeRequest, ResultView } from './compute-gateway.js';
import { EffectApprovals, type ApprovalReceipt } from '../domain/effect-approval.js';

/** Host-maintained private mapping. Business API 没有模型/runtime 列表接口，
 * 因此 model_id 必须由部署方提供，不能由模型或客户端猜测。 */
export interface BusinessBindings {
  projectId?: string;
  projectTitle?: string;
  allowProjectCreate?: boolean;
  models: Record<string, { modelId: number; runtimeId?: number }>;
  datasets?: Record<string, { datasetRef: string; sha256: string }>;
}

type ParamRule = { kind: 'int' | 'float' | 'bool' | 'choice' | 'nullable_int' | 'int_array' | 'string'; min?: number; max?: number; choices?: readonly string[] };
/** 与 trainning/internal/task/params.go 的服务端白名单一致；本地先拒绝，避免发送注定 422 的请求。 */
const COMMON_PARAMS: Record<string, ParamRule> = {
  num_topics: { kind: 'int', min: 2, max: 100 },
  vocab_size: { kind: 'int', min: 100, max: 100000 },
  epochs: { kind: 'int', min: 1, max: 500 },
  batch_size: { kind: 'int', min: 1, max: 512 },
  hidden_dim: { kind: 'int', min: 16, max: 4096 },
  learning_rate: { kind: 'float', min: 0.000001, max: 1 },
  patience: { kind: 'int', min: 0, max: 500 },
  skip_eval: { kind: 'bool' },
  skip_viz: { kind: 'bool' },
  no_early_stopping: { kind: 'bool' },
  language: { kind: 'choice', choices: ['en', 'zh', 'chinese', 'english'] },
};
const NEURAL_PARAMS: Record<string, ParamRule> = {
  dropout: { kind: 'float', min: 0, max: 1 },
  num_layers: { kind: 'int', min: 1, max: 10 },
  embedding_dim: { kind: 'int', min: 8, max: 8192 },
  hidden_sizes: { kind: 'int_array', min: 4, max: 8192 },
};
const MODEL_PARAMS: Record<string, Record<string, ParamRule>> = {
  theta: {
    model_size: { kind: 'choice', choices: ['0.6B', '4B', '8B'] },
    mode: { kind: 'choice', choices: ['zero_shot', 'supervised', 'unsupervised'] },
    kl_start: { kind: 'float', min: 0, max: 1 },
    kl_end: { kind: 'float', min: 0, max: 1 },
    kl_warmup: { kind: 'int', min: 0, max: 500 },
    dropout: { kind: 'float', min: 0, max: 1 },
    num_layers: { kind: 'int', min: 1, max: 10 },
    hidden_sizes: { kind: 'int_array', min: 4, max: 8192 },
    embedding_provider: { kind: 'choice', choices: ['cloud', 'local', 'qwen', 'openai', 'dashscope', 'siliconflow', 'zhipu', 'volcengine', 'openai_compatible'] },
    embedding_cloud_provider: { kind: 'choice', choices: ['openai', 'dashscope', 'siliconflow', 'zhipu', 'volcengine', 'openai_compatible'] },
    embedding_model: { kind: 'string' },
    embedding_api_base: { kind: 'string' },
    embedding_api_key_env: { kind: 'string' },
    embedding_dimensions: { kind: 'int', min: 1, max: 65536 },
  },
  lda: { max_iter: { kind: 'int', min: 1, max: 10000 } },
  hdp: { max_topics: { kind: 'int', min: 2, max: 1000 }, alpha: { kind: 'float', min: 0.000001, max: 1000 } },
  stm: { max_iter: { kind: 'int', min: 1, max: 10000 } },
  btm: { n_iter: { kind: 'int', min: 1, max: 10000 }, alpha: { kind: 'float', min: 0.000001, max: 1000 }, beta: { kind: 'float', min: 0.000001, max: 1000 } },
  etm: NEURAL_PARAMS,
  ctm: { ...NEURAL_PARAMS, inference_type: { kind: 'choice', choices: ['zeroshot', 'combined'] } },
  dtm: NEURAL_PARAMS,
  nvdm: NEURAL_PARAMS,
  gsm: NEURAL_PARAMS,
  prodlda: NEURAL_PARAMS,
  bertopic: {
    n_neighbors: { kind: 'int', min: 2, max: 100 },
    n_components: { kind: 'int', min: 2, max: 50 },
    min_cluster_size: { kind: 'int', min: 2, max: 10000 },
    min_samples: { kind: 'nullable_int', min: 1, max: 10000 },
    top_n_words: { kind: 'int', min: 1, max: 100 },
    random_state: { kind: 'int', min: 0, max: 2147483647 },
  },
};
const CLOUD_EMBEDDING_PROVIDERS = new Set(['cloud', 'openai', 'dashscope', 'siliconflow', 'zhipu', 'volcengine', 'openai_compatible']);
const EXTENDED_PARAMETER_NAMESPACES = new Set(['model', 'trainer', 'fit', 'prepare', 'main', 'config', 'pipeline', 'word2vec', 'embedding']);

export interface BusinessApiConfiguration { baseUrl: string; identifier: string | null; password: string | null; bindingsPath: string | null }
export const businessApiConfiguration = (): BusinessApiConfiguration => ({
  baseUrl: process.env.THETA_BUSINESS_API_URL ?? '',
  identifier: process.env.THETA_BUSINESS_IDENTIFIER ?? null,
  password: process.env.THETA_BUSINESS_PASSWORD ?? null,
  bindingsPath: process.env.THETA_BUSINESS_BINDINGS ?? null,
});
const digest = (value: string): string => createHash('sha256').update(value).digest('hex');
/** 授权与配置绑定：端点、账号或模型映射变化都会使旧确认失效。凭据只以摘要进入指纹。 */
const fingerprintOf = (configuration: BusinessApiConfiguration): string => contentHash({
  baseUrl: configuration.baseUrl,
  account: configuration.identifier === null ? null : digest(configuration.identifier),
  credential: configuration.password === null ? null : digest(configuration.password),
  bindings: configuration.bindingsPath ? readFileSync(configuration.bindingsPath, 'utf8') : null,
});
export const businessApiConfigurationFingerprint = (override: Partial<BusinessApiConfiguration> = {}): string =>
  fingerprintOf({ ...businessApiConfiguration(), ...override });

interface Session { cookie: string; csrf: string; expiresAt: number }
interface ApiResponse { status: number; body: unknown; headers: Headers }

/** 公开 Business API 适配器：Node 进程充当浏览器，内存持有 Session Cookie 与 CSRF Token。
 * 不向 LLM/浏览器暴露凭据；写请求带允许的 Origin；结果不确定的 POST 不自动重发。 */
export class BusinessApiComputeGateway implements ComputeGateway {
  private session: Session | null = null;
  constructor(private readonly store: ResearchStore, private readonly fetcher: typeof fetch = fetch, private readonly options: { baseUrl?: string; identifier?: string; password?: string; bindingsPath?: string; now?: () => number } = {}) {
    const url = new URL(this.baseUrl);
    if (url.username || url.password || url.search || url.hash || url.pathname !== '/') throw new Error('Business API 地址必须是不含凭据、路径、查询或片段的来源地址。');
    if (url.protocol !== 'https:' && !(url.protocol === 'http:' && ['127.0.0.1', 'localhost', '[::1]'].includes(url.hostname))) throw new Error('Business API 必须使用 HTTPS；本机允许 HTTP。');
  }
  private get baseUrl(): string { return (this.options.baseUrl ?? process.env.THETA_BUSINESS_API_URL ?? '').replace(/\/$/u, ''); }
  private get now(): number { return (this.options.now ?? Date.now)(); }
  private configuration(): BusinessApiConfiguration {
    return {
      baseUrl: this.baseUrl,
      identifier: this.options.identifier ?? process.env.THETA_BUSINESS_IDENTIFIER ?? null,
      password: this.options.password ?? process.env.THETA_BUSINESS_PASSWORD ?? null,
      bindingsPath: this.options.bindingsPath ?? process.env.THETA_BUSINESS_BINDINGS ?? null,
    };
  }
  private bindings(): BusinessBindings {
    const file = this.configuration().bindingsPath;
    if (!file) throw new Error('Business API 模式需要 THETA_BUSINESS_BINDINGS：宿主维护的项目与 model_id/runtime_id 映射。');
    const parsed = JSON.parse(readFileSync(file, 'utf8')) as BusinessBindings;
    if (!parsed || typeof parsed.models !== 'object' || parsed.models === null) throw new Error('Business API 绑定文件缺少 models 映射。');
    return parsed;
  }
  private fingerprint(): string { return fingerprintOf(this.configuration()); }

  /** 单次 HTTP 调用。GET 之外的请求带 Origin 与 CSRF；401 只对读请求重登重试，避免重复副作用。 */
  private async api(route: string, init: { method?: string; json?: unknown; form?: FormData; signal?: AbortSignal; retry?: boolean } = {}): Promise<ApiResponse> {
    const method = (init.method ?? 'GET').toUpperCase();
    const write = !['GET', 'HEAD', 'OPTIONS'].includes(method);
    if (write) await this.ensureSession(init.signal);
    const headers: Record<string, string> = { accept: 'application/json' };
    if (write) headers.origin = new URL(this.baseUrl).origin;
    if (this.session) headers.cookie = this.session.cookie;
    if (write && this.session?.csrf) headers['x-csrf-token'] = this.session.csrf;
    let body: BodyInit | undefined;
    if (init.form) body = init.form;
    else if (init.json !== undefined) { headers['content-type'] = 'application/json'; body = JSON.stringify(init.json); }
    const response = await this.fetcher(this.baseUrl + route, { method, headers, ...(body === undefined ? {} : { body }), signal: init.signal ?? AbortSignal.timeout(15000), redirect: 'error' });
    const status = response.status;
    let parsed: unknown = null;
    const text = status === 204 ? '' : await response.text();
    if (text) { try { parsed = JSON.parse(text); } catch { parsed = null; } }
    const code = (parsed as { error?: { code?: string } } | null)?.error?.code;
    if (status === 401) {
      this.session = null;
      if (!write && init.retry !== false) { await this.ensureSession(init.signal); return this.api(route, { ...init, retry: false }); }
      throw new Error('Business API 会话已失效或被拒绝，请重新登录后再试。');
    }
    if (status === 403 && code === 'csrf_failed' && write && init.retry !== false) {
      await this.refreshSession(init.signal);
      return this.api(route, { ...init, retry: false });
    }
    if (status === 403 && code === 'origin_rejected') throw new Error('Business API 拒绝请求 Origin：这是部署配置问题，不重试。');
    if (!response.ok) throw new Error(this.describe(status, parsed));
    return { status, body: parsed, headers: response.headers };
  }
  private describe(status: number, parsed: unknown): string {
    const error = (parsed as { error?: { code?: string; message?: string; field?: string } } | null)?.error;
    const suffix = [error?.code, error?.field].filter(Boolean).join(' / ');
    return 'Business API 返回 HTTP ' + status + (suffix ? '（' + suffix + '）' : '') + (error?.message ? '：' + error.message : '');
  }
  private async ensureSession(signal?: AbortSignal): Promise<void> {
    if (this.session && this.session.expiresAt > this.now) return;
    const { identifier, password } = this.configuration();
    if (!identifier || !password) throw new Error('Business API 需要 THETA_BUSINESS_IDENTIFIER 与 THETA_BUSINESS_PASSWORD（本机私有配置，不提交）。');
    const response = await this.fetcher(this.baseUrl + '/api/v1/auth/login', {
      method: 'POST', redirect: 'error', signal: signal ?? AbortSignal.timeout(15000),
      headers: { accept: 'application/json', 'content-type': 'application/json', origin: new URL(this.baseUrl).origin },
      body: JSON.stringify({ identifier, password }),
    });
    const text = await response.text();
    let parsed: unknown = null; try { parsed = text ? JSON.parse(text) : null; } catch { parsed = null; }
    if (!response.ok) throw new Error(this.describe(response.status, parsed));
    const cookie = this.sessionCookie(response.headers);
    if (!cookie) throw new Error('Business API 登录成功但未返回 __Host-theta_session Cookie，无法继续。');
    const body = parsed as { csrf_token?: string; expires_at?: string } | null;
    const csrf = typeof body?.csrf_token === 'string' ? body.csrf_token : '';
    const expires = body?.expires_at ? Date.parse(body.expires_at) : Number.NaN;
    this.session = { cookie, csrf, expiresAt: Number.isFinite(expires) ? expires : this.now + 15 * 60 * 1000 };
  }
  private async refreshSession(signal?: AbortSignal): Promise<void> {
    const response = await this.fetcher(this.baseUrl + '/api/v1/auth/me', {
      method: 'GET', redirect: 'error', signal: signal ?? AbortSignal.timeout(15000),
      headers: { accept: 'application/json', ...(this.session ? { cookie: this.session.cookie } : {}) },
    });
    const text = await response.text();
    let parsed: unknown = null; try { parsed = text ? JSON.parse(text) : null; } catch { parsed = null; }
    if (response.status === 401) { this.session = null; await this.ensureSession(signal); return; }
    if (!response.ok) throw new Error(this.describe(response.status, parsed));
    if (!this.session) throw new Error('Business API 缺少会话，无法刷新 CSRF Token。');
    const body = parsed as { csrf_token?: string } | null;
    if (typeof body?.csrf_token !== 'string' || !body.csrf_token) throw new Error('Business API 未返回 CSRF Token，写操作已停止。');
    this.session = { ...this.session, csrf: body.csrf_token };
  }
  private sessionCookie(headers: Headers): string | null {
    const values = typeof headers.getSetCookie === 'function' ? headers.getSetCookie() : [headers.get('set-cookie') ?? ''];
    for (const value of values) {
      const match = /(?:^|,\s*)__Host-theta_session=([^;]*)/u.exec(value ?? '');
      if (match) return '__Host-theta_session=' + match[1];
    }
    return null;
  }

  private assertRepresentable(request: ComputeRequest): Record<string, unknown> {
    const plan = request.plan;
    if (plan.device) throw new Error('Business API 没有设备选择协议：不能丢弃 device 后提交。');
    const rules = MODEL_PARAMS[plan.modelId];
    if (!rules) throw new Error('Business API 未登记此模型的参数契约：' + plan.modelId);
    const params: Record<string, unknown> = { 'input.text_column': plan.textColumn };
    if (plan.timeColumn) params['input.time_column'] = plan.timeColumn;
    if (plan.labelColumn) params['input.label_column'] = plan.labelColumn;
    if (plan.covariates?.length) params['input.covariates'] = [...plan.covariates];
    for (const [key, value] of Object.entries(plan.params)) {
      const rule = COMMON_PARAMS[key] ?? rules[key];
      if (rule) params[key] = this.validateParam(key, value, rule);
      else {
        if (key === 'embedding.model_path') throw new Error('远端本地模型路径由 worker 宿主配置，不能把 Agent 所在机器的路径作为参数提交。');
        const namespace = key.includes('.') ? key.slice(0, key.indexOf('.')) : '';
        if (!EXTENDED_PARAMETER_NAMESPACES.has(namespace)) throw new Error('Business API 不接受该参数，已拒绝提交：' + key);
        params[key] = this.validateExtendedParam(key, value);
      }
    }
    if (plan.modelId === 'theta' && params.embedding_provider === undefined) params.embedding_provider = 'local';
    const provider = params.embedding_provider;
    if (typeof provider === 'string' && CLOUD_EMBEDDING_PROVIDERS.has(provider)) {
      if (plan.modelId !== 'theta' || (params.mode ?? 'zero_shot') !== 'zero_shot') throw new Error('远端 embedding API 仅支持 THETA zero_shot；微调仍需 Worker 本地 Qwen。');
      const base = params.embedding_api_base;
      if (base !== undefined) {
        if (typeof base !== 'string') throw new Error('embedding_api_base 必须是 URL 字符串。');
        const url = new URL(base);
        const localHttp = url.protocol === 'http:' && ['localhost', '127.0.0.1', '[::1]'].includes(url.hostname);
        if (url.protocol !== 'https:' && !localHttp) throw new Error('远端 embedding_api_base 必须使用 HTTPS；本机测试允许 HTTP。');
        if (url.username || url.password || url.search || url.hash) throw new Error('embedding_api_base 不得包含凭据、查询参数或片段。');
      }
      const keyEnv = params.embedding_api_key_env;
      if (keyEnv !== undefined && (typeof keyEnv !== 'string' || !/^[A-Za-z_][A-Za-z0-9_]*$/u.test(keyEnv))) throw new Error('embedding_api_key_env 必须是环境变量名，不能提交密钥内容。');
    }
    return params;
  }
  private validateExtendedParam(name: string, value: unknown): unknown {
    if (value === null || typeof value === 'boolean') return value;
    if (typeof value === 'string') {
      if (!value || value.length > 500 || value.startsWith('-')) throw new Error('扩展参数 ' + name + ' 必须是非空且有界的字符串。');
      return value;
    }
    if (typeof value === 'number') {
      if (!Number.isFinite(value)) throw new Error('扩展参数 ' + name + ' 必须是有限数值。');
      return value;
    }
    if (Array.isArray(value) && value.length >= 1 && value.length <= 32 && value.every(item => typeof item === 'number' && Number.isInteger(item) && item > 0)) return value;
    throw new Error('扩展参数 ' + name + ' 必须是标量或 1–32 个正整数的数组。');
  }
  private validateParam(name: string, value: unknown, rule: ParamRule): unknown {
    if (value === null && rule.kind === 'nullable_int') return null;
    if (rule.kind === 'bool') { if (typeof value !== 'boolean') throw new Error('参数 ' + name + ' 必须是布尔值。'); return value; }
    if (rule.kind === 'choice') { if (typeof value !== 'string' || !rule.choices?.includes(value)) throw new Error('参数 ' + name + ' 取值不在 Business API 允许范围内：' + String(value)); return value; }
    if (rule.kind === 'string') { if (typeof value !== 'string' || !value || value.length > 500 || value.startsWith('-')) throw new Error('参数 ' + name + ' 必须是非空且有界的字符串。'); return value; }
    if (rule.kind === 'int_array') {
      if (!Array.isArray(value) || value.length < 1 || value.length > 10 || !value.every(item => typeof item === 'number' && Number.isInteger(item) && item >= (rule.min ?? -Infinity) && item <= (rule.max ?? Infinity))) {
        throw new Error('参数 ' + name + ' 必须包含 1–10 个、每个 ' + rule.min + '–' + rule.max + ' 的整数。');
      }
      return [...value];
    }
    const numeric = typeof value === 'number' ? value : Number.NaN;
    if (!Number.isFinite(numeric)) throw new Error('参数 ' + name + ' 必须是数值。');
    if ((rule.kind === 'int' || rule.kind === 'nullable_int') && !Number.isInteger(numeric)) throw new Error('参数 ' + name + ' 必须是整数。');
    if (rule.min !== undefined && numeric < rule.min) throw new Error('参数 ' + name + '=' + numeric + ' 低于 Business API 下限 ' + rule.min + '；不会为凑合服务端而修改训练参数。');
    if (rule.max !== undefined && numeric > rule.max) throw new Error('参数 ' + name + '=' + numeric + ' 超过 Business API 上限 ' + rule.max + '；不会为凑合服务端而修改训练参数。');
    if (name === 'dropout' && numeric >= 1) throw new Error('参数 dropout 必须小于 1。');
    return value;
  }

  private async project(bindings: BusinessBindings, signal?: AbortSignal): Promise<string> {
    if (bindings.projectId) {
      await this.api('/api/v1/projects/' + encodeURIComponent(bindings.projectId), { signal });
      return bindings.projectId;
    }
    const title = bindings.projectTitle?.trim();
    if (!title) throw new Error('Business API 绑定缺少 projectId 或 projectTitle。');
    for (let page = 1; page <= 5; page++) {
      const listing = await this.api('/api/v1/projects?page=' + page + '&page_size=100', { signal });
      const items = (listing.body as { items?: Array<{ id?: string; title?: string }> } | null)?.items ?? [];
      const found = items.find(item => item.title === title);
      if (found?.id) return found.id;
      if (!(listing.body as { has_more?: boolean } | null)?.has_more) break;
    }
    if (!bindings.allowProjectCreate) throw new Error('Business API 中不存在标题为「' + title + '」的项目；如需自动创建请在私有绑定中显式设置 allowProjectCreate=true。');
    const created = await this.api('/api/v1/projects', { method: 'POST', json: { title }, signal });
    const id = (created.body as { id?: string } | null)?.id;
    if (!id) throw new Error('Business API 未返回新项目 ID。');
    return id;
  }
  private async dataset(request: ComputeRequest, bindings: BusinessBindings, projectId: string, signal?: AbortSignal): Promise<string> {
    const registered = bindings.datasets?.[request.dataset.datasetRef];
    if (registered) {
      const detail = await this.api('/api/v1/projects/' + encodeURIComponent(projectId) + '/datasets/' + encodeURIComponent(registered.datasetRef), { signal });
      const sha = (detail.body as { sha256?: string } | null)?.sha256;
      if (sha !== request.dataset.sha256) throw new Error('远端已登记数据集内容 hash 与本次方案不一致，拒绝复用。');
      return registered.datasetRef;
    }
    const file = await openAsBlob(request.dataset.managedPath);
    const form = new FormData();
    form.append('file', file, request.dataset.fileName);
    form.append('display_name', request.dataset.fileName);
    let created: ApiResponse;
    try {
      created = await this.api('/api/v1/projects/' + encodeURIComponent(projectId) + '/datasets', { method: 'POST', form, signal: signal ?? AbortSignal.timeout(300000) });
    } catch (error) {
      // 同一项目内已存在相同内容时服务端返回 409；只有找到 sha256 完全一致的已登记记录才复用。
      if (!(error instanceof Error) || !/dataset_duplicate/u.test(error.message)) throw error;
      const reused = await this.datasetByHash(projectId, request.dataset.sha256, signal);
      if (!reused) throw error;
      return reused;
    }
    const body = created.body as { dataset_ref?: string; sha256?: string } | null;
    if (!body?.dataset_ref) throw new Error('Business API 未返回 dataset_ref，上传结果待核对。');
    if (body.sha256 !== request.dataset.sha256) throw new Error('上传后远端 sha256=' + String(body.sha256) + ' 与本地托管副本 ' + request.dataset.sha256 + ' 不一致，已停止提交。');
    return body.dataset_ref;
  }
  private async datasetByHash(projectId: string, sha256: string, signal?: AbortSignal): Promise<string | null> {
    for (let page = 1; page <= 5; page++) {
      const listing = await this.api('/api/v1/projects/' + encodeURIComponent(projectId) + '/datasets?page=' + page + '&page_size=100', { signal });
      const items = (listing.body as { items?: Array<{ dataset_ref?: string; sha256?: string }>; has_more?: boolean } | null);
      const found = (items?.items ?? []).find(item => item.sha256 === sha256 && item.dataset_ref);
      if (found?.dataset_ref) return found.dataset_ref;
      if (!items?.has_more) return null;
    }
    return null;
  }

  /** 宿主自检：登录并读取当前账号；只返回账号摘要，不返回 Cookie 或 CSRF 明文。 */
  async authenticate(signal?: AbortSignal): Promise<{ user: Record<string, unknown>; expiresAt: string | null; csrfAvailable: boolean }> {
    await this.ensureSession(signal);
    const response = await this.fetcher(this.baseUrl + '/api/v1/auth/me', {
      method: 'GET', redirect: 'error', signal: signal ?? AbortSignal.timeout(15000),
      headers: { accept: 'application/json', ...(this.session ? { cookie: this.session.cookie } : {}) },
    });
    const text = await response.text();
    let parsed: unknown = null; try { parsed = text ? JSON.parse(text) : null; } catch { parsed = null; }
    if (!response.ok) throw new Error(this.describe(response.status, parsed));
    const body = parsed as { user?: Record<string, unknown>; csrf_token?: string; expires_at?: string } | null;
    if (typeof body?.csrf_token === 'string' && body.csrf_token && this.session) this.session = { ...this.session, csrf: body.csrf_token };
    const user = body?.user ?? {};
    return { user: { id: user.id, username: user.username, display_name: user.display_name, role: user.role, status: user.status }, expiresAt: body?.expires_at ?? null, csrfAvailable: Boolean(body?.csrf_token) };
  }
  /** 宿主验收：解析绑定项目；不存在时只有 allowProjectCreate=true 才会创建。 */
  async ensureProject(signal?: AbortSignal): Promise<string> { return this.project(this.bindings(), signal); }
  /** 宿主验收：上传托管副本并核对远端 sha256，返回远端 dataset_ref。 */
  async ensureDataset(dataset: Dataset, signal?: AbortSignal): Promise<string> {
    const projectId = await this.ensureProject(signal);
    return this.dataset({ dataset } as ComputeRequest, this.bindings(), projectId, signal);
  }
  /** 只检查本地参数契约，不发送请求；用于验收前如实报告方案是否可提交。 */
  representableParams(request: ComputeRequest): Record<string, unknown> { return this.assertRepresentable(request); }

  async submit(request: ComputeRequest, authorization?: ApprovalReceipt): Promise<ComputeJob> {
    new EffectApprovals(this.store).assert(authorization, 'compute.submit', this.baseUrl, request);
    const params = this.assertRepresentable(request);
    if (!this.configuration().baseUrl) throw new Error('缺少 THETA_BUSINESS_API_URL。');
    if (request.execution?.computeConfigurationFingerprint !== this.fingerprint()) throw new Error('Business API 端点、账号或模型映射已变化，请重新确认。');
    const bindings = this.bindings();
    const model = bindings.models[request.plan.modelId];
    if (!model || !Number.isInteger(model.modelId) || model.modelId <= 0) throw new Error('私有绑定缺少模型 ' + request.plan.modelId + ' 的 model_id；不能猜测模型 ID。');
    let existing: { taskId?: string; hash: string; rejected?: string } | undefined;
    try { existing = this.store.get('remote-job', request.jobId); } catch (error) { if (!(error instanceof Error) || !error.message.startsWith('记录不存在')) throw error; }
    const hash = contentHash({ request, configuration: this.fingerprint() });
    if (existing) {
      if (existing.hash !== hash) throw new Error('提交内容与原请求不一致。');
      if (existing.taskId) return this.status(request.jobId);
      if (existing.rejected) throw new Error('Business API 已拒绝该提交：' + existing.rejected);
      throw new Error('上次远端提交结果不确定，请先在 Business API 核对任务；不会自动重复提交。');
    }
    if (!this.store.putIfAbsent('remote-job', request.jobId, { hash })) throw new Error('另一个宿主正在提交此任务，请先查询状态，不会重复发送请求。');
    try {
      const projectId = await this.project(bindings, undefined);
      const datasetRef = await this.dataset(request, bindings, projectId, undefined);
      const task = await this.api('/api/v1/tasks', {
        method: 'POST',
        json: { project_id: projectId, dataset_ref: datasetRef, model_id: model.modelId, ...(model.runtimeId ? { runtime_id: model.runtimeId } : {}), job_name: request.jobId, params },
      });
      const id = (task.body as { id?: number | string } | null)?.id;
      if (typeof id !== 'number' && typeof id !== 'string') throw new Error('Business API 未返回有效 task ID，提交结果待核对。');
      this.store.put('remote-job', request.jobId, { hash, taskId: String(id), projectId, datasetRef, modelId: model.modelId });
      return this.map(request.jobId, task.body as Record<string, unknown>);
    } catch (error) {
      // 明确 4xx 拒绝可以记录原因；其余情况保持不确定，禁止自动重发。
      if (error instanceof Error && /HTTP 4[0-9][0-9]/u.test(error.message)) this.store.put('remote-job', request.jobId, { hash, rejected: error.message });
      throw error;
    }
  }
  private id(jobId: string): string {
    const mapping = this.store.get<{ taskId?: string }>('remote-job', jobId);
    if (!mapping.taskId || !/^\d+$/u.test(mapping.taskId)) throw new Error('缺少已确认的 Business API 任务 ID。');
    return mapping.taskId;
  }
  private map(id: string, task: Record<string, unknown>): ComputeJob {
    const state = String(task.status ?? '').toLowerCase();
    const status = ({ pending: 'queued', queued: 'queued', running: 'running', cancelling: 'running', succeeded: 'completed', success: 'completed', completed: 'completed', failed: 'failed', cancelled: 'cancelled', canceled: 'cancelled' } as const)[state];
    if (!status) throw new Error('未知 Business API 任务状态：' + state);
    return { id, status, phase: String(task.phase ?? state), percent: Math.max(0, Math.min(100, Number(task.progress ?? 0) || 0)), ...(task.error_message ? { error: String(task.error_message) } : {}) };
  }
  async status(jobId: string): Promise<ComputeJob> { return this.map(jobId, (await this.api('/api/v1/tasks/' + this.id(jobId))).body as Record<string, unknown>); }
  async cancel(jobId: string): Promise<ComputeJob> { await this.api('/api/v1/tasks/' + this.id(jobId) + '/cancel', { method: 'POST' }); return this.status(jobId); }
  async results(jobId: string, _view: ResultView, _offset?: number): Promise<unknown> {
    if ((await this.status(jobId)).status !== 'completed') throw new Error('Business API 任务尚未完成。');
    const download = await this.api('/api/v1/tasks/' + this.id(jobId) + '/model/download');
    return { jobId, download: download.body, evidence: [], limitations: ['Business API 只提供模型产物下载地址，没有结构化结果 manifest 接口；不能据此虚构主题词、指标或图表。'] };
  }
}
