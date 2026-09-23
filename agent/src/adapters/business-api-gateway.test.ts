import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { BusinessApiComputeGateway, businessApiConfigurationFingerprint, type BusinessBindings } from './business-api-gateway.js';
import { EffectApprovals } from '../domain/effect-approval.js';
import { ResearchStore } from '../memory/research-store.js';

const BASE = 'http://127.0.0.1:4319';
const PROJECT = '10000000-0000-0000-0000-000000000001';
const DATASET = '20000000-0000-0000-0000-000000000002';

interface Call { url: string; method: string; headers: Record<string, string>; body: unknown }
const json = (status: number, body: unknown, extra: Record<string, string> = {}): Response =>
  new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json', ...extra } });
const failure = (status: number, code: string): Response => json(status, { error: { code, message: code } });

const setup = (bindings: Partial<BusinessBindings> = {}) => {
  const home = mkdtempSync(path.join(os.tmpdir(), 'theta-business-'));
  const bindingsPath = path.join(home, 'bindings.json');
  writeFileSync(bindingsPath, JSON.stringify({ projectId: PROJECT, models: { lda: { modelId: 7, runtimeId: 3 } }, datasets: { local: { datasetRef: DATASET, sha256: 'sha-local' } }, ...bindings }));
  const store = new ResearchStore(home);
  const approvals = new EffectApprovals(store);
  const options = { baseUrl: BASE, identifier: 'tester', password: 'secret-password', bindingsPath };
  const request = (overrides: Record<string, unknown> = {}) => ({
    jobId: 'job-one', runId: 'run-one',
    execution: { computeConfigurationFingerprint: businessApiConfigurationFingerprint(options) },
    dataset: { datasetRef: 'local', sha256: 'sha-local', fileName: 'data.csv', managedPath: path.join(home, 'data.csv'), sizeBytes: 12 },
    plan: { modelId: 'lda', textColumn: 'text', params: { num_topics: 5, max_iter: 20 }, rationale: 'baseline', timeoutSeconds: 60 },
    ...overrides,
  });
  const permit = (payload: unknown) => {
    const value = approvals.request('session', { action: 'compute.submit', target: BASE, payload, summary: 'test task' });
    return approvals.decide(value.id, 'session', value.hash, true);
  };
  return { home, bindingsPath, store, options, request, permit };
};

test('Business API gateway keeps the session in memory, sends Origin/CSRF on writes and maps task state', async () => {
  const context = setup();
  const calls: Call[] = [];
  let logins = 0;
  let status = 'queued';
  const fetcher: typeof fetch = async (input, init = {}) => {
    const url = String(input); const method = String(init.method ?? 'GET');
    calls.push({ url, method, headers: (init.headers ?? {}) as Record<string, string>, body: init.body });
    if (url.endsWith('/api/v1/auth/login')) { logins++; return json(200, { csrf_token: 'csrf-one', expires_at: new Date(Date.now() + 3600000).toISOString() }, { 'set-cookie': '__Host-theta_session=session-one; Path=/; HttpOnly' }); }
    if (url.includes('/datasets/')) return json(200, { dataset_ref: DATASET, sha256: 'sha-local' });
    if (url.endsWith('/api/v1/projects/' + PROJECT)) return json(200, { id: PROJECT, title: 'study' });
    if (url.endsWith('/api/v1/tasks')) return json(201, { id: 42, status: 'queued', phase: 'scheduling', progress: 0 });
    if (url.endsWith('/api/v1/tasks/42')) return json(200, status === 'completed' ? { id: 42, status, phase: 'completed', progress: 100 } : { id: 42, status: 'running', phase: 'training', progress: 60 });
    if (url.endsWith('/api/v1/tasks/42/model/download')) return json(200, { task_id: 42, download_url: 'https://theta.code-soul.com/api/v1/local-storage/token', expires_at: new Date().toISOString() });
    throw new Error('unexpected request ' + url);
  };
  try {
    const gateway = new BusinessApiComputeGateway(context.store, fetcher, context.options);
    const request = context.request();
    const job = await gateway.submit(request, context.permit(request));
    assert.equal(job.status, 'queued');
    assert.equal(job.percent, 0);
    assert.equal(logins, 1, '登录应只发生一次并复用内存会话');
    const task = calls.find(call => call.url.endsWith('/api/v1/tasks'))!;
    assert.equal(task.headers.origin, BASE, '写请求必须带允许的 Origin');
    assert.equal(task.headers.cookie, '__Host-theta_session=session-one');
    assert.equal(task.headers['x-csrf-token'], 'csrf-one');
    const posted = JSON.parse(String(task.body)) as { project_id: string; dataset_ref: string; model_id: number; runtime_id: number; params: Record<string, unknown> };
    assert.equal(posted.project_id, PROJECT); assert.equal(posted.dataset_ref, DATASET);
    assert.equal(posted.model_id, 7); assert.equal(posted.runtime_id, 3);
    assert.deepEqual(posted.params, { 'input.text_column': 'text', num_topics: 5, max_iter: 20 });
    for (const read of calls.filter(call => call.method === 'GET')) assert.equal(read.headers.origin, undefined, 'GET 不需要 Origin');
    assert.equal((await gateway.status('job-one')).percent, 60);
    await assert.rejects(gateway.results('job-one', 'summary'), /尚未完成/);
    status = 'completed';
    const results = await gateway.results('job-one', 'summary') as { download: unknown; limitations: string[] };
    assert.ok(results.download); assert.match(results.limitations[0]!, /manifest/);
    await gateway.status('job-one');
    assert.equal(logins, 1, '未过期的会话不应重新登录');
  } finally { rmSync(context.home, { recursive: true, force: true }); }
});

test('Business API gateway refuses plans the public parameter contract cannot represent, without sending anything', async () => {
  const context = setup();
  const calls: Call[] = [];
  const fetcher: typeof fetch = async (input, init = {}) => {
    calls.push({ url: String(input), method: String(init.method ?? 'GET'), headers: {}, body: init.body });
    return json(500, { error: { code: 'unexpected', message: 'should not be reached' } });
  };
  const gateway = new BusinessApiComputeGateway(context.store, fetcher, context.options);
  const cases: Array<[Record<string, unknown>, RegExp]> = [
    [{ params: { num_topics: 1, max_iter: 20 } }, /下限/],
    [{ device: 'cuda:0' }, /device/],
    [{ params: { num_topics: 5, unknown_flag: true } }, /不接受该参数/],
    [{ params: { 'unknown.flag': true } }, /不接受该参数/],
    [{ params: { 'embedding.model_path': '/tmp/model' } }, /宿主配置/],
  ];
  try {
    for (const [plan, expected] of cases) {
      const base = context.request();
      const request = { ...base, plan: { ...base.plan, ...plan } };
      await assert.rejects(gateway.submit(request, context.permit(request)), expected);
    }
    const theta = context.request({ plan: { modelId: 'theta', textColumn: 'text', params: { num_topics: 5, embedding_provider: 'cloud', embedding_cloud_provider: 'openai_compatible', embedding_model: 'embed-large', embedding_api_base: 'https://embedding.example/v1', embedding_api_key_env: 'REMOTE_EMBEDDING_KEY', embedding_dimensions: 1024 }, rationale: 'zero shot', timeoutSeconds: 60 } });
    assert.deepEqual(gateway.representableParams(theta), {
      'input.text_column': 'text', num_topics: 5, embedding_provider: 'cloud', embedding_cloud_provider: 'openai_compatible',
      embedding_model: 'embed-large', embedding_api_base: 'https://embedding.example/v1', embedding_api_key_env: 'REMOTE_EMBEDDING_KEY', embedding_dimensions: 1024,
    });
    assert.throws(() => gateway.representableParams({ ...theta, plan: { ...theta.plan, params: { ...theta.plan.params, mode: 'supervised' } } }), /仅支持 THETA zero_shot/);
    assert.throws(() => gateway.representableParams({ ...theta, plan: { ...theta.plan, params: { ...theta.plan.params, embedding_api_key_env: 'raw-secret!' } } }), /环境变量名/);
    const extended = context.request({ jobId: 'job-extended', plan: { ...context.request().plan, timeColumn: 'timestamp', covariates: ['channel'], params: { num_topics: 5, 'model.alpha': .2, 'trainer.hidden_sizes': [64, 32] } } });
    assert.deepEqual(gateway.representableParams(extended), {
      'input.text_column': 'text', 'input.time_column': 'timestamp', 'input.covariates': ['channel'],
      num_topics: 5, 'model.alpha': .2, 'trainer.hidden_sizes': [64, 32],
    });
    assert.equal(calls.length, 0, '本地拒绝不应产生任何外部请求');
  } finally { rmSync(context.home, { recursive: true, force: true }); }
});

test('Business API gateway never submits when the uploaded dataset hash differs from the local copy', async () => {
  const context = setup({ datasets: {} });
  writeFileSync(context.request().dataset.managedPath, 'text\nhello\n');
  const calls: Call[] = [];
  const fetcher: typeof fetch = async (input, init = {}) => {
    const url = String(input); const method = String(init.method ?? 'GET');
    calls.push({ url, method, headers: (init.headers ?? {}) as Record<string, string>, body: init.body });
    if (url.endsWith('/api/v1/auth/login')) return json(200, { csrf_token: 'csrf-one' }, { 'set-cookie': '__Host-theta_session=session-one; Path=/' });
    if (url.endsWith('/api/v1/projects/' + PROJECT)) return json(200, { id: PROJECT });
    if (url.endsWith('/datasets')) return json(201, { dataset_ref: DATASET, sha256: 'sha-other' });
    throw new Error('unexpected request ' + url);
  };
  try {
    const gateway = new BusinessApiComputeGateway(context.store, fetcher, context.options);
    const request = context.request();
    await assert.rejects(gateway.submit(request, context.permit(request)), /不一致/);
    const upload = calls.find(call => call.url.endsWith('/datasets'))!;
    assert.ok(upload.body instanceof FormData, '上传必须是 multipart/form-data');
    assert.equal(upload.headers['content-type'], undefined, 'boundary 由运行时生成，不能手工设置');
    assert.equal(calls.some(call => call.url.endsWith('/api/v1/tasks')), false, 'hash 不符时不得创建任务');
  } finally { rmSync(context.home, { recursive: true, force: true }); }
});

test('Business API gateway refreshes CSRF once on csrf_failed and never repeats an uncertain submit', async () => {
  const context = setup();
  const calls: Call[] = [];
  let taskPosts = 0;
  let breakNetwork = false;
  const fetcher: typeof fetch = async (input, init = {}) => {
    const url = String(input); const method = String(init.method ?? 'GET');
    calls.push({ url, method, headers: (init.headers ?? {}) as Record<string, string>, body: init.body });
    if (url.endsWith('/api/v1/auth/login')) return json(200, { csrf_token: 'csrf-one' }, { 'set-cookie': '__Host-theta_session=session-one; Path=/' });
    if (url.endsWith('/api/v1/auth/me')) return json(200, { csrf_token: 'csrf-two' });
    if (url.includes('/datasets/')) return json(200, { dataset_ref: DATASET, sha256: 'sha-local' });
    if (url.endsWith('/api/v1/projects/' + PROJECT)) return json(200, { id: PROJECT });
    if (url.endsWith('/api/v1/tasks')) {
      taskPosts++;
      if (breakNetwork) throw new Error('network interrupted');
      if (taskPosts === 1) return failure(403, 'csrf_failed');
      return json(201, { id: 42, status: 'queued', phase: 'scheduling', progress: 0 });
    }
    throw new Error('unexpected request ' + url);
  };
  try {
    const gateway = new BusinessApiComputeGateway(context.store, fetcher, context.options);
    const request = context.request();
    const job = await gateway.submit(request, context.permit(request));
    assert.equal(job.status, 'queued');
    const posts = calls.filter(call => call.url.endsWith('/api/v1/tasks'));
    assert.equal(posts.length, 2, 'csrf_failed 后只重试一次');
    assert.equal(posts[0]!.headers['x-csrf-token'], 'csrf-one');
    assert.equal(posts[1]!.headers['x-csrf-token'], 'csrf-two');
    assert.ok(calls.some(call => call.url.endsWith('/api/v1/auth/me')), '刷新 CSRF 需要调用 /auth/me');

    breakNetwork = true; taskPosts = 0;
    const uncertain = context.request({ jobId: 'job-uncertain' });
    const receipt = context.permit(uncertain);
    await assert.rejects(gateway.submit(uncertain, receipt), /network interrupted/);
    await assert.rejects(gateway.submit(uncertain, receipt), /不确定/);
    assert.equal(taskPosts, 1, '结果不确定的提交不会自动重发');
  } finally { rmSync(context.home, { recursive: true, force: true }); }
});
