import assert from 'node:assert/strict';
import test from 'node:test';
import { createHash } from 'node:crypto';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import type { AddressInfo } from 'node:net';
import os from 'node:os';
import path from 'node:path';
import { BusinessApiComputeGateway, businessApiConfigurationFingerprint } from './business-api-gateway.js';
import { EffectApprovals } from '../domain/effect-approval.js';
import { ResearchStore } from '../memory/research-store.js';

const PROJECT = '10000000-0000-0000-0000-000000000001';
const DATASET = '20000000-0000-0000-0000-000000000002';

/** 以真实 HTTP 复现文档契约的假 Business API：Origin 校验、Cookie 会话、CSRF 轮换与 multipart 上传。 */
async function fakeBusinessApi(fixtureSha: string, options: { duplicate?: boolean } = {}) {
  const state = { csrf: 'csrf-one', session: 'session-one', created: 0, uploads: 0, taskPosts: 0, uploadBody: '', rotationPending: false, completed: false, cancelled: false, params: null as Record<string, unknown> | null };
  const server = createServer((request: IncomingMessage, response: ServerResponse) => {
    const chunks: Buffer[] = [];
    request.on('data', chunk => chunks.push(chunk as Buffer));
    request.on('end', () => {
      const raw = Buffer.concat(chunks);
      const url = new URL(request.url ?? '/', 'http://127.0.0.1');
      const method = request.method ?? 'GET';
      const payload = raw.length ? raw.toString('utf8') : '';
      const send = (status: number, body: unknown, headers: Record<string, string> = {}) => {
        response.writeHead(status, { 'content-type': 'application/json', ...headers });
        response.end(body === null ? undefined : JSON.stringify(body));
      };
      const fail = (status: number, code: string) => send(status, { error: { code, message: code } });
      const cookie = request.headers.cookie ?? '';
      const writes = !['GET', 'HEAD', 'OPTIONS'].includes(method);
      if (writes && request.headers.origin !== base) return fail(403, 'origin_rejected');
      if (url.pathname === '/api/v1/auth/login') {
        const body = JSON.parse(payload) as { identifier?: string; password?: string };
        if (body.identifier !== 'tester' || body.password !== 'secret') return fail(401, 'invalid_credentials');
        return send(200, { user: { id: 'user-1', username: 'tester' }, csrf_token: state.csrf, expires_at: new Date(Date.now() + 3600000).toISOString() }, { 'set-cookie': '__Host-theta_session=' + state.session + '; Path=/; HttpOnly; Secure' });
      }
      if (cookie !== '__Host-theta_session=' + state.session) return fail(401, 'unauthorized');
      if (url.pathname === '/api/v1/auth/me') return send(200, { user: { id: 'user-1', username: 'tester' }, csrf_token: state.csrf });
      if (writes && request.headers['x-csrf-token'] !== state.csrf) return fail(403, 'csrf_failed');
      if (url.pathname === '/api/v1/projects' && method === 'GET') return send(200, { items: [], page: 1, page_size: 100, has_more: false });
      if (url.pathname === '/api/v1/projects' && method === 'POST') { state.created++; state.rotationPending = true; return send(201, { id: PROJECT, title: (JSON.parse(payload) as { title: string }).title }); }
      if (url.pathname === '/api/v1/projects/' + PROJECT) return send(200, { id: PROJECT, title: 'THETA Agent 验收' });
      if (url.pathname === '/api/v1/projects/' + PROJECT + '/datasets' && method === 'POST') {
        state.uploads++;
        if (state.rotationPending) { state.csrf = 'csrf-two'; state.rotationPending = false; return fail(403, 'csrf_failed'); }
        if (options.duplicate) return fail(409, 'dataset_duplicate');
        const contentType = String(request.headers['content-type'] ?? '');
        const boundary = contentType.split('boundary=')[1] ?? '';
        const filePart = raw.toString('binary').split('--' + boundary).find(part => part.includes('name="file"')) ?? '';
        const bytes = filePart.slice(filePart.indexOf('\r\n\r\n') + 4, filePart.lastIndexOf('\r\n'));
        state.uploadBody = bytes;
        const sha = createHash('sha256').update(Buffer.from(bytes, 'binary')).digest('hex');
        return send(201, { dataset_ref: DATASET, display_name: 'business-acceptance.csv', sha256: sha, status: 'ready' });
      }
      if (url.pathname === '/api/v1/projects/' + PROJECT + '/datasets' && method === 'GET') return send(200, { items: [{ dataset_ref: 'existing-ref', sha256: fixtureSha }], page: 1, page_size: 100, has_more: false });
      if (url.pathname === '/api/v1/tasks' && method === 'POST') {
        state.taskPosts++;
        state.params = (JSON.parse(payload) as { params: Record<string, unknown> }).params;
        return send(201, { id: 1, status: 'queued', phase: 'scheduling', progress: 0 });
      }
      if (url.pathname === '/api/v1/tasks/1') return send(200, { id: 1, status: state.completed ? 'completed' : state.cancelled ? 'cancelled' : 'running', phase: state.completed ? 'completed' : 'training', progress: state.completed ? 100 : 45 });
      if (url.pathname === '/api/v1/tasks/1/cancel') { state.cancelled = true; return send(200, { id: 1, status: 'cancelled', phase: 'cancelled', progress: 0 }); }
      if (url.pathname === '/api/v1/tasks/1/model/download') return send(200, { task_id: 1, download_url: base + '/api/v1/local-storage/token' });
      return fail(404, 'not_found');
    });
  });
  let base = '';
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve));
  base = 'http://127.0.0.1:' + (server.address() as AddressInfo).port;
  return { base, state, close: () => new Promise<void>(resolve => server.close(() => resolve())) };
}

test('Business API gateway completes the documented flow over real HTTP, refreshing CSRF and verifying the uploaded hash', async () => {
  const home = mkdtempSync(path.join(os.tmpdir(), 'theta-business-http-'));
  const fixture = path.join(home, 'business-acceptance.csv');
  const bytes = Buffer.from('text,timestamp\nlate parcel delivery,2026-08-01\nrefund not received,2026-08-02\n');
  writeFileSync(fixture, bytes);
  const sha256 = createHash('sha256').update(bytes).digest('hex');
  const server = await fakeBusinessApi(sha256);
  const bindingsPath = path.join(home, 'bindings.json');
  writeFileSync(bindingsPath, JSON.stringify({ projectTitle: 'THETA Agent 验收', allowProjectCreate: true, models: { lda: { modelId: 1 } } }));
  const store = new ResearchStore(home);
  const approvals = new EffectApprovals(store);
  const options = { baseUrl: server.base, identifier: 'tester', password: 'secret', bindingsPath };
  const dataset = { datasetRef: 'acceptance', sha256, fileName: 'business-acceptance.csv', managedPath: fixture, sizeBytes: bytes.length };
  const request = {
    jobId: 'job-http', runId: 'run-http',
    execution: { computeConfigurationFingerprint: businessApiConfigurationFingerprint(options) },
    dataset, plan: { modelId: 'lda', textColumn: 'text', params: { num_topics: 5, max_iter: 20 }, rationale: 'acceptance', timeoutSeconds: 60 },
  };
  try {
    const gateway = new BusinessApiComputeGateway(store, fetch, options);
    const account = await gateway.authenticate();
    assert.equal(account.user.username, 'tester');
    assert.equal(account.csrfAvailable, true);
    assert.equal(await gateway.ensureProject(), PROJECT);
    assert.equal(server.state.created, 1, '缺少项目时应按 allowProjectCreate 创建一次');
    assert.equal(await gateway.ensureDataset(dataset), DATASET);
    assert.equal(server.state.uploads, 2, 'CSRF 轮换后应刷新并重试一次上传');
    assert.equal(createHash('sha256').update(Buffer.from(server.state.uploadBody, 'binary')).digest('hex'), sha256, '远端收到的字节必须与本地托管副本一致');
    const permitFor = (payload: unknown) => {
      const value = approvals.request('session', { action: 'compute.submit', target: server.base, payload, summary: 'acceptance' });
      return approvals.decide(value.id, 'session', value.hash, true);
    };
    const rejected = { ...request, jobId: 'job-http-rejected', plan: { ...request.plan, params: { num_topics: 1 } } };
    await assert.rejects(gateway.submit(rejected, permitFor(rejected)), /下限/);
    assert.equal(server.state.taskPosts, 0, '本地拒绝的方案不得创建远端任务');
    const job = await gateway.submit(request, permitFor(request));
    assert.equal(job.status, 'queued');
    assert.deepEqual(server.state.params, { 'input.text_column': 'text', num_topics: 5, max_iter: 20 });
    assert.equal((await gateway.status('job-http')).percent, 45);
    assert.equal((await gateway.cancel('job-http')).status, 'cancelled');
    server.state.completed = true; server.state.cancelled = false;
    const results = await gateway.results('job-http', 'summary') as { download: { download_url: string }; limitations: string[] };
    assert.match(results.download.download_url, /model\/download|local-storage/u);
    assert.match(results.limitations[0]!, /manifest/);
  } finally { await server.close(); rmSync(home, { recursive: true, force: true }); }
});

test('Business API gateway reuses the registered dataset when the server reports dataset_duplicate', async () => {
  const home = mkdtempSync(path.join(os.tmpdir(), 'theta-business-duplicate-'));
  const fixture = path.join(home, 'business-acceptance.csv');
  const bytes = Buffer.from('text,timestamp\nlate parcel delivery,2026-08-01\n');
  writeFileSync(fixture, bytes);
  const sha256 = createHash('sha256').update(bytes).digest('hex');
  const server = await fakeBusinessApi(sha256, { duplicate: true });
  const bindingsPath = path.join(home, 'bindings.json');
  writeFileSync(bindingsPath, JSON.stringify({ projectId: PROJECT, models: { lda: { modelId: 1 } } }));
  const options = { baseUrl: server.base, identifier: 'tester', password: 'secret', bindingsPath };
  const dataset = { datasetRef: 'acceptance', sha256, fileName: 'business-acceptance.csv', managedPath: fixture, sizeBytes: bytes.length };
  try {
    const gateway = new BusinessApiComputeGateway(new ResearchStore(home), fetch, options);
    assert.equal(await gateway.ensureDataset(dataset), 'existing-ref', '409 dataset_duplicate 后应按 sha256 复用已登记数据集');
    assert.equal(server.state.uploads, 1, '不重复上传相同内容');
  } finally { await server.close(); rmSync(home, { recursive: true, force: true }); }
});
