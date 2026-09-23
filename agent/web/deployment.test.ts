import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { once } from 'node:events';
import { createAgentServer } from './server.js';
import { MiniMaxInferenceProvider } from '../src/providers/minimax.js';
import { loadAgentServerOptions } from './deployment.js';

const provider = new MiniMaxInferenceProvider({
  apiKey: 'test', baseUrl: 'https://example.invalid/v1', model: 'fixture',
  fetchImpl: async () => new Response(JSON.stringify({ choices: [{ message: { content: 'ok' } }] }), { status: 200 }),
});

test('deployed configuration fails closed when authentication boundaries are missing', () => {
  assert.throws(() => loadAgentServerOptions({ THETA_AGENT_SERVICE_MODE: 'deployed' }), /THETA_BUSINESS_API_URL/u);
  assert.throws(() => loadAgentServerOptions({ THETA_AGENT_SERVICE_MODE: 'deployed', THETA_BUSINESS_API_URL: 'https://api.example' }), /ALLOWED_ORIGINS/u);
  assert.throws(() => loadAgentServerOptions({ THETA_AGENT_SERVICE_MODE: 'deployed', THETA_BUSINESS_API_URL: 'https://api.example', THETA_AGENT_ALLOWED_ORIGINS: 'https://theta.example' }), /TRUSTED_HOSTS/u);
});

test('public API authenticates, enforces CSRF and isolates users', async t => {
  const home = mkdtempSync(path.join(tmpdir(), 'theta-agent-deployed-'));
  const options = {
    mode: 'deployed' as const,
    allowedOrigins: ['https://theta.example'],
    trustedHosts: ['agent.example', '127.0.0.1'],
    requireCsrf: true,
    authenticate: async (request: import('node:http').IncomingMessage) => {
      const id = request.headers.authorization === 'Bearer bob' ? 'bob' : 'alice';
      return { principal: { id }, csrfToken: `csrf-${id}` };
    },
  };
  const server = createAgentServer(home, () => provider, options);
  server.listen(0, '127.0.0.1'); await once(server, 'listening');
  const base = `http://127.0.0.1:${(server.address() as { port: number }).port}`;
  t.after(async () => {
    server.closeAllConnections();
    await new Promise<void>(resolve => server.close(() => resolve()));
    rmSync(home, { recursive: true, force: true });
  });
  const request = async (route: string, user = 'alice', body?: unknown, csrf = true) => {
    const response = await fetch(base + route, {
      method: body === undefined ? 'GET' : 'POST',
      headers: {
        host: 'agent.example', origin: 'https://theta.example', authorization: `Bearer ${user}`,
        ...(body === undefined ? {} : { 'content-type': 'application/json' }),
        ...(body !== undefined && csrf ? { 'x-csrf-token': `csrf-${user}` } : {}),
      },
      ...(body === undefined ? {} : { body: JSON.stringify(body) }),
    });
    return { status: response.status, ...(await response.json()) } as { status: number; ok: boolean; data?: any; error?: { code: string } };
  };

  assert.equal((await request('/healthz')).status, 200);
  const capabilities = await request('/api/v1/agent/capabilities');
  assert.equal(capabilities.status, 200);
  assert.equal(capabilities.data.models.length, 12);
  const lda = capabilities.data.models.find((model: { modelId: string }) => model.modelId === 'lda');
  assert.ok(lda.supportedParameters.includes('num_topics'));
  assert.equal(lda.parameters.num_topics.type, 'int');
  assert.equal(lda.parameters.num_topics.default, 20);
  assert.equal(capabilities.data.compute.workerApiUnchanged, true);
  assert.equal((await request('/api/v3/health')).status, 404);

  const rejected = await request('/api/v1/agent/projects', 'alice', { name: 'Alice project' }, false);
  assert.equal(rejected.status, 403); assert.equal(rejected.error?.code, 'csrf_failed');
  const aliceProject = (await request('/api/v1/agent/projects', 'alice', { name: 'Alice project' })).data;
  const aliceRun = (await request('/api/v1/agent/conversations', 'alice', { projectId: aliceProject.id, researchGoal: 'Alice run' })).data;
  assert.ok(aliceRun.runId);
  assert.equal((await request('/api/v1/agent/runs', 'bob')).data.runs.length, 0);
  assert.equal((await request(`/api/v1/agent/runs/${aliceRun.runId}`, 'bob')).status, 404);
  assert.equal((await request('/api/v1/agent/projects', 'bob')).data.projects.length, 0);
  const consultation = '/api/v1/agent/consultations?scope=manual%3Ashared-name';
  assert.equal((await request(consultation, 'alice', { id: 'private-consultation' }, false)).status, 403);
  assert.equal((await request(consultation, 'alice', { id: 'private-consultation' })).status, 200);
  assert.equal((await request(consultation, 'bob')).data.threads.length, 0);
  const unauthorizedReply = await request('/api/v1/agent/advisory', 'bob', { message: 'read', consultation: { scope: 'manual:shared-name', id: 'private-consultation', requestId: 'r1' } });
  assert.equal(unauthorizedReply.status, 404);
});

test('desktop APIs require the per-launch token even on loopback', async t => {
  const { createManualServer } = await import('./manual-server.js');
  const home = mkdtempSync(path.join(tmpdir(), 'theta-desktop-auth-'));
  const localToken = 'a'.repeat(64);
  const agent = createAgentServer(path.join(home, 'agent'), () => undefined, { mode: 'local', localToken });
  const manual = createManualServer(path.join(home, 'manual'), undefined, { localToken });
  t.after(async () => {
    for (const server of [agent, manual]) {
      server.closeAllConnections();
      await new Promise<void>(resolve => server.close(() => resolve()));
    }
    rmSync(home, { recursive: true, force: true });
  });
  for (const [server, route] of [[agent, '/api/v3/health'], [manual, '/health']] as const) {
    server.listen(0, '127.0.0.1'); await once(server, 'listening');
    const url = `http://127.0.0.1:${(server.address() as { port: number }).port}${route}`;
    assert.equal((await fetch(url)).status, 403);
    assert.equal((await fetch(url, { headers: { 'x-theta-desktop-token': 'wrong' } })).status, 403);
    assert.equal((await fetch(url, { headers: { 'x-theta-desktop-token': localToken } })).status, 200);
    assert.equal((await fetch(url, { headers: { 'x-theta-desktop-token': localToken, origin: 'https://evil.example' } })).status, 403);
  }
});
