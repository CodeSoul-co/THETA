import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:net';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { serviceEnvironment, portAvailable, root } from './workbench.mjs';

test('固定当前仓库数据与本机三服务，旧目录/远程配置不能切走工作台', () => {
  const env = serviceEnvironment('/repo with spaces', {
    THETA_AGENT_HOME: '/old/.theta_agent', THETA_PROJECT_ROOT: '/trash',
    THETA_COMPUTE_URL: 'https://old', THETA_BUSINESS_API_URL: 'https://old',
    THETA_AGENT_SERVICE_MODE: 'deployed', NODE_ENV: 'production',
    THETA_AGENT_API_URL: 'https://old', NEXT_PUBLIC_LOCAL_NO_AUTH: 'true',
    DEEPSEEK_API_KEY: 'fixture-private-key', THETA_WORKER_NEURAL_PYTHON: '/custom/python',
  });
  assert.equal(env.THETA_AGENT_HOME, '/repo with spaces/.theta_agent');
  assert.equal(env.THETA_PROJECT_ROOT, '/repo with spaces');
  assert.equal(env.THETA_AGENT_API_URL, 'http://127.0.0.1:4318');
  assert.equal(env.THETA_MANUAL_LOCAL_API_URL, 'http://127.0.0.1:4321');
  assert.equal(env.THETA_AGENT_API_HOST, '127.0.0.1');
  assert.equal(env.NODE_ENV, 'development');
  assert.equal(env.THETA_AGENT_SERVICE_MODE, 'local');
  assert.equal(env.THETA_COMPUTE_URL, '');
  assert.equal(env.THETA_BUSINESS_API_URL, '');
  assert.equal(env.NEXT_PUBLIC_LOCAL_NO_AUTH, 'false');
  assert.equal(env.DEEPSEEK_API_KEY, 'fixture-private-key');
  assert.equal(env.THETA_WORKER_NEURAL_PYTHON, '/custom/python');
  assert.equal(env.THETA_PYTHON, '/repo with spaces/agent/.venv/bin/python');
  assert.equal(root, path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..'));
});

test('端口冲突只检查，不关闭已在运行的服务', async () => {
  const server = createServer();
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const port = server.address().port;
  try { assert.equal(await portAvailable(port), false); assert.equal(server.listening, true); }
  finally { await new Promise(resolve => server.close(resolve)); }
  assert.equal(await portAvailable(port), true);
});
