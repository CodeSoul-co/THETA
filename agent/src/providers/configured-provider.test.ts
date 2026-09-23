import assert from 'node:assert/strict';
import test from 'node:test';
import { createConfiguredProvider, configuredProviderSummaries, selectConfiguredProvider } from './configured-provider.js';

test('provider selection prefers configured DeepSeek, preserves explicit choice and fails closed', () => {
  const env = { DEEPSEEK_API_KEY: 'fixture-deepseek', MINIMAX_API_KEY: 'fixture-minimax', OPENAI_API_KEY: 'fixture-openai' };
  assert.equal(createConfiguredProvider(env)?.id, 'deepseek');
  assert.equal(createConfiguredProvider({ ...env, THETA_INFERENCE_PROVIDER: 'openai' })?.id, 'gpt');
  assert.equal(createConfiguredProvider({ MINIMAX_API_KEY: 'fixture' })?.id, 'minimax');
  assert.equal(createConfiguredProvider({}), undefined);
  assert.throws(() => createConfiguredProvider({ ...env, THETA_INFERENCE_PROVIDER: 'glm' }), /GLM_API_KEY/);
  assert.ok(!JSON.stringify(configuredProviderSummaries(env)).includes('fixture'));
});

test('switching removes stale generic overrides without changing named credentials; failure is atomic', () => {
  const env = { THETA_INFERENCE_PROVIDER: 'deepseek', THETA_INFERENCE_API_KEY: 'fixture-old', THETA_INFERENCE_BASE_URL: 'https://proxy.example/v1', THETA_INFERENCE_MODEL: 'old', DEEPSEEK_API_KEY: 'fixture-deepseek', OPENAI_API_KEY: 'fixture-openai' };
  assert.equal(selectConfiguredProvider('openai', env).id, 'gpt');
  assert.equal(env.DEEPSEEK_API_KEY, 'fixture-deepseek');
  assert.equal(env.OPENAI_API_KEY, 'fixture-openai');
  assert.equal(env.THETA_INFERENCE_API_KEY, undefined);
  const snapshot = { ...env };
  assert.throws(() => selectConfiguredProvider('glm', env), /GLM_API_KEY/);
  assert.deepEqual(env, snapshot);
});

test('each supplier uses its own credential, endpoint and completion contract', async (t) => {
  const calls: Array<{ url: string; key: string; body: Record<string, unknown> }> = [];
  t.mock.method(globalThis, 'fetch', async (url: string, init: RequestInit) => {
    calls.push({ url, key: (init.headers as Record<string, string>).Authorization, body: JSON.parse(String(init.body)) });
    return new Response(JSON.stringify({ id: 'fixture', choices: [{ message: { content: '{"ok":true}' } }] }), { status: 200 });
  });
  const env = { DEEPSEEK_API_KEY: 'fixture-d', DEEPSEEK_BASE_URL: 'https://deepseek-proxy.example/v1', OPENAI_API_KEY: 'fixture-o', MINIMAX_API_KEY: 'fixture-m', GLM_API_KEY: 'fixture-g' };
  for (const id of ['deepseek', 'openai', 'minimax', 'glm']) {
    const provider = createConfiguredProvider({ ...env, THETA_INFERENCE_PROVIDER: id })!;
    await provider.infer({ runId: 'test', stepId: 'test', modelAlias: 'test', input: { messages: [{ role: 'user', content: 'test' }] } });
  }
  assert.deepEqual(calls.map((call) => call.key), ['Bearer fixture-d', 'Bearer fixture-o', 'Bearer fixture-m', 'Bearer fixture-g']);
  assert.equal(calls[0].url, 'https://deepseek-proxy.example/v1/chat/completions');
  assert.ok(calls[1].url.startsWith('https://api.openai.com/'));
  assert.ok(calls[2].url.startsWith('https://api.minimax.io/'));
  assert.ok(calls[3].url.startsWith('https://open.bigmodel.cn/'));
  assert.equal(calls[1].body.max_completion_tokens, 4096);
  for (const index of [0, 2, 3]) {
    assert.equal(calls[index].body.max_tokens, 4096);
    assert.equal('max_completion_tokens' in calls[index].body, false);
  }
});

test('generic compatible endpoints and explicit DeepSeek proxies retain their protocol', () => {
  const env = { THETA_INFERENCE_API_KEY: 'fixture', THETA_INFERENCE_BASE_URL: 'http://127.0.0.1:11434/v1', THETA_INFERENCE_MODEL: 'local-model' };
  assert.equal(createConfiguredProvider(env)?.id, 'openai-compatible');
  assert.equal(createConfiguredProvider({ ...env, THETA_INFERENCE_BASE_URL: 'https://api.deepseek.com' })?.id, 'deepseek');
  assert.equal(createConfiguredProvider({ ...env, THETA_INFERENCE_PROVIDER: 'deepseek' })?.id, 'deepseek');
  assert.throws(() => createConfiguredProvider({ ...env, THETA_INFERENCE_MODEL: '' }), /THETA_INFERENCE_MODEL/);
});
