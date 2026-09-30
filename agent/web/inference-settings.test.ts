import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { localInferenceSettings } from './inference-settings.js';
import { once } from 'node:events';
import { createAgentServer } from './server.js';
import { createConfiguredProvider } from '../src/providers/configured-provider.js';

test('Web configuration uses desktop validation, masks and persists keys, and survives restart', () => {
  const before = { ...process.env }, home = mkdtempSync(path.join(tmpdir(), 'theta-web-settings-'));
  try {
    for (const key of Object.keys(process.env)) if (/^(THETA_INFERENCE_|OPENAI_|DEEPSEEK_|GLM_|MINIMAX_)/.test(key)) delete process.env[key];
    process.env.OPENAI_API_KEY = 'server-fixture-key';
    const settings = localInferenceSettings(home);
    assert.equal(settings.catalog().providers.find(item => item.selected)?.id, 'openai');
    settings.save({ providerId: 'openai', baseUrl: 'https://proxy.example/v1', model: 'custom-model' });
    assert.equal(createConfiguredProvider()?.model, 'custom-model');
    assert.equal(settings.catalog().selection?.providerId, 'openai');
    assert.ok(!readFileSync(path.join(home, 'settings.json'), 'utf8').includes('server-fixture-key'));
    assert.ok(!JSON.stringify(settings.catalog()).includes('server-fixture-key'));
    delete process.env.THETA_INFERENCE_API_KEY;
    assert.equal(localInferenceSettings(home).catalog().selection?.model, 'custom-model');
    const snapshot = readFileSync(path.join(home, 'settings.json'), 'utf8');
    assert.throws(() => settings.save({ providerId: 'openai', baseUrl: 'http://unsafe.example/v1', model: 'bad' }), /HTTPS/);
    assert.equal(readFileSync(path.join(home, 'settings.json'), 'utf8'), snapshot);
    settings.save({ providerId: 'openai-compatible', baseUrl: 'http://127.0.0.1:12345/v1', model: 'local', apiKey: 'custom-fixture' });
    settings.save({ providerId: 'openai-compatible', model: 'local-2', apiKey: '' });
    assert.equal(createConfiguredProvider()?.model, 'local-2');
    settings.save({ providerId: 'openai-compatible', clearApiKey: true });
    assert.equal(createConfiguredProvider(), undefined);
  } finally {
    for (const key of Object.keys(process.env)) if (!(key in before)) delete process.env[key];
    Object.assign(process.env, before); rmSync(home, { recursive: true, force: true });
  }
});


test('Web settings HTTP endpoint saves masked profiles without dropping unfinished configuration', async () => {
  const before = { ...process.env }, home = mkdtempSync(path.join(tmpdir(), 'theta-web-settings-http-'));
  let server: ReturnType<typeof createAgentServer> | undefined;
  try {
    for (const key of Object.keys(process.env)) if (/^(THETA_INFERENCE_|THETA_DESKTOP_TOKEN|OPENAI_|DEEPSEEK_|GLM_|MINIMAX_)/.test(key)) delete process.env[key];
    server = createAgentServer(home); server.listen(0, '127.0.0.1'); await once(server, 'listening');
    const url = `http://127.0.0.1:${(server.address() as { port: number }).port}/api/v3/inference/settings`;
    const save = (llm: unknown) => fetch(url, { method: 'PATCH', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ llm }) });
    let response = await save({ providerId: 'glm', baseUrl: 'https://open.bigmodel.cn/api/paas/v4', model: 'fixture-model' });
    assert.equal(response.status, 200);
    let settings = (await response.json()).data;
    assert.equal(settings.llm.model, 'fixture-model'); assert.equal(settings.llm.providerId, 'glm');
    assert.equal(settings.llm.apiKeyConfigured, false);
    response = await save({ providerId: 'glm', apiKey: 'http-fixture-key' }); assert.equal(response.status, 200);
    settings = (await response.json()).data; assert.equal(settings.llm.apiKeyConfigured, true);
    assert.ok(!JSON.stringify(settings).includes('http-fixture-key'));
    response = await save({ providerId: 'glm', apiKey: '', model: 'fixture-updated' }); assert.equal(response.status, 200);
    assert.equal((await response.json()).data.llm.apiKeyConfigured, true);
    response = await save({ providerId: 'glm', baseUrl: 'http://unsafe.example/v1' }); assert.equal(response.status, 400);
    settings = (await (await fetch(url)).json()).data;
    assert.equal(settings.llm.baseUrl, 'https://open.bigmodel.cn/api/paas/v4'); assert.equal(settings.llm.model, 'fixture-updated');
    response = await save({ providerId: 'glm', clearApiKey: true }); assert.equal(response.status, 200);
    settings = (await response.json()).data;
    assert.equal(settings.llm.apiKeyConfigured, false); assert.equal(settings.llm.model, 'fixture-updated');
  } finally {
    if (server) { server.closeAllConnections(); await new Promise<void>(resolve => server!.close(() => resolve())); }
    for (const key of Object.keys(process.env)) if (!(key in before)) delete process.env[key];
    Object.assign(process.env, before); rmSync(home, { recursive: true, force: true });
  }
});
