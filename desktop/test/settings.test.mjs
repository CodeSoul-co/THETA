import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync, mkdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import settings from '../settings.cjs';

test('settings never store plaintext keys and preserve keys when the field is empty', t => {
  const root = mkdtempSync(path.join(tmpdir(), 'theta-settings-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const file = path.join(root, 'settings.json');
  const config = { providerId: 'openai', baseUrl: 'https://api.example.com/v1', model: 'test', models: {}, apiKey: 'private-test-key' };
  settings.saveInference(file, config, () => 'encrypted-fixture');
  assert.ok(!readFileSync(file, 'utf8').includes('private-test-key'));
  settings.saveInference(file, { ...config, apiKey: '' }, () => { throw new Error('should not encrypt'); });
  assert.equal(settings.readSettings(file).providers.openai.encryptedKey, 'encrypted-fixture');
  settings.saveInference(file, { ...config, apiKey: '', clearApiKey: true }, () => 'unused');
  assert.equal(settings.readSettings(file).providers.openai.encryptedKey, undefined);
});

test('settings reject insecure remote endpoints and invalid model directories without losing existing settings', t => {
  const root = mkdtempSync(path.join(tmpdir(), 'theta-settings-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const file = path.join(root, 'settings.json');
  const config = { providerId: 'openai', baseUrl: 'http://127.0.0.1:11434/v1', model: 'local', models: {} };
  settings.saveInference(file, config, () => 'unused');
  const previous = readFileSync(file, 'utf8');
  for (const baseUrl of ['file:///tmp/test', 'http://remote.example.com/v1', 'https://user:password@example.com']) {
    assert.throws(() => settings.saveInference(file, { ...config, baseUrl }, () => 'unused'));
  }

  assert.equal(readFileSync(file, 'utf8'), previous);
  const model = path.join(root, 'model'); mkdirSync(model); writeFileSync(path.join(model, 'config.json'), '{}');
  const embedding = settings.embeddingSettings(settings.readSettings(file));
  assert.equal(embedding.provider, 'zhipu'); assert.equal(embedding.model, 'embedding-3');
  assert.equal(embedding.localModel, 'Qwen/Qwen3-Embedding-0.6B');
  assert.throws(() => settings.saveEmbedding(file, { ...embedding, localPath: '/missing-model' }, () => 'unused'));
  settings.saveEmbedding(file, { ...embedding, localPath: model, mode: 'cloud', apiKey: 'embedding-private' }, () => 'encrypted-embedding');
  assert.equal(settings.readSettings(file).embedding.localPath, model);
  assert.ok(!readFileSync(file, 'utf8').includes('embedding-private'));
  assert.equal(settings.readSettings(file).embedding.encryptedKey, 'encrypted-embedding');
  assert.equal(settings.readSettings(file).providers.openai.model, 'local');
});
