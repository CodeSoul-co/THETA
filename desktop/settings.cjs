const { existsSync, readFileSync, writeFileSync, renameSync } = require('node:fs');
const path = require('node:path');
const MODEL_KEYS = ['QWEN_MODEL_0_6B', 'QWEN_MODEL_4B', 'QWEN_MODEL_8B', 'SBERT_MODEL_PATH'];
const PROVIDER_IDS = ['openai', 'deepseek', 'minimax', 'glm', 'openai-compatible'];
function readSettings(file) {
  if (!existsSync(file)) return { providerId: 'openai', providers: {}, models: {} };
  const value = JSON.parse(readFileSync(file, 'utf8'));
  if (!value.providers) return { providerId: 'openai-compatible', models: value.models ?? {}, providers: { 'openai-compatible': { baseUrl: value.baseUrl, model: value.model, models: [value.model], encryptedKey: value.encryptedKey } } };
  return value;
}
function writeSettings(file, value) {
  writeFileSync(file + '.tmp', JSON.stringify(value, null, 2), { mode: 0o600 });
  renameSync(file + '.tmp', file);
  return value;
}
function saveInference(file, input, encrypt) {
  const previous = readSettings(file);
  const providerId = input.providerId;
  if (!PROVIDER_IDS.includes(providerId)) throw new Error('未知模型供应商');
  const existing = previous.providers[providerId] ?? {};
  const baseUrl = String(input.baseUrl || existing.baseUrl || '').trim();
  const endpoint = new URL(baseUrl);
  if (!['https:', 'http:'].includes(endpoint.protocol) || endpoint.username || endpoint.password) throw new Error('请输入有效的 API 地址');
  if (endpoint.protocol === 'http:' && !['127.0.0.1', 'localhost', '[::1]'].includes(endpoint.hostname)) throw new Error('远程 API 必须使用 HTTPS');
  const model = String(input.model || existing.model || '').trim();
  if (!model || model.length > 200) throw new Error('请填写模型名称');
  const key = String(input.apiKey ?? '').trim();
  const profile = { baseUrl, model, models: [...new Set([...(Array.isArray(input.models) ? input.models : existing.models ?? []), model])].filter(item => typeof item === 'string' && item.length <= 200).slice(0, 50),
    encryptedKey: input.clearApiKey ? undefined : key ? encrypt(key) : existing.encryptedKey };
  return writeSettings(file, { ...previous, providerId, providers: { ...previous.providers, [providerId]: profile } });
}
const EMBEDDING_DEFAULTS = { mode: 'local', localModel: 'Qwen/Qwen3-Embedding-0.6B', localPath: '', sbertPath: '', provider: 'zhipu', baseUrl: 'https://open.bigmodel.cn/api/paas/v4', model: 'embedding-3', dimensions: null };
function embeddingSettings(value) {
  return { ...EMBEDDING_DEFAULTS, localPath: value.models?.QWEN_MODEL_0_6B || '', sbertPath: value.models?.SBERT_MODEL_PATH || '', ...value.embedding };
}
function saveEmbedding(file, input, encrypt) {
  const previous = readSettings(file);
  const existing = embeddingSettings(previous);
  if (!['local', 'cloud'].includes(input.mode) || !['zhipu', 'openai_compatible'].includes(input.provider)) throw new Error('未知 Embedding 模式或供应商');
  const baseUrl = String(input.baseUrl || '').trim().replace(/\/+$/, '').replace(/\/embeddings$/, '');
  const endpoint = new URL(baseUrl);
  if (endpoint.username || endpoint.password || endpoint.search || endpoint.hash || !(endpoint.protocol === 'https:' || endpoint.protocol === 'http:' && ['127.0.0.1', 'localhost', '[::1]'].includes(endpoint.hostname))) throw new Error('Embedding API 需要 HTTPS；本机服务可以使用 HTTP');
  const model = String(input.model || '').trim();
  if (!model) throw new Error('请填写云端 Embedding 模型名称');
  const dimensions = input.dimensions === null || input.dimensions === '' || input.dimensions === undefined ? null : Number(input.dimensions);
  if (dimensions !== null && (!Number.isInteger(dimensions) || dimensions < 1 || dimensions > 65536)) throw new Error('向量维度需要为正整数或留空');
  const paths = {};
  for (const name of ['localPath', 'sbertPath']) {
    const directory = String(input[name] || '').trim();
    if (directory && (!path.isAbsolute(directory) || !existsSync(path.join(directory, 'config.json')))) throw new Error('本地模型目录需要包含 config.json');
    paths[name] = directory;
  }
  const apiKey = String(input.apiKey || '').trim();
  const embedding = { mode: input.mode, localModel: String(input.localModel || EMBEDDING_DEFAULTS.localModel).trim(), ...paths,
    provider: input.provider, baseUrl, model, dimensions,
    encryptedKey: input.clearApiKey ? undefined : apiKey ? encrypt(apiKey) : existing.encryptedKey };
  return writeSettings(file, { ...previous, embedding });
}
module.exports = { MODEL_KEYS, readSettings, saveInference, saveEmbedding, embeddingSettings };
