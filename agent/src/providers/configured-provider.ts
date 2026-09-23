import { MiniMaxInferenceProvider } from './minimax.js';
import { inferenceProviderSpecs } from './provider-registry.js';

type Environment = Readonly<Record<string, string | undefined>>;
const text = (env: Environment, key: string): string | undefined => env[key]?.trim() || undefined;
const canonical = (id: string): string => ({ openai: 'gpt', 'openai-compatible': 'custom' }[id.toLowerCase()] ?? id.toLowerCase());
const genericFields = ['THETA_INFERENCE_API_KEY', 'THETA_INFERENCE_BASE_URL', 'THETA_INFERENCE_MODEL'] as const;

export const providerIdForBaseUrl = (baseUrl: string): string => {
  const hostname = new URL(baseUrl).hostname;
  return inferenceProviderSpecs.find((spec) => new URL(spec.defaultBaseUrl).hostname === hostname)?.id ?? 'openai-compatible';
};

/** Select one complete credential/endpoint profile; never mix two suppliers. */
export const createConfiguredProvider = (env: Environment = process.env): MiniMaxInferenceProvider | undefined => {
  const genericKey = text(env, 'THETA_INFERENCE_API_KEY');
  const selected = text(env, 'THETA_INFERENCE_PROVIDER');
  if (genericKey) {
    const baseUrl = text(env, 'THETA_INFERENCE_BASE_URL');
    const model = text(env, 'THETA_INFERENCE_MODEL');
    if (!baseUrl || !model) throw new Error('请同时配置 THETA_INFERENCE_BASE_URL 和 THETA_INFERENCE_MODEL。');
    const providerId = selected ? canonical(selected) : providerIdForBaseUrl(baseUrl);
    return new MiniMaxInferenceProvider({ apiKey: genericKey, baseUrl, model, providerId: providerId === 'custom' ? 'openai-compatible' : providerId });
  }
  const spec = selected
    ? inferenceProviderSpecs.find((candidate) => candidate.id === canonical(selected))
    : ['deepseek', 'minimax', 'gpt', 'glm'].map((id) => inferenceProviderSpecs.find((candidate) => candidate.id === id)!).find((candidate) => text(env, candidate.envApiKey));
  if (!spec) {
    if (selected) throw new Error('该供应商需要完整的自定义 API 地址、模型名和密钥。输入 /model 进行配置。');
    return undefined;
  }
  const apiKey = text(env, spec.envApiKey);
  if (!apiKey) {
    if (selected) throw new Error(`${spec.displayName} 尚未配置 ${spec.envApiKey}，请检查本地环境文件。`);
    return undefined;
  }
  const timeout = text(env, `${spec.id === 'gpt' ? 'OPENAI' : spec.id.toUpperCase()}_TIMEOUT_MS`);
  return new MiniMaxInferenceProvider({
    apiKey, providerId: spec.id, providerName: spec.displayName,
    baseUrl: text(env, spec.envBaseUrl) ?? text(env, spec.envBaseUrl.replace(/_API_BASE$/u, '_BASE_URL')) ?? spec.defaultBaseUrl,
    model: text(env, spec.envModel) ?? spec.defaultModel,
    ...(timeout ? { timeoutMs: Number(timeout) } : {}),
  });
};

/** Public inventory deliberately excludes credentials and URLs. */
export const configuredProviderSummaries = (env: Environment = process.env) => inferenceProviderSpecs.map((spec) => ({
  id: spec.id === 'gpt' ? 'openai' : spec.id,
  name: spec.displayName,
  model: text(env, spec.envModel) ?? spec.defaultModel,
  configured: Boolean(text(env, spec.envApiKey)),
}));

/** Process-only selection; independent provider credentials stay untouched. */
export const selectConfiguredProvider = (id: string, env: NodeJS.ProcessEnv = process.env): MiniMaxInferenceProvider => {
  const candidate: NodeJS.ProcessEnv = { ...env, THETA_INFERENCE_PROVIDER: canonical(id) };
  for (const key of genericFields) delete candidate[key];
  const provider = createConfiguredProvider(candidate);
  if (!provider) throw new Error('该供应商尚未配置。');
  for (const key of genericFields) delete env[key];
  env.THETA_INFERENCE_PROVIDER = canonical(id);
  return provider;
};
