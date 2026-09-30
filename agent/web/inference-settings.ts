import { createRequire } from 'node:module';
import path from 'node:path';
import { inferenceProviderSpecs } from '../src/providers/provider-registry.js';
import { createConfiguredProvider } from '../src/providers/configured-provider.js';

const require = createRequire(import.meta.url);
const { readSettings, saveInference } = require('../../runtime/settings.cjs');
const { createCredentials } = require('../../runtime/credentials.cjs');

/** The local Web UI uses the same validation/encryption as the desktop bridge. */
export function localInferenceSettings(home: string) {
  const file = path.join(home, 'settings.json'), credentials = createCredentials(home);
  const specs = [...inferenceProviderSpecs, { id: 'openai-compatible', displayName: 'Custom / Local', defaultBaseUrl: 'http://127.0.0.1:11434/v1', defaultModel: '', defaultModels: [], envApiKey: '', envBaseUrl: '', envModel: '' }];
  const selectedId = () => {
    const provider = createConfiguredProvider();
    return provider?.id === 'gpt' ? 'openai' : provider?.id ?? null;
  };
  function apply() {
    const value = readSettings(file), selected = value.providers[value.providerId];
    if (!selected) return;
    for (const key of ['THETA_INFERENCE_API_KEY', 'THETA_INFERENCE_BASE_URL', 'THETA_INFERENCE_MODEL', 'THETA_INFERENCE_PROVIDER']) delete process.env[key];
    if (selected.encryptedKey) Object.assign(process.env, { THETA_INFERENCE_PROVIDER: value.providerId,
      THETA_INFERENCE_BASE_URL: selected.baseUrl, THETA_INFERENCE_MODEL: selected.model,
      THETA_INFERENCE_API_KEY: credentials.decrypt(selected.encryptedKey) });
    else {
      process.env.THETA_INFERENCE_PROVIDER = 'disabled';
      const spec = specs.find(spec => (spec.id === 'gpt' ? 'openai' : spec.id) === value.providerId);
      if (spec?.envApiKey) delete process.env[spec.envApiKey];
    }
  }
  function catalog() {
    const value = readSettings(file), selected = selectedId() ?? (value.providers[value.providerId] ? value.providerId : null);
    const genericId = process.env.THETA_INFERENCE_API_KEY && process.env.THETA_INFERENCE_BASE_URL
      ? selectedId() : undefined;
    return { kind: 'inference.provider.list', selection: selected ? { providerId: selected, model: createConfiguredProvider()?.model ?? value.providers[selected]?.model ?? '', source: 'local Web' } : null,
      providers: specs.map(spec => {
        const id = spec.id === 'gpt' ? 'openai' : spec.id, saved = value.providers[id];
        const generic = id === genericId;
        const model = saved?.model ?? (generic ? process.env.THETA_INFERENCE_MODEL : process.env[spec.envModel]) ?? spec.defaultModel;
        const baseUrl = saved?.baseUrl ?? (generic ? process.env.THETA_INFERENCE_BASE_URL : process.env[spec.envBaseUrl] || process.env[spec.envBaseUrl.replace(/_API_BASE$/, '_BASE_URL')]) ?? spec.defaultBaseUrl;
        const configured = saved ? Boolean(saved.encryptedKey) : Boolean(generic || process.env[spec.envApiKey]);
        return { id, displayName: spec.displayName, baseUrl, configured, credentialConfigured: configured, configuredModel: model,
          selected: selected === id, local: false, category: 'compatible', models: saved?.models ?? [...new Set([model, ...spec.defaultModels])].filter(Boolean),
          capabilities: { streaming: false, reasoning: true, reasoningEffort: false } };
      }) };
  }
  apply();
  return { catalog, save(input: Record<string, unknown>) {
    const id = String(input.providerId ?? ''), current = catalog().providers.find(provider => provider.id === id);
    if (!current) throw new Error('未知模型供应商');
    const spec = specs.find(spec => (spec.id === 'gpt' ? 'openai' : spec.id) === id)!;
    const generic = process.env.THETA_INFERENCE_API_KEY && selectedId() === id;
    // Preserve a server-provided key when the masked field is left untouched.
    const key = input.apiKey || (!readSettings(file).providers[id] && (generic ? process.env.THETA_INFERENCE_API_KEY : process.env[spec.envApiKey]));
    saveInference(file, { ...current, ...input, baseUrl: input.baseUrl || current.baseUrl, model: input.model || current.configuredModel, apiKey: key }, credentials.encrypt);
    apply();
  } };
}
