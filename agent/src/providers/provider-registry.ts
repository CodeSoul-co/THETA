export interface InferenceProviderSpec {
  id: string;
  displayName: string;
  envApiKey: string;
  defaultBaseUrl: string;
  envBaseUrl: string;
  defaultModels: string[];
  defaultModel: string;
  envModel: string;
  category: 'compatible' | 'direct';
}


export const inferenceProviderSpecs: readonly InferenceProviderSpec[] = [
  {
    id: 'gpt',
    displayName: 'OpenAI',
    envApiKey: 'OPENAI_API_KEY',
    defaultBaseUrl: 'https://api.openai.com/v1',
    envBaseUrl: 'OPENAI_API_BASE',
    defaultModels: ['gpt-4.1-mini', 'gpt-4o-mini', 'gpt-4o'],
    defaultModel: 'gpt-4o-mini',
    envModel: 'OPENAI_MODEL',
    category: 'compatible',
  },
  {
    id: 'deepseek',
    displayName: 'DeepSeek',
    envApiKey: 'DEEPSEEK_API_KEY',
    defaultBaseUrl: 'https://api.deepseek.com',
    envBaseUrl: 'DEEPSEEK_API_BASE',
    defaultModels: ['deepseek-v4-pro', 'deepseek-v4-flash'],
    defaultModel: 'deepseek-v4-pro',
    envModel: 'DEEPSEEK_MODEL',
    category: 'compatible',
  },
  {
    id: 'glm',
    displayName: 'GLM',
    envApiKey: 'GLM_API_KEY',
    defaultBaseUrl: 'https://open.bigmodel.cn/api/paas/v4',
    envBaseUrl: 'GLM_API_BASE',
    defaultModels: ['glm-4', 'glm-4-plus', 'glm-4-flash'],
    defaultModel: 'glm-4',
    envModel: 'GLM_MODEL',
    category: 'compatible',
  },
  {
    id: 'minimax',
    displayName: 'MiniMax',
    envApiKey: 'MINIMAX_API_KEY',
    defaultBaseUrl: 'https://api.minimax.io/v1',
    envBaseUrl: 'MINIMAX_API_BASE',
    defaultModels: ['MiniMax-M2.7'],
    defaultModel: 'MiniMax-M2.7',
    envModel: 'MINIMAX_MODEL',
    category: 'direct',
  },
] as const;
