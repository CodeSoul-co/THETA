import type { ModelParameters } from './model-parameters.ts'

export interface AnalysisConfig {
  plotLanguage: string
  stopwords?: { id: number | string; name: string; count: number }
  models: string[]
  modelSize: string
  embeddingProvider: 'local' | 'cloud'
  cloudConfirmed: boolean
  cloudSelection?: { provider: string; endpoint: string; model: string }
  externalRequestLimit: number
  mode: 'zero_shot' | 'unsupervised' | 'supervised'
  vocabSize: number
  parameters: Record<string, ModelParameters>
  textColumn?: string
  timeColumn?: string
  labelColumn?: string
  covariates?: string[]
}
export interface EditorTrainingPlan {
  modelId: string; params: ModelParameters; textColumn: string; timeColumn?: string; labelColumn?: string; covariates?: string[]
  rationale: string; timeoutSeconds: number; device?: string; externalRequestLimit?: number
}
export const DEFAULT_ANALYSIS_CONFIG: AnalysisConfig = {
  plotLanguage: 'zh', models: ['theta'], vocabSize: 5000, modelSize: '0.6B',
  embeddingProvider: 'local', cloudConfirmed: false, externalRequestLimit: 200,
  mode: 'zero_shot', parameters: {},
}
export function configFromPlans(plans: EditorTrainingPlan[]): AnalysisConfig {
  const first = plans[0]; const theta = plans.find(plan => plan.modelId === 'theta')
  return { ...DEFAULT_ANALYSIS_CONFIG,
    models: plans.map(plan => plan.modelId), parameters: Object.fromEntries(plans.map(plan => [plan.modelId, { ...plan.params }])),
    textColumn: first.textColumn, timeColumn: plans.find(plan => plan.timeColumn)?.timeColumn,
    labelColumn: plans.find(plan => plan.labelColumn)?.labelColumn, covariates: plans.find(plan => plan.modelId === 'stm')?.covariates ?? [],
    plotLanguage: ['en', 'english'].includes(String(first.params.language)) ? 'en' : 'zh',
    vocabSize: Number(first.params['prepare.vocab_size'] ?? first.params.vocab_size ?? 5000),
    modelSize: String(theta?.params.model_size ?? '0.6B'),
    embeddingProvider: theta?.params.embedding_provider === 'cloud' ? 'cloud' : 'local',
    mode: (theta?.params.mode ?? 'zero_shot') as AnalysisConfig['mode'], externalRequestLimit: theta?.externalRequestLimit ?? 200,
  }
}
export function plansFromConfig(config: AnalysisConfig, originals: EditorTrainingPlan[]): EditorTrainingPlan[] {
  return config.models.map(modelId => {
    const original = originals.find(plan => plan.modelId === modelId)
    const params = { ...config.parameters[modelId] }
    // The shared controls own these values; hidden legacy values must not override them.
    for (const key of ['prepare.vocab_size', 'vocab_size', 'language', 'lang', 'mode', 'model_size', 'embedding_provider', 'embedding_cloud_provider', 'embedding_model', 'embedding_api_base', 'embedding_api_key_env', 'text.stopwords']) delete params[key]
    params.vocab_size = config.vocabSize; params.language = config.plotLanguage
    if (modelId === 'theta') Object.assign(params, { mode: config.mode, model_size: config.modelSize, embedding_provider: config.embeddingProvider })
    return { modelId, params, textColumn: config.textColumn ?? originals[0].textColumn,
      ...(config.timeColumn ? { timeColumn: config.timeColumn } : {}), ...(config.labelColumn ? { labelColumn: config.labelColumn } : {}),
      ...(modelId === 'stm' ? { covariates: config.covariates ?? [] } : {}),
      rationale: original?.rationale ?? '用户在训练配置中追加的模型', timeoutSeconds: original?.timeoutSeconds ?? 43200,
      device: original?.device ?? 'cpu', ...(modelId === 'theta' ? { externalRequestLimit: config.externalRequestLimit } : {}),
    }
  })
}
