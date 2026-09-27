import type { ModelParameters } from './model-parameters.ts'

export interface AnalysisConfig {
  plotLanguage: string
  stopwords?: { id: number | string; name: string; count: number }
  models: string[]
  topicExploration?: boolean
  topicCounts?: string
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
  plotLanguage: 'zh', models: ['lda'], vocabSize: 5000, modelSize: '0.6B',
  embeddingProvider: 'local', cloudConfirmed: false, externalRequestLimit: 200,
  mode: 'zero_shot', parameters: {},
}
export function configFromPlans(plans: EditorTrainingPlan[]): AnalysisConfig {
  if (!plans.length) return { ...DEFAULT_ANALYSIS_CONFIG, models: [], parameters: {} }
  const first = plans[0]; const theta = plans.find(plan => plan.modelId === 'theta')
  const models = [...new Set(plans.map(plan => plan.modelId))]
  const counts = [...new Set(plans.map(plan => Number(plan.params[plan.modelId === 'hdp' ? 'max_topics' : 'num_topics'] ?? plan.params['config.num_topics'])).filter(Number.isInteger))]
  const exploration = models.length < plans.length && counts.length > 1
  return { ...DEFAULT_ANALYSIS_CONFIG, topicExploration: exploration, topicCounts: exploration ? counts.join(', ') : '',
    models, parameters: Object.fromEntries(plans.map(plan => [plan.modelId, { ...plan.params }])),
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
  const counts = explorationCounts(config)
  return config.models.flatMap(modelId => (counts.length ? counts : [undefined]).map(count => {
    const original = originals.find(plan => plan.modelId === modelId)
    const params = { ...config.parameters[modelId] }
    // The shared controls own these values; hidden legacy values must not override them.
    for (const key of ['prepare.vocab_size', 'vocab_size', 'language', 'lang', 'mode', 'model_size', 'embedding_provider', 'embedding_cloud_provider', 'embedding_model', 'embedding_api_base', 'embedding_api_key_env', 'text.stopwords']) delete params[key]
    params.vocab_size = config.vocabSize; params.language = config.plotLanguage
    if (modelId === 'theta') Object.assign(params, { mode: config.mode, model_size: config.modelSize, embedding_provider: config.embeddingProvider })
    if (count !== undefined) {
      for (const key of ['num_topics', 'max_topics', 'nr_topics', 'config.num_topics']) delete params[key]
      params[modelId === 'hdp' ? 'max_topics' : 'num_topics'] = count
    }
    return { modelId, params, textColumn: config.textColumn ?? originals[0].textColumn,
      ...(config.timeColumn ? { timeColumn: config.timeColumn } : {}), ...(config.labelColumn ? { labelColumn: config.labelColumn } : {}),
      ...(modelId === 'stm' ? { covariates: config.covariates ?? [] } : {}),
      rationale: original?.rationale ?? '用户在训练配置中追加的模型', timeoutSeconds: original?.timeoutSeconds ?? 43200,
      device: original?.device ?? 'cpu', ...(modelId === 'theta' ? { externalRequestLimit: config.externalRequestLimit } : {}),
    }
  }))
}

export function explorationCounts(config: Pick<AnalysisConfig, 'topicExploration' | 'topicCounts' | 'models'>): number[] {
  if (!config.topicExploration) return []
  const tokens = (config.topicCounts ?? '').trim().split(/[\s,，、;；]+/).filter(Boolean)
  if (!tokens.length || tokens.some(value => !/^\d+$/.test(value) || Number(value) < 2 || Number(value) > 500)) throw new Error('请填写 2–500 的整数主题数，例如 5、10、15、20')
  const values = [...new Set(tokens.map(Number))]
  if (values.length > 12 || values.length * config.models.length > 48) throw new Error('最多 12 个主题数、48 组模型实验，请减少组合数量')
  return values
}
