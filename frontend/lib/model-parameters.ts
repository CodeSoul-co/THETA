export type ParameterValue = string | number | boolean | number[] | null
export type ModelParameters = Record<string, ParameterValue>
export interface ParameterSpec {
  type: string; default: ParameterValue; choices?: (string | number)[] | null
  target: string; source: string; note?: string; nullable?: boolean
}
export interface ModelContract { modelId: string; description: string; parameters: Record<string, ParameterSpec> }

const labels: Record<string, string> = {
  num_topics: '主题数', max_topics: '主题截断上限', max_iter: '最大迭代次数', n_iter: '采样迭代次数',
  epochs: '训练轮数', batch_size: '批大小', learning_rate: '学习率', hidden_dim: '隐藏层维度', hidden_sizes: '各隐藏层宽度',
  dropout: '随机失活比例', num_layers: '隐藏层层数', patience: '早停耐心值', no_early_stopping: '禁用早停', early_stopping: '启用早停',
  embedding_dim: '词向量维度', word_embedding_dim: '词嵌入维度', alpha: '文档主题先验', beta: '主题词先验', eta: '词分布先验',
  learning_method: '学习方法', random_state: '随机种子', seed: '随机种子', n_jobs: '并行线程数',
  inference_type: '推断方式', n_neighbors: '近邻数量', n_components: '降维维度', min_cluster_size: '最小聚类规模', min_samples: '密度采样阈值', top_n_words: '每主题关键词数',
  time_slices: '时间片数量', evolution_weight: '时间演化权重', variance: '演化方差',
  kl_start: 'KL 退火初始权重', kl_end: 'KL 退火最终权重', kl_warmup: 'KL 预热轮数', kl_weight: 'KL 损失权重',
  encoder_dropout: '编码器失活比例', encoder_activation: '编码器激活函数', activation: '激活函数',
  train_word_embeddings: '训练词嵌入', train_embeddings: '训练词向量', use_pretrained_embeddings: '使用预训练词向量',
  contrastive_weight: '对比学习权重', contrastive_temp: '对比学习温度', weight_decay: '权重衰减',
  stage1_epochs: '第一阶段训练轮数', stage2_epochs: '第二阶段训练轮数', stage1_lr: '第一阶段学习率', stage2_lr: '第二阶段学习率',
  lora_r: '低秩适配秩', lora_alpha: '低秩适配缩放系数', lora_dropout: '低秩适配失活比例',
  min_delta: '早停最小改进量', use_scheduler: '启用学习率调度', scheduler_patience: '调度等待轮数', scheduler_factor: '学习率衰减系数',
  train_ratio: '训练集比例', val_ratio: '验证集比例', test_ratio: '测试集比例', num_workers: '数据加载进程数', workers: '工作进程数',
  pin_memory: '锁页内存', persistent_workers: '持久数据加载进程', prefetch_factor: '预取批次数',
  max_length: '嵌入窗口长度', clean: '启用文本清洗', with_time: '准备时间数据', bow_only: '仅准备词袋', skip_sbert: '跳过语义嵌入',
  skip_eval: '跳过评估', skip_viz: '跳过可视化', calculate_probabilities: '计算文档主题概率', verbose: '详细日志',
  dev_mode: '快速开发模式', dev: '快速开发模式', model_type: '主题分布形式', learn_priors: '学习先验',
  gamma: '全局浓度参数', kappa: '学习衰减指数', tau: '学习偏移量', K: '局部主题截断数',
  max_doc_words: '每文档词数上限', min_count: '最低词频', window: '上下文窗口', window_size: '词对窗口',
  enable_temporal: '启用时间分析', no_train_word_embeddings: '冻结词嵌入', no_pin_memory: '关闭锁页内存', no_persistent_workers: '关闭持久加载进程',
  normalize: '嵌入归一化', embedding_dimensions: '云嵌入维度', model_path: '本地模型路径',
}
export const parameterLabel = (key: string) => labels[key.split('.').at(-1)!] ?? '模型参数'
export const parameterChoiceLabel = (value: string | number) => ({ true: '开启', false: '关闭', batch: '批量学习', online: '在线学习', zeroshot: '零样本推断', combined: '联合推断', relu: '线性整流', softplus: '平滑整流', gelu: '高斯误差线性', tanh: '双曲正切', sigmoid: '逻辑函数' }[String(value)] ?? String(value))
// These are controlled by the data/embedding panels, never arbitrary credential fields.
const managed = new Set(['vocab_size', 'prepare.vocab_size', 'language', 'lang', 'mode', 'model_size', 'text.stopwords',
  'embedding_provider', 'embedding_cloud_provider', 'embedding_model', 'embedding_api_base', 'embedding_api_key_env'])
export function editableParameters(contract: ModelContract) {
  return Object.entries(contract.parameters).filter(([key]) => !managed.has(key))
}
export function parameterGroup(key: string) {
  if (key.startsWith('prepare.')) return '预处理参数'
  if (key.includes('.') || key.startsWith('skip_')) return '模型专属与高级参数'
  return '训练参数'
}
export function parseParameter(raw: string, spec: ParameterSpec): ParameterValue | undefined {
  if (raw.trim() === '') return undefined
  if (spec.nullable && raw.trim().toLowerCase() === 'null') return null
  if (spec.type === 'array') {
    const values = raw.split(/[,，\s]+/).filter(Boolean).map(Number)
    if (!values.length || values.some(v => !Number.isInteger(v) || v <= 0)) throw new Error('请输入用逗号分隔的正整数')
    return values
  }
  if (spec.type === 'int' || spec.type === 'float') {
    const number = Number(raw)
    if (!Number.isFinite(number) || (spec.type === 'int' && !Number.isInteger(number))) throw new Error(spec.type === 'int' ? '请输入整数' : '请输入有效数字')
    return number
  }
  if (spec.type === 'bool') return raw === 'true'
  return raw
}
