/** Chinese presentation only. Never use these labels as API keys or artifact paths. */
export function conversationDisplayText(value: string): string {
  // Keep model-facing citations and link destinations intact in stored messages.
  const visible = value.replace(/\[\[cite\]\][\s\S]*?(?:\[\[\/cite\]\]|$)/gu, '')
  return visible.split(/(\]\([^\n]*?\))/gu).map((part, index) => index % 2 ? part
    : part.replace(/\bjob-[0-9a-f]{6,}(?:-[0-9a-f]+)*(?:…|\.\.\.)?/giu, '本次任务')).join('').trim()
}

const keyOf = (value: string) => value.replace(/([a-z])([A-Z])/g, '$1 $2').replace(/[_\-.]+/g, ' ').replace(/\s+/g, ' ').trim().toLowerCase()

const artifactNames: Record<string, string> = {
  'document clustering by dominant topic': '文档主导主题聚类',
  'document clusters with outlier detection': '文档聚类与离群检测',
  'document volume over time': '文档数量随时间变化',
  'high frequency word evolution': '高频词年度演变',
  'intertopic distance map': '主题间距离图',
  'representative topic evolution': '代表性主题演变',
  'top salient terms': '高显著度词',
  'topic clustering heatmap with dendrogram': '主题聚类热图与树状图',
  'topic coherence': '主题一致性', 'topic correlation network': '主题相关网络',
  'topic distribution similarity evolution': '主题分布相似度演变',
  'topic exclusivity': '主题排他性', 'topic proportion distribution': '主题占比分布',
  'topic strength by year': '年度主题强度', 'topic word distribution': '主题词分布',
  'topic word cloud': '主题词云', 'word cloud': '词云', 'word distribution': '词分布',
  'topic evolution': '主题演变', evolution: '演变', 'word importance': '词语重要性',
  'doc topic umap': '文档主题降维投影', 'document projection': '文档投影数据',
  'platform dominant topics': '分组主导主题', 'platform topic deviation': '分组主题偏差',
  'platform topic stacked': '分组主题堆积图', 'stm gamma coefficients': '协变量系数',
  'covariate topic heatmap': '协变量与主题热图', 'source topic sankey': '来源与主题流向',
  'year topic sankey': '年份与主题流向', 'topic similarity': '主题相似度',
  'reconstruction and kl loss': '重构与散度损失曲线', 'topic network circular': '环形主题相关网络',
  'training loss': '训练损失曲线', 'training perplexity': '训练困惑度曲线',
  'dimension topic heatmap': '维度与主题热图', 'domain topic distribution over time': '领域主题年度分布',
  'topic word evolution': '主题词演变', 'word distribution by time': '词语时间分布',
  'topic wordcloud grid': '主题词云总览', 'topic table': '主题汇总表',
  'document counts by year': '年度文档数量', 'evaluation metrics': '评估指标',
  'intertopic distance': '主题间距离', 'topic beta similarity': '主题词相似度',
  'topic correlations': '主题相关性', 'topic network circular edges': '环形主题网络连边',
  'topic network edges': '主题网络连边', 'topic proportions': '主题占比',
  'topic weights by year': '年度主题权重', 'topic word weights': '主题词权重',
  'word counts by year': '年度词频', 'document topic': '文档主题分布',
  'topic keywords': '主题关键词', 'dtm word evolution': '动态主题词演变',
  'training curves': '训练曲线数据', 'group topic over time': '分组主题年度变化',
  'stm group topic means': '分组主题均值', 'topic weights by group': '分组主题权重',
  'pyldavis interactive': '交互式主题探索', 'pyldavis interactive scope': '交互图适用范围',
  config: '训练配置', info: '模型信息', model: '模型文件', 'etm model': '嵌入主题模型',
  theta: '文档主题矩阵', beta: '主题词矩阵', 'beta over time': '时序主题词矩阵',
  gamma: '协变量系数矩阵', sigma: '协方差矩阵', covariates: '协变量矩阵',
  'covariate effects': '协变量效应', 'covariate info': '协变量信息',
  'topic words': '主题词', vocab: '词表', metrics: '评估指标',
  'result manifest': '结果清单', 'publication manifest': '发布清单',
  'chart status': '图表生成状态', 'additional chart status': '补充图表生成状态',
  'temporal scope': '时间分析范围', readme: '使用说明', index: '完整研究报告',
  worker: '计算日志', visualization: '可视化日志', 'training history': '训练历史',
  'bow matrix': '词袋矩阵', 'document topics': '文档主题矩阵',
  'vocab embeddings': '词表嵌入', embeddings: '文本嵌入', 'topic embeddings': '主题嵌入',
  'word2vec embeddings': '词向量', metadata: '元数据', 'source rows': '原始行索引', texts: '文本数据',
}

export function artifactLabel(name: string, kind = 'artifact'): string {
  if (/^adjustments\/[a-f0-9]{12}\//u.test(name)) return `${artifactLabel(name.split('/').pop() ?? name, kind)}（调整版）`
  const base = (name.split(/[\\/]/).pop() ?? name).replace(/\.(png|jpe?g|webp|gif|svg|pdf|csv|tsv|json|jsonl|html|npy|npz|pkl|pt|joblib|md|txt|log)$/i, '')
  // User-authored Chinese titles remain intact.
  if (/[\u3400-\u9fff]/u.test(base)) return base
  const key = keyOf(base)
  if (artifactNames[key]) return artifactNames[key]
  const dated = /^(.*) (\d{8}) (\d{6})$/.exec(key)
  if (dated && artifactNames[dated[1]]) return `${artifactNames[dated[1]]} · ${dated[2]} ${dated[3]}`
  const adjusted = /^(.*) adjusted (\d+)$/.exec(key)
  if (adjusted) return `${artifactLabel(adjusted[1], kind)}（调整版 ${adjusted[2]}）`
  const topic = /^topic (\d+)(?: (.*))?$/.exec(key)
  if (topic) return `主题 ${topic[1]}${topic[2] ? ` · ${artifactLabel(topic[2], kind)}` : ''}`
  const numbered = /^(.*?)(?: k)? (\d+)$/.exec(key) ?? /^(.*?) k(\d+)$/.exec(key)
  if (numbered && artifactNames[numbered[1]]) return `${artifactNames[numbered[1]]} · ${numbered[2]}`
  const mode = /^(.*) (zero shot|unsupervised|supervised)$/.exec(key)
  if (mode && artifactNames[mode[1]]) return `${artifactNames[mode[1]]}（${modeLabel(mode[2])}）`
  // Unknown extensions stay accessible, without presenting machine names as titles.
  const category = kind === 'figure' ? '其他图表' : kind === 'table' ? '其他数据表' : kind === 'matrix' ? '其他矩阵' : '其他结果文件'
  let id = 0
  for (const character of name) id = (Math.imul(id, 31) + character.charCodeAt(0)) >>> 0
  return `${category} · ${String(id % 1000000).padStart(6, '0')}`
}

const statuses: Record<string, string> = {
  pending: '等待中', queued: '排队中', waiting: '等待中', idle: '尚未开始', created: '已创建', creating: '正在创建',
  running: '运行中', training: '训练中', completed: '已完成', succeeded: '已完成', success: '已完成', done: '已完成',
  failed: '失败', error: '出错', cancelling: '正在取消', cancelled: '已取消', canceled: '已取消', stopped: '已停止', stopping: '正在停止',
  ready: '已就绪', incomplete: '未完成', skipped: '已跳过', unknown: '状态待确认',
  'not requested': '尚未生成', 'pending upload': '等待上传', 'submitting dlc': '正在提交训练',
  'awaiting approval': '等待确认', 'awaiting confirmation': '等待确认', 'needs input': '等待补充信息',
  'awaiting user': '等待用户操作', 'in progress': '进行中', paused: '已暂停',
  'waiting human': '等待人工确认', 'needs approval': '等待确认', 'not started': '尚未开始',
  interrupted: '已中断', unavailable: '暂不可用', complete: '已完成',
}
export const statusLabel = (value?: string) => statuses[keyOf(value ?? '')] ?? (value && /[\u3400-\u9fff]/u.test(value) ? value : '状态待确认')
const phases: Record<string, string> = {
  initializing: '初始化', initialization: '初始化', loading: '加载数据', 'loading data': '加载数据',
  preprocessing: '数据预处理', preprocess: '数据预处理', embedding: '生成嵌入', vectorizing: '生成向量',
  training: '模型训练', evaluation: '模型评估', evaluating: '模型评估', visualization: '生成图表',
  visualizing: '生成图表', reporting: '整理报告', saving: '保存结果', exporting: '导出结果',
  uploading: '上传数据', downloading: '下载文件', mining: '主题挖掘', inference: '模型推理',
  scheduling: '调度任务', preparing: '检查与规范化数据', 'preparing data': '数据预处理', 'preparing environment': '准备运行环境',
  'loading model': '加载模型', 'building vocabulary': '构建词表', 'generating embeddings': '生成嵌入',
  'generating report': '生成报告', 'saving results': '保存结果', 'collecting results': '收集结果',
}
export const phaseLabel = (value?: string) => value ? phases[keyOf(value)] ?? statusLabel(value) : '尚未开始'
export const modeLabel = (value: string) => ({ 'zero shot': '零样本分析', unsupervised: '无监督分析', supervised: '有监督分析', automatic: '自动模式', manual: '手动模式' }[keyOf(value)] ?? '主题分析')

const fields: Record<string, string> = {
  topic: '主题', 'topic id': '主题编号', 'topic name': '主题名称', name: '名称', id: '编号',
  'topic count': '主题数', 'num topics': '主题数', 'n topics': '主题数', 'n components': '主题数',
  'document id': '文档编号', 'doc id': '文档编号', 'document index': '文档序号',
  'dominant topic': '主导主题', 'document count': '文档数量', count: '数量', year: '年份', timestamp: '时间',
  word: '词语', term: '词语', keywords: '关键词', 'top words': '高权重词', weight: '权重', weights: '权重',
  proportion: '占比', probability: '概率', frequency: '词频', importance: '重要性',
  metric: '指标', value: '数值', score: '得分', coherence: '一致性', 'c v': '一致性得分', 'c npmi': '归一化互信息',
  npmi: '归一化互信息', 'u mass': '语义一致性', diversity: '多样性', 'topic diversity': '主题多样性',
  td: '主题多样性', irbo: '主题差异度', 'i rbo': '主题差异度', umass: '语义连贯性', ppl: '困惑度',
  perplexity: '困惑度', exclusivity: '排他性', 'topic coherence': '主题一致性',
  loss: '损失', 'train loss': '训练损失', 'training loss': '训练损失', 'val loss': '验证损失',
  'reconstruction loss': '重构损失', 'kl loss': '散度损失', 'kl divergence': '散度',
  epoch: '轮次', epochs: '训练轮数', batch: '批次', 'batch size': '批大小',
  'learning rate': '学习率', lr: '学习率', 'hidden dim': '隐藏维度', 'hidden size': '隐藏层大小',
  'embedding dim': '嵌入维度', 'embedding dimension': '嵌入维度', 'embedding model': '嵌入模型',
  'vocab size': '词表大小', 'num layers': '层数', 'n layers': '层数', 'num neurons': '神经元数',
  'random state': '随机种子', seed: '随机种子', patience: '早停耐心值', dropout: '随机失活率',
  alpha: '文档主题先验', beta: '主题词先验', model: '模型', 'model id': '模型', 'model type': '模型类型',
  mode: '分析模式', algorithm: '算法', iterations: '迭代次数', 'max iter': '最大迭代次数',
  source: '来源', target: '目标', group: '分组', platform: '平台', text: '文本',
  'n documents': '文档数量', 'matrix row': '矩阵行号', 'mean topic weight': '平均主题权重',
  'mean weight': '平均权重', 'weight mass': '权重总量', strength: '主题强度', keyword: '关键词', time: '时间',
  'source topic': '来源主题', 'target topic': '目标主题', 'pearson r': '皮尔逊相关系数',
  pc1: '第一主成分', pc2: '第二主成分', 'recon loss': '重构损失',
  'train ppl': '训练困惑度', 'val ppl': '验证困惑度',
  correlation: '相关系数', similarity: '相似度', distance: '距离', rank: '排序',
  'x': '横坐标', 'y': '纵坐标', 'z': '纵深坐标',
}
export function fieldLabel(value: string): string {
  const key = keyOf(value)
  if (fields[key]) return fields[key]
  const topic = /^(?:topic ?|t)(\d+)$/.exec(key)
  if (topic) return `主题 ${topic[1]}`
  if (/^\d+$/.test(value) || /[\u3400-\u9fff]/u.test(value)) return value
  // Arbitrary dataset columns may be user-authored; don't invent a translation.
  return value
}

/** Provider failures are system messages, never generated research conclusions. */
export function advisoryErrorMessage(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error)
  if (/\b503\b|service.*busy|service_unavailable/iu.test(message)) return '当前模型服务繁忙（503），请稍后重试。训练与结果不受影响。'
  if (/\b429\b|rate.limit/iu.test(message)) return '当前模型服务请求过于频繁，请稍后重试。'
  if (/abort|timeout|超时/iu.test(message)) return '对话助手响应超时，请稍后重试。训练与结果不受影响。'
  return '暂时无法完成结果咨询，请稍后重试。训练与结果不受影响。'
}

/** Use only on system-generated status text, never user messages or research prose. */
export function systemText(value: string | undefined, fallback = '任务状态已更新'): string {
  if (!value) return ''
  const key = keyOf(value)
  if (statuses[key]) return statuses[key]
  if (phases[key]) return phases[key]
  let translated = value.replace(/\bEpoch\s+(\d+)\s*\/\s*(\d+)/gi, '训练轮次 $1 / $2')
    .replace(/\b(completed|succeeded|failed|cancelled|canceled|queued|pending|running|training|loading|preprocessing|evaluating|saving)\b/gi, word => phases[word.toLowerCase()] ?? statuses[word.toLowerCase()] ?? word)
    .replace(/\b(loss|perplexity|coherence|diversity|epoch|epochs|batch|learning_rate)\b/gi, word => fieldLabel(word))
  return /[\u3400-\u9fff]/u.test(translated) ? translated : fallback
}
