export interface TrainingEvent {
  source?: 'local' | 'cloud'; scope?: 'documents' | 'vocabulary';
  completedBatches?: number; totalBatches?: number; chunks?: number; chunkTotal?: number;
  id: string | number
  at: number | null
  kind: 'epoch' | 'iteration' | 'batch' | 'embedding' | 'early_stop' | 'command' | 'visualizing'
  current?: number
  total?: number | null
  stage?: string | null
  batch?: { current: number; total: number }
  activity?: string
  metrics?: Record<string, number>
}
export interface TrainingWorkerState {
  id: string; status: string; phase: string; percent: number
  phaseHistory?: { phase: string; at: number }[]
  telemetry?: {
    elapsedSeconds: number | null
    lastLogAgeSeconds: number | null
    health: string
    events?: TrainingEvent[]
    detail?: TrainingEvent | null
  }
}
const metricLabels: Record<string, string> = { loss: '损失', train_loss: '训练损失', val_loss: '验证损失', recon_loss: '重构损失', kl_loss: 'KL 损失', perplexity: '困惑度', ce_loss: '分类损失', contrastive_loss: '对比损失' }
const activityLabels: Record<string, string> = { 'Generating embeddings': '生成嵌入', 'Embedding vocabulary': '词表嵌入', 'Cleaning text': '清理正文', Tokenizing: '分词', BOW: '生成词袋', Batches: '生成嵌入' }
export function trainingEventText(event: TrainingEvent): string {
  const parts: string[] = []
  if (event.kind === 'embedding') {
    parts.push(`${event.source === 'cloud' ? '云端' : '本地'}嵌入`)
    parts.push(`已完成${event.scope === 'vocabulary' ? '词条' : '文本'} ${event.current}/${event.total}`)
    if (event.total) parts.push(`${Math.floor((event.current ?? 0) / event.total * 100)}%`)
    if (event.chunks != null) parts.push(`已编码文本块 ${event.chunks}${event.chunkTotal != null ? `/${event.chunkTotal}` : ''}`)
    if (event.completedBatches != null) parts.push(`已完成批次 ${event.completedBatches}${event.totalBatches != null ? `/${event.totalBatches}` : ''}`)
    return parts.join(' · ')
  }
  if (event.kind === 'command') return '启动计算步骤'
  if (event.kind === 'visualizing') return '生成可视化'
  if (event.kind === 'early_stop') return `第 ${event.current} 轮触发提前停止`
  if (event.stage) parts.push(event.stage === 'stage1' ? '第一阶段' : '第二阶段')
  if (event.kind === 'iteration') parts.push(`迭代 ${event.current}/${event.total}`)
  else if (event.current !== undefined) parts.push(`训练轮次 ${event.current}${event.total ? `/${event.total}` : ''}${event.kind === 'epoch' ? ' · 本轮完成' : ''}`)
  if (event.activity && activityLabels[event.activity]) parts.push(activityLabels[event.activity])
  if (event.batch) parts.push(`批次 ${event.batch.current}/${event.batch.total}`)
  for (const [key, value] of Object.entries(event.metrics ?? {})) {
    if (metricLabels[key] && Number.isFinite(value)) parts.push(`${metricLabels[key]} ${value.toLocaleString('zh-CN', { maximumSignificantDigits: 6 })}`)
  }
  return parts.join(' · ')
}
export function trainingElapsed(seconds: number): string {
  const value = Math.max(0, Math.floor(seconds))
  return value < 60 ? `${value} 秒` : value < 3600 ? `${Math.floor(value / 60)} 分 ${value % 60} 秒` : `${Math.floor(value / 3600)} 小时 ${Math.floor(value % 3600 / 60)} 分`
}
