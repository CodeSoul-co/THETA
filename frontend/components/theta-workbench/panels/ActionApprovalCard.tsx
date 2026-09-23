import { useId, useRef, useState } from 'react'
import { fieldLabel } from '@/lib/presentation'
import { Check, ChevronRight, ClipboardCheck, Loader2, Pencil } from 'lucide-react'
import { postAction, type WebAgentInteraction } from '../api/client.ts'
import css from './ActionApprovalCard.module.css'

type Props = { runId: string; interaction: WebAgentInteraction; onApproved: () => void; decisionError?: string }

/** One decision applies only to the displayed version of one operation. */
export function ActionApprovalCard({ runId, interaction, onApproved, decisionError }: Props): React.ReactElement {
  const card = interaction.card!
  const titleId = useId()
  const feedbackId = useId()
  const draftKey = `theta.card-feedback.${runId}.${card.actionRef}.${card.contentHash}`
  const [feedback, setFeedback] = useState(() => {
    try { return sessionStorage.getItem(draftKey) ?? '' } catch { return '' }
  })
  const [editing, setEditing] = useState(() => !!feedback)
  const [busy, setBusy] = useState<'approve' | 'reject' | 'revise'>()
  const [error, setError] = useState(decisionError ?? '')
  const inFlight = useRef(false)
  const editButtonRef = useRef<HTMLButtonElement>(null)
  const preparing = interaction.status === 'running'
  const unavailable = busy != null || preparing
  const lines = card.description.split('\n').map(line => line.trim()).filter(Boolean)
  const lineValue = (...labels: string[]): string | undefined => {
    const expression = new RegExp(`^(?:${labels.join('|')})[:：]\\s*(.+)$`, 'u')
    return lines.map(line => expression.exec(line)?.[1]).find(Boolean)
  }
  const taskLine = lineValue('研究目标', '问题', '任务', '目标')
  const dataLine = lineValue('数据', '数据文件')
  const methodLine = lineValue('方法', '分析方法')
  const modelLine = lineValue('模型')
  const textColumn = lineValue('文本列', '正文列')
  const language = lineValue('语言', '数据语言')
  const scopeNotes = lines.filter(line => /^(输出|交付)/u.test(line))
  const overviewLines = lines.filter((line, index) => !/^(?:任务|研究目标|问题|目标)[:：]/u.test(line) && !(index === 0 && /[？?]$/u.test(line)))
  const parameters = lines.find(line => /^参数[:：]/u.test(line))
  let parameterSummary = parameters?.replace(/^参数[:：]\s*/u, '') ?? ''
  let embeddingSummary = ''
  if (parameters) {
    try {
      const params = JSON.parse(parameters.replace(/^参数[:：]\s*/u, '')) as Record<string, unknown>
      if (params.embedding_provider === 'local') embeddingSummary = `本地 · Qwen3-Embedding ${params.model_size ?? '0.6B'}，不发送云端嵌入请求`
      if (params.embedding_provider === 'cloud') embeddingSummary = `云端 · ${params.embedding_cloud_provider ?? ''} · ${params.embedding_model ?? ''}（发送正文与派生词表，按供应商计费）`
      const preferred = [['num_topics', '主题数'], ['max_iter', '最大迭代'], ['epochs', '训练轮数'], ['learning_rate', '学习率'], ['batch_size', '批大小']]
        .flatMap(([key, label]) => params[key] != null ? [`${label} ${String(params[key])}`] : [])
      parameterSummary = preferred.length ? preferred.join('，') : Object.entries(params).slice(0, 5).map(([key, value]) => `${fieldLabel(key)} ${String(value)}`).join('，')
    } catch { /* The complete, authoritative description remains available below. */ }
  }
  const method = [methodLine, modelLine].filter(Boolean).join(' · ') || '由 Agent 根据当前数据与研究目标确定'
  const planFacts = [
    { label: '研究目标', value: taskLine || '执行当前对话中确认的研究目标' },
    { label: '数据文件', value: dataLine || '使用当前对话已附加的数据集' },
    { label: '分析方法', value: method },
    { label: '文本预处理', value: [textColumn && `正文列 ${textColumn}`, language].filter(Boolean).join('，') || '按数据理解结果执行清洗与分词' },
    { label: '模型参数', value: parameterSummary || '使用当前方案中的已校验参数' },
    ...(embeddingSummary ? [{ label: '嵌入方式', value: embeddingSummary }] : []),
    ...(lineValue('外部 embedding') ? [
      { label: '云端地址', value: lineValue('外部 embedding')! },
      { label: '发送与额度', value: lines.filter(line => /^(将发送：|最多 \d+ 次 HTTP)/u.test(line)).join(' ') },
    ] : []),
    { label: '预期输出', value: scopeNotes.map(line => line.replace(/^[^:：]+[:：]\s*/u, '')).join('；') || '主题结构、关键词、代表文本与可视化结果' },
  ]
  const isResultRead = /整理结果|读取结果/u.test(card.title)
  const isDeepRead = /深入|解读/u.test(card.title) && !isResultRead
  const isCancel = /取消|停止/u.test(card.title) || /^确认停止当前训练/u.test(card.description)
  const isTraining = !isCancel && /启动训练/u.test(card.title)
  const resultFacts = [
    { label: '结果任务', value: taskLine || '当前已完成的训练任务' },
    { label: '当前状态', value: '训练已完成，结果已保存在受管工作区' },
    { label: '读取范围', value: '模型产出、指标、主题词与原始数据标签' },
    { label: '可视化导出', value: '全部可用图表、表格和原始矩阵' },
    { label: '不会执行', value: '不会重新训练，也不会调用外部 embedding' },
    { label: '后续操作', value: '结合原始文本的深入解读仍需单独确认' },
  ]
  const deepFacts = [
    { label: '解读对象', value: taskLine || '当前选择的已完成结果' },
    { label: '证据范围', value: '已选择结果、图表源数据与对应原始文本行' },
    { label: '解读目标', value: '围绕当前研究问题解释主题、差异与潜在机制' },
    { label: '输出内容', value: '可追溯结论、证据引用与局限说明' },
    { label: '不会执行', value: '不会重新训练，也不会修改已保存结果' },
    { label: '授权边界', value: '仅限本次选择的结果和当前解读请求' },
  ]
  const facts = isCancel ? [
    { label: '操作范围', value: '停止当前对话关联的训练任务' },
    { label: '保留内容', value: '保留已生成的状态记录与产物，不启动新的训练' },
  ] : isResultRead ? resultFacts : isDeepRead ? deepFacts : isTraining ? planFacts : [
    { label: '本次操作', value: card.description },
  ]
  const edit = (value: string) => {
    setFeedback(value)
    try { sessionStorage.setItem(draftKey, value) } catch { /* Optional draft persistence. */ }
  }
  const submit = async (action: 'approve' | 'reject' | 'revise') => {
    if (inFlight.current || preparing) return
    if (action === 'revise' && !feedback.trim()) { setError('请填写要修改的内容。'); return }
    if (action === 'approve' && feedback.trim()) return
    inFlight.current = true
    setBusy(action)
    setError('')
    try {
      await postAction(runId, { action: action === 'approve' ? 'approveCheckpoint' : action,
        checkpointId: card.actionRef, expectedContentHash: card.contentHash,
        reason: action === 'reject' ? `拒绝本次操作，暂不执行。${feedback.trim()}` : feedback.trim() })
      try { sessionStorage.removeItem(draftKey) } catch { /* Optional draft persistence. */ }
      // Remain locked until the parent receives the authoritative decision state.
      onApproved()
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause))
      inFlight.current = false
      setBusy(undefined)
      onApproved()
    }
  }
  const stateText = busy === 'approve' ? '正在提交确认' : busy === 'revise' ? '正在提交修改' : preparing ? 'Agent 正在处理' : '等待确认'
  const visualTitle = isCancel ? '取消任务确认' : isTraining ? '训练方案确认' : isResultRead ? '结果读取确认' : isDeepRead ? '结果解读确认' : '操作确认'
  const approveLabel = isCancel ? '确认停止' : isResultRead ? '允许读取' : isDeepRead ? '开始解读' : isTraining ? '接受方案' : '确认执行'
  const reviseLabel = isResultRead || isDeepRead ? '调整范围' : '继续修改'
  return (
    <section className={css.card} aria-labelledby={titleId} aria-busy={unavailable}>
      <header className={css.header}>
        <span className={css.requestIcon} aria-hidden="true">{unavailable ? <Loader2 className={css.spinner} size={16} /> : <ClipboardCheck size={16} />}</span>
        <div className={css.heading}><strong id={titleId}>{visualTitle}</strong><span role="status">（{stateText}）</span></div>
      </header>
      <div className={css.body}>
        <dl className={css.facts}>{facts.map(({ label, value }) => <div key={label}><dt>{label}</dt><dd title={value}>{value}</dd></div>)}</dl>
        <details className={css.details}>
          <summary><ChevronRight size={14} aria-hidden="true" />查看完整方案与执行范围</summary>
          <p>{overviewLines.join('\n') || card.description}</p>
        </details>
        {editing && <label className={css.feedback} htmlFor={feedbackId}>
          <span>希望怎样修改？</span>
          <textarea id={feedbackId} aria-label="修改意见" rows={2} maxLength={4000} value={feedback} disabled={unavailable} autoFocus
            placeholder="例如：把迭代次数改为 10，其他设置保持不变。"
            onChange={event => edit(event.target.value)} />
          <small>提交后由 Agent 更新方案，新方案仍需你确认。</small>
        </label>}
        {error && <p className={css.error} role="alert">{error} 意见已保留，请核对当前卡片后重试。</p>}
      </div>
      <footer className={css.footer}>
        <div className={css.actions}>
          {editing ? <button type="button" className={css.secondary} disabled={unavailable} onClick={() => { edit(''); setEditing(false); setError(''); window.requestAnimationFrame(() => editButtonRef.current?.focus()) }}>取消修改</button>
            : <button ref={editButtonRef} type="button" className={css.secondary} disabled={unavailable} onClick={() => setEditing(true)}>{reviseLabel}</button>}
          <button type="button" className={css.primary} disabled={unavailable || (editing ? !feedback.trim() : !!feedback.trim())} onClick={() => void submit(editing ? 'revise' : 'approve')}>
            {busy && busy !== 'reject' ? <Loader2 size={14} className={css.spinner} /> : editing ? <Pencil size={14} /> : <Check size={15} />}
            {busy && busy !== 'reject' ? '提交中…' : editing ? '提交修改' : approveLabel}
          </button>
        </div>
      </footer>
    </section>
  )
}
