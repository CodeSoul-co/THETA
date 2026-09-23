import { useId, useRef, useState } from 'react'
import { Cloud, Laptop, Loader2 } from 'lucide-react'
import { postAction, type WebAgentInteraction } from '../api/client.ts'
import css from './ActionApprovalCard.module.css'

/** Choosing the embedding source only prepares a separate, versioned training approval. */
export function EmbeddingChoiceCard({ runId, interaction, onApproved, decisionError }: {
  runId: string; interaction: WebAgentInteraction; onApproved: () => void; decisionError?: string
}): React.ReactElement {
  const card = interaction.card!
  const titleId = useId()
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(decisionError ?? '')
  const inFlight = useRef(false)
  const disabled = busy || interaction.status === 'running'
  const choose = async (action: 'embedding_local' | 'embedding_cloud' | 'reject') => {
    if (inFlight.current || disabled) return
    inFlight.current = true
    setBusy(true)
    setError('')
    try {
      await postAction(runId, { action, checkpointId: card.actionRef, expectedContentHash: card.contentHash })
      onApproved()
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause))
      setBusy(false)
      inFlight.current = false
      onApproved()
    }
  }
  return <section className={css.card} aria-labelledby={titleId} aria-busy={disabled}>
    <header className={css.header}>
      <span className={css.requestIcon} aria-hidden="true">{busy ? <Loader2 size={16} className={css.spinner} /> : <Cloud size={16} />}</span>
      <div className={css.heading}><strong id={titleId}>选择嵌入方式</strong><span>选择后，再确认训练方案</span></div>
    </header>
    <div className={css.body}>
      <p className={css.scopeNotes}>将文本转换成模型可用的语义向量。请为本次 THETA 零样本分析选择一种方式。</p>
      <div className={css.embeddingOptions}>
        <button type="button" disabled={disabled} onClick={() => void choose('embedding_local')}>
          <Laptop size={18} aria-hidden="true" /><strong>本地嵌入</strong>
          <span>使用本机 Qwen 模型，不向云端发送嵌入文本。下一步检查本地资源。</span>
          <small>选择本地 →</small>
        </button>
        <button type="button" disabled={disabled || !card.cloudAvailable} onClick={() => void choose('embedding_cloud')}>
          <Cloud size={18} aria-hidden="true" /><strong>云端嵌入</strong>
          <span>使用已配置的云服务，发送正文与派生词表，并产生供应商费用。</span>
          <small>{card.cloudAvailable ? '选择云端 →' : '云端配置未就绪'}</small>
        </button>
      </div>
      <p className={css.scopeNotes}>{card.description.split('\n').filter(line => /^(云端：|云端地址：|本次外部请求上限：)/u.test(line)).join('\n')}</p>
      <p className={css.scopeNotes}>现在不会开始训练，也不会发送嵌入请求。下一张训练卡会列出具体参数和执行范围，仍需你确认。</p>
      <details className={css.details}><summary>查看完整说明</summary><p>{card.description}</p></details>
      {error && <p className={css.error} role="alert">{error}</p>}
    </div>
    <footer className={css.footer}><div className={css.actions}><button type="button" className={css.secondary} disabled={disabled} onClick={() => void choose('reject')}>暂不训练</button></div></footer>
  </section>
}
