import { useEffect, useMemo, useState } from 'react'
import { SlidersHorizontal } from 'lucide-react'
import { AnalysisConfigPanel } from '@/components/project/analysis-config-panel'
import { configFromPlans, plansFromConfig } from '@/lib/analysis-config'
import { getTrainingEditor, trainingEditorService, submitTrainingEditor, type TrainingEditorData, type WebAgentInteraction } from '../api/client.ts'
import css from './ActionApprovalCard.module.css'

export function TrainingConfigurationCard({ runId, interaction, onApproved }: {
  runId: string; interaction: WebAgentInteraction; onApproved: () => void
}) {
  const card = interaction.card!
  const key = `theta.training-editor.${runId}.${card.actionRef}.${card.contentHash}`
  const [open, setOpen] = useState(() => { try { return sessionStorage.getItem(key) !== 'closed' } catch { return true } })
  const [data, setData] = useState<TrainingEditorData>()
  const [error, setError] = useState('')
  const [reload, setReload] = useState(0)
  const service = useMemo(() => trainingEditorService(runId), [runId])
  useEffect(() => {
    let cancelled = false
    getTrainingEditor(runId).then(value => { if (!cancelled) { setData(value); setError('') } })
      .catch(cause => { if (!cancelled) setError(cause instanceof Error ? cause.message : String(cause)) })
    return () => { cancelled = true }
  }, [runId, card.actionRef, card.contentHash, reload])
  const initialConfig = useMemo(() => data ? { ...configFromPlans(data.plans), stopwords: data.stopwords } : undefined, [data])
  const changeOpen = (value: boolean) => {
    setOpen(value)
    try { sessionStorage.setItem(key, value ? 'open' : 'closed') } catch { /* Optional view preference; server approval is authoritative. */ }
  }
  return <section className={css.card}>
    <header className={css.header}><span className={css.requestIcon}><SlidersHorizontal size={16} /></span><div className={css.heading}><strong>训练配置</strong><span>等待你确认</span></div></header>
    <div className={css.body}>
      <p className={css.overview}>{data ? `${data.datasetName} · ${data.plans.map(plan => plan.modelId.toUpperCase()).join('、')}` : '正在读取 Agent 预填方案…'}</p>
      <p className={css.scopeNotes}>模型与参数已预填，可增减模型、手动调整。确认后收起配置卡，再按最终配置开始执行。</p>
      {error && <p role="alert" className={css.error}>{error}</p>}
    </div>
    <footer className={css.footer}><div className={css.actions}><button type="button" className={css.primary} disabled={interaction.status === 'running'} onClick={() => { changeOpen(true); if (!data) setReload(value => value + 1) }}>{error ? '重新加载配置' : '查看与调整配置'}</button></div></footer>
    {data && initialConfig && <AnalysisConfigPanel key={key} open={open && interaction.status !== 'running'} onOpenChange={changeOpen} projectKey={key}
      datasetName={data.datasetName} datasetSizeBytes={data.datasetSizeBytes} initialConfig={initialConfig} columns={data.columns} service={service}
      description="Agent 已预填建议模型与各自参数；你可以手动修改，最终确认后按顺序训练。"
      confirmLabel="确认并开始训练" onConfirm={async config => {
        await submitTrainingEditor(runId, { checkpointId: card.actionRef, expectedContentHash: card.contentHash!, plans: plansFromConfig(config, data.plans),
          cloudConfirmed: config.cloudConfirmed, cloudSelection: config.cloudSelection, stopwordsId: config.stopwords?.id })
        changeOpen(false); onApproved()
      }} />}
  </section>
}
