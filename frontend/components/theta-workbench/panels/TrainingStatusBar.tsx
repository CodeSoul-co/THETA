import { useId, useState } from 'react'
import { Activity, ArrowUpRight, ChartNoAxesCombined, ChevronDown } from 'lucide-react'
import type { WebResultCatalog, WebRunStatus } from '../api/client.ts'
import { usePreferences } from '../preferences.tsx'
import { trainingSummary } from './training-summary.ts'
import css from './TrainingStatusCard.module.css'

const COLLAPSED_KEY = 'theta.workspace.training-banner-collapsed.v1'

export const TrainingStatusBar = ({ status, catalog, onOpen }: {
  status?: WebRunStatus
  catalog?: WebResultCatalog
  onOpen: (tab: 'progress' | 'results') => void
}): React.ReactElement | null => {
  const { locale } = usePreferences()
  const zh = locale === 'zh-CN'
  const detailsId = useId()
  const [collapsed, setCollapsed] = useState(() => {
    try { return localStorage.getItem(COLLAPSED_KEY) === '1' } catch { return false }
  })
  const summary = trainingSummary(status, catalog)
  if (!summary.state) return null
  const title = ({
    running: zh ? '模型正在训练' : 'Training in progress',
    queued: zh ? '训练正在排队' : 'Training queued',
    completed: zh ? '训练完成' : 'Training complete',
    failed: zh ? '训练需要关注' : 'Training needs attention',
    cancelled: zh ? '训练已结束 · 含取消任务' : 'Training ended · includes cancellations',
  } as Record<string, string>)[summary.state] ?? (zh ? '训练进度' : 'Training progress')
  const subtitle = summary.total > 1
    ? (zh ? `${summary.completed} / ${summary.total} 个训练任务已完成${summary.active ? ` · ${summary.active} 个进行中` : ''}` : `${summary.completed} / ${summary.total} jobs complete`)
    : summary.current?.modelId.toUpperCase() ?? (zh ? '当前研究任务' : 'Current research')
  const toggle = () => setCollapsed(current => {
    try { localStorage.setItem(COLLAPSED_KEY, current ? '0' : '1') } catch { /* Optional preference. */ }
    return !current
  })

  return <section className={css.card} data-state={summary.state} data-collapsed={collapsed || undefined} aria-label={zh ? '当前对话训练状态' : 'Conversation training status'}>
    <div className={css.header}>
      <span className={css.icon}>{summary.ready ? <ChartNoAxesCombined size={17} /> : <Activity size={17} />}</span>
      <div className={css.copy}><strong>{title}</strong><small>{subtitle}</small></div>
      <button type="button" className={css.action} onClick={() => onOpen(summary.ready ? 'results' : 'progress')}>
        {summary.ready ? (zh ? '查看结果' : 'View results') : (zh ? '查看进度' : 'View progress')}<ArrowUpRight size={14} />
      </button>
      <button type="button" className={css.toggle} onClick={toggle} aria-expanded={!collapsed} aria-controls={detailsId} aria-label={collapsed ? (zh ? '展开训练状态' : 'Expand training status') : (zh ? '收起训练状态' : 'Collapse training status')}>
        {collapsed ? (zh ? '展开' : 'Expand') : (zh ? '收起' : 'Collapse')}<ChevronDown size={14} />
      </button>
    </div>
    <div id={detailsId} className={css.details} hidden={collapsed}>
      <div className={css.stats}>
        <span><b>{summary.total || 1}</b>{zh ? '训练任务' : 'jobs'}</span>
        <span><b>{summary.figures}</b>{zh ? '图表' : 'figures'}</span>
        <span><b>{summary.tables}</b>{zh ? '数据表' : 'tables'}</span>
        {summary.failed > 0 && <span><b>{summary.failed}</b>{zh ? '失败' : 'failed'}</span>}
        {summary.cancelled > 0 && <span><b>{summary.cancelled}</b>{zh ? '已取消' : 'cancelled'}</span>}
      </div>
      {summary.state === 'running' || summary.state === 'queued'
        ? <div className={css.progress}><div className={css.track} role="progressbar" aria-label={zh ? '整体训练进度' : 'Overall training progress'} aria-valuemin={0} aria-valuemax={100} aria-valuenow={summary.percent}><i style={{ width: `${summary.percent}%` }} /></div><small>{summary.percent}%</small></div>
        : <p className={css.caption}>{summary.ready ? (zh ? '结果已保存，可按模型查看图表与原始数据。' : 'Results saved. Explore charts and source data by model.') : (zh ? '展开右侧进度，查看任务详情。' : 'Open progress to view the job details.')}</p>}
    </div>
  </section>
}
