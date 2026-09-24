import { DATASET_ACCEPT, datasetFilesError, documentCollectionError, isDocumentCollection } from '@/lib/dataset-files'
import { useFileDrop } from '@/lib/use-file-drop'
import { useEffect, useMemo, useRef, useState } from 'react'
import { artifactLabel, fieldLabel, modeLabel, systemText } from "@/lib/presentation"
import type { ReactNode } from 'react'
import { Activity, ArrowRight, BarChart3, ChevronDown, Database, FileText, PanelRightClose, UploadCloud } from 'lucide-react'
import type { WebAttachment, WebResultCatalog, WebRunEvent, WebRunResults, WebRunStatus } from '../api/client.ts'
import css from '../styles/app.module.css'
import { usePreferences } from '../preferences.tsx'
import { ResultsPreview } from './ResultsPreview.tsx'
import { DatasetFileIcon } from '../ui/DatasetFileIcon.tsx'
import { trainingSummary } from './training-summary.ts'

type DrawerSection = 'data' | 'progress' | 'results' | 'interpretation'

interface DetailPaneProps {
  runId?: string
  status?: WebRunStatus
  events: WebRunEvent[]
  results?: WebRunResults
  resultCatalog?: WebResultCatalog
  plan?: Record<string, unknown>
  attachments: WebAttachment[]
  initialSection: 'progress' | 'results'
  onCollapse: () => void
  onOpenResults: (destination: 'results' | 'interpretation') => void
  onAttach: (attachment: WebAttachment) => void
  onUploadDataset: (files: File[]) => Promise<void>
}

interface ParameterEntry { label: string; value: string }

const record = (value: unknown): Record<string, unknown> =>
  value !== null && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {}

const readableKey = fieldLabel

const primitive = (value: unknown): string | undefined => {
  if (typeof value === 'string' || typeof value === 'number') return String(value)
  if (typeof value === 'boolean') return value ? '是' : '否'
  if (Array.isArray(value) && value.length <= 8 && value.every((item) => ['string', 'number', 'boolean'].includes(typeof item))) return value.join(', ')
  return undefined
}

const MODEL_PARAMETER = /model|algorithm|topic|layer|neuron|hidden|epoch|batch|learning|rate|alpha|beta|seed|embedding|dimension|cluster|mode|covariate|iteration|dropout|window|min|max|k\b/iu
const PRIVATE_OR_PATH = /hash|path|ref|token|secret|key|dataset|corpus|file/iu

const parameterEntries = (value: unknown): ParameterEntry[] => {
  const entries: ParameterEntry[] = []
  const visit = (current: unknown, prefix = '', depth = 0): void => {
    if (entries.length >= 10 || depth > 3) return
    for (const [key, child] of Object.entries(record(current))) {
      const itemPath = prefix ? `${prefix}.${key}` : key
      const formatted = primitive(child)
      if (formatted != null && MODEL_PARAMETER.test(itemPath) && !PRIVATE_OR_PATH.test(itemPath)) entries.push({ label: readableKey(key), value: /mode/i.test(key) ? modeLabel(formatted) : formatted })
      else if (formatted == null) visit(child, itemPath, depth + 1)
    }
  }
  visit(value)
  return entries
}

const trainingEvent = (event: WebRunEvent): boolean => /training|train|训练/iu.test(`${event.type} ${systemText(event.title, "训练状态更新")} ${event.detail ?? ''}`)

export const DetailPane = ({
  runId, status, events, results, resultCatalog, plan, attachments, initialSection,
  onCollapse, onOpenResults, onAttach, onUploadDataset,
}: DetailPaneProps): React.ReactElement => {
  const { locale } = usePreferences()
  const zh = locale === 'zh-CN'
  const [openSections, setOpenSections] = useState<Set<DrawerSection>>(() => new Set(['data', initialSection]))
  const [uploading, setUploading] = useState(false)
  const [openArtifacts, setOpenArtifacts] = useState<Set<string>>(() => new Set())
  const [openResults, setOpenResults] = useState<Set<string>>(() => new Set())
  const [uploadError, setUploadError] = useState<string>()
  const uploadInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    setOpenSections((current) => new Set([...current, initialSection]))
  }, [initialSection])

  const toggle = (section: DrawerSection): void => setOpenSections((current) => {
    const next = new Set(current)
    if (next.has(section)) next.delete(section)
    else next.add(section)
    return next
  })

  const datasets = attachments.filter((attachment) => attachment.kind === 'dataset')
  const hasRichResults = (results?.visualizations.length ?? 0) > 0 || (results?.topics.length ?? 0) > 0 || Object.keys(results?.metrics ?? {}).length > 0
  const catalogResults = resultCatalog?.results ?? []
  // 默认展开第一个真正有图表的任务，避免用户看到一片"空"。
  useEffect(() => {
    const first = [...catalogResults].reverse().find((item) => (item.artifacts?.files?.length ?? 0) > 0)
    if (first) setOpenResults((current) => current.size ? current : new Set([first.jobId]))
  }, [catalogResults])

  const resultCount = catalogResults.filter(result => result.status === 'completed').length || (hasRichResults ? 1 : 0)
  const summary = trainingSummary(status, resultCatalog)
  const latestResult = summary.current
  const statusRecord = record(status)
  const receipt = record(statusRecord.trainingReceipt)
  const parameters = useMemo(() => parameterEntries({ plan, receipt }), [plan, receipt])
  const presentationProgress = status?.presentation?.progress
  const rawProgress = presentationProgress?.percent ??
    (presentationProgress ? (presentationProgress.current / Math.max(presentationProgress.total, 1)) * 100 : undefined) ??
    (typeof receipt.progress === 'number' ? receipt.progress : results?.progress ?? 0)
  const progress = summary.state ? summary.percent : Math.max(0, Math.min(100, rawProgress))
  const trainingState = summary.state
  const modelName = latestResult?.modelId?.toUpperCase() ?? String(record(plan).modelId ?? (zh ? '尚未选择' : 'Not selected'))
  const recentEvents = events.filter(trainingEvent).slice(-4).reverse()
  const statusLabel = ({ queued: zh ? '等待执行' : 'Queued', running: zh ? '训练中' : 'Training', completed: zh ? '已完成' : 'Completed', failed: zh ? '执行失败' : 'Failed', cancelled: zh ? '已取消' : 'Cancelled' } as Record<string, string>)[trainingState ?? ''] ?? (zh ? '尚未开始' : 'Not started')

  const chooseDataset = async (files: File[]): Promise<void> => {
    if (!files.length || uploading) return
    const problem = isDocumentCollection(files) ? documentCollectionError(files, locale) : datasetFilesError(files, locale)
    if (problem) { setUploadError(problem); return }
    setUploading(true)
    setUploadError(undefined)
    try {
      await onUploadDataset(files)
    } catch (cause) {
      setUploadError(cause instanceof Error ? cause.message : String(cause))
    } finally {
      setUploading(false)
      if (uploadInputRef.current) uploadInputRef.current.value = ''
    }
  }

  const { dragging, dropProps } = useFileDrop(files => { void chooseDataset(files) }, uploading, setUploadError)

  const header = (section: DrawerSection, icon: ReactNode, title: string, meta?: string | number): React.ReactElement => (
    <button type="button" className={css.drawerSectionHeader} aria-expanded={openSections.has(section)} onClick={() => toggle(section)}>
      <span className={css.drawerSectionIcon}>{icon}</span>
      <strong>{title}</strong>
      {meta !== undefined && <small>{meta}</small>}
      <ChevronDown className={css.drawerSectionChevron} size={15} />
    </button>
  )

  return (
    <aside className={css.detail} style={{ position: 'relative', width: '100%', minWidth: 0, flex: 1 }} aria-label={zh ? '研究助手' : 'Research assistant'}>
      <div className={css.researchDrawerHeader}>
        <div><strong>{zh ? '研究助手' : 'Research assistant'}</strong><span>{zh ? '你的数据、训练与结果' : 'Your data, training and results'}</span></div>
        <button type="button" onClick={onCollapse} aria-label={zh ? '收起研究助手' : 'Collapse research assistant'}><PanelRightClose size={17} /></button>
      </div>

      <div className={css.researchDrawerBody}>
        <section className={css.drawerSection} data-open={openSections.has('data') || undefined}>
          {header('data', <Database size={17} />, zh ? '研究数据' : 'Research data', datasets.length || undefined)}
          {openSections.has('data') && <div className={css.drawerSectionBody}>
            <input
              ref={uploadInputRef}
              className={css.visuallyHidden}
              type="file"
              accept={DATASET_ACCEPT}
              onChange={(event) => void chooseDataset(Array.from(event.target.files ?? []))}
            />
            {datasets.length > 0 ? datasets.map((dataset) => (
              <div className={css.drawerDatasetCard} key={dataset.id}>
                <DatasetFileIcon filename={dataset.label} size={19} />
                <div><strong title={dataset.label}>{dataset.label}</strong><small>{zh ? '已保存在当前项目；移除输入框引用不会删除数据' : 'Saved in this project; removing a reference does not delete the data'}</small></div>
                <span>{zh ? '已就绪' : 'Ready'}</span>
              </div>
            )) : <p className={css.inspectorEmpty}>{zh ? '还没有研究数据。你可以先上传，之后发送消息时 Agent 会自动带上它。' : 'No research data yet. Upload it now and the Agent will receive it with your next message.'}</p>}
            <button {...dropProps} type="button" className={`${css.drawerUploadButton} ${dragging ? css.contextCardDragging : ""}`} disabled={uploading} onClick={() => uploadInputRef.current?.click()}>
              <UploadCloud size={17} aria-hidden="true" />
              <span><strong>{uploading ? (zh ? '正在上传…' : 'Uploading…') : datasets.length ? (zh ? '更换数据集' : 'Replace dataset') : (zh ? '上传数据集' : 'Upload dataset')}</strong><small>{zh ? (dragging ? '松开即可上传' : '拖拽文件到这里，或点击选择文件') : (dragging ? 'Drop to upload' : 'Drag a file here, or click to choose')}</small></span>
            </button>
            {uploadError && <p className={css.drawerUploadError} role="alert">{uploadError}</p>}
          </div>}
        </section>

        <section className={css.drawerSection} data-open={openSections.has('progress') || undefined}>
          {header('progress', <Activity size={17} />, zh ? '训练进度' : 'Training progress', trainingState ? statusLabel : undefined)}
          {openSections.has('progress') && <div className={css.drawerSectionBody}>
            <div className={css.drawerTrainingSummary} data-state={trainingState ?? 'idle'}>
              <div><span>{summary.total > 1 ? (zh ? '训练任务' : 'Training jobs') : (zh ? '模型' : 'Model')}</span><strong>{summary.total > 1 ? `${summary.completed} / ${summary.total} ${zh ? '已完成' : 'complete'}` : modelName}</strong></div>
              <div><span>{zh ? '状态' : 'Status'}</span><strong>{statusLabel}</strong></div>
              <div className={css.drawerProgressTrack} aria-label={`${Math.round(progress)}%`}><i style={{ width: `${progress}%` }} /></div>
              <b>{Math.round(progress)}%</b>
            </div>
            {parameters.length > 0 && <div className={css.drawerParameterGrid}>{parameters.map((parameter, index) => <span key={`${parameter.label}-${index}`}><small>{parameter.label}</small><strong>{parameter.value}</strong></span>)}</div>}
            {recentEvents.length > 0 && <div className={css.drawerTimeline}>{recentEvents.map((event) => <div key={event.id}><span /><p><strong>{event.title}</strong>{event.detail && <small>{systemText(event.detail, "训练详情已更新")}</small>}</p><time>{new Date(event.timestamp).toLocaleTimeString(locale, { hour: '2-digit', minute: '2-digit' })}</time></div>)}</div>}
            {!trainingState && <p className={css.inspectorEmpty}>{zh ? '训练开始后，这里会持续同步阶段与进度。' : 'Training phases and progress will appear here.'}</p>}
          </div>}
        </section>

        <section className={css.drawerSection} data-open={openSections.has('results') || undefined}>
          {header('results', <BarChart3 size={17} />, zh ? '训练结果' : 'Training results', resultCount || undefined)}
          {openSections.has('results') && <div className={css.drawerSectionBody}>
            {catalogResults.length > 0 && <div className={css.drawerResultList}>{catalogResults.slice().reverse().map((result) => {
              const artifacts = result.artifacts as { files: Array<{ name: string; kind: string; url: string }>; archiveUrl?: string } | undefined
              const deliverable = (artifacts?.files ?? []).filter((file) => file.kind === 'figure' || file.kind === 'table')
              const open = openResults.has(result.jobId)
              const statusText = ({ completed: zh ? '已完成' : 'Completed', running: zh ? '训练中' : 'Training', queued: zh ? '等待中' : 'Queued', failed: zh ? '失败' : 'Failed', cancelled: zh ? '已取消' : 'Cancelled' } as Record<string, string>)[result.status]
              return <article key={result.jobId} data-open={open || undefined}>
                <button type="button" className={css.drawerResultHeader} aria-expanded={open} onClick={() => setOpenResults((current) => { const next = new Set(current); if (next.has(result.jobId)) next.delete(result.jobId); else next.add(result.jobId); return next })}>
                  <strong>{result.modelId.toUpperCase()}</strong>
                  <span>{statusText}</span>
                  <small>{deliverable.length ? `${deliverable.filter((file) => file.kind === 'figure').length} ${zh ? '图' : 'fig'} · ${deliverable.filter((file) => file.kind === 'table').length} ${zh ? '表' : 'tbl'}` : (result.artifacts ? `${result.artifacts.fileCount} ${zh ? '个文件' : 'files'}` : '')}</small>
                  <ChevronDown className={css.drawerResultChevron} size={14} />
                </button>
                <div className={css.drawerProgressTrack}><i style={{ width: `${Math.max(0, Math.min(100, result.percent ?? (result.status === 'completed' ? 100 : 0)))}%` }} /></div>
                {open && <div className={css.drawerResultFiles} data-scroll="true">
                  {deliverable.length ? deliverable.map((file) => {
                    const name = file.name.split('/').pop() ?? file.name
                    const family = file.name.replace(/\.(png|jpe?g|svg|pdf)$/iu, '')
                    const preview = artifacts?.files.find((item) => item.name === family + '.png') ?? artifacts?.files.find((item) => item.kind === 'figure' && item.name.replace(/\.(png|jpe?g|svg|pdf)$/iu, '') === family && item.name.endsWith('.png'))
                    const bundleUrl = runId ? `/api/v3/runs/${encodeURIComponent(runId)}/results/${encodeURIComponent(result.jobId)}/figure-bundle?path=${encodeURIComponent(name)}` : undefined
                    const expanded = openArtifacts.has(file.url)
                    return <div key={file.url} className={css.drawerResultFile} data-expanded={expanded || undefined}>
                      <span>{file.kind === 'figure' ? (zh ? '图' : 'Fig') : (zh ? '表' : 'Table')}</span>
                      <button type="button" className={css.drawerResultFileName} title={artifactLabel(file.name, file.kind)} onClick={() => setOpenArtifacts((current) => { const next = new Set(current); if (next.has(file.url)) next.delete(file.url); else next.add(file.url); return next })}>{artifactLabel(file.name, file.kind)}</button>
                      <div className={css.drawerResultFileActions}>
                        {file.kind === 'figure' && <button type="button" title={zh ? '引用这张图，让 Agent 按指令调整' : 'Quote this figure'} onClick={() => window.dispatchEvent(new CustomEvent('theta:chart-quote', { detail: { chartName: artifactLabel(file.name, file.kind), figure: file.name, jobId: result.jobId } }))}>{zh ? '引用' : 'Quote'}</button>}
                        <a href={bundleUrl ?? file.url} download={bundleUrl ? `figure-${name.replace(/\.[^.]+$/u, '')}.zip` : name} title={zh ? '下载图、绘图数据与绘图代码' : 'Download figure, data and code'}>{zh ? '下载' : 'Get'}</a>
                      </div>
                      {expanded && file.kind === 'figure' && preview && <div className={css.drawerResultPreview}><img src={preview.url} alt={artifactLabel(file.name, file.kind)} loading="lazy" /></div>}
                    </div>
                  }) : <div className={css.inspectorEmpty}>
                    <p>{zh ? '这个任务还没整理出图表。' : 'No report yet for this run.'}</p>
                    {result.status === 'completed' && <button type="button" className={css.drawerResultRetry} onClick={() => window.dispatchEvent(new CustomEvent('theta:request-report', { detail: { jobId: result.jobId, modelId: result.modelId } }))}>{zh ? '让 Agent 整理这个任务的结果' : 'Ask the Agent to prepare this report'}</button>}
                  </div>}
                  {artifacts?.archiveUrl && <a className={css.drawerResultFile} href={artifacts.archiveUrl} download={`theta-${result.modelId}-figures-and-data.zip`}>
                    <span>{zh ? '打包' : 'Zip'}</span>
                    <strong>{zh ? '这个模型的图与数据打包下载' : 'Download this model\'s figures and data'}</strong>
                  </a>}
                </div>}
              </article>
            })}</div>}
            {hasRichResults && runId && <ResultsPreview runId={runId} results={results} onAttach={onAttach} />}
            {!resultCount && <p className={css.inspectorEmpty}>{zh ? '训练完成后，主题、指标和图表会出现在这里。' : 'Topics, metrics, and charts appear here after training.'}</p>}
            <button type="button" className={css.drawerDestination} disabled={!resultCount} onClick={() => onOpenResults('results')}>{zh ? '打开完整结果' : 'Open full results'}<ArrowRight size={14} /></button>
          </div>}
        </section>

        <section className={css.drawerSection} data-open={openSections.has('interpretation') || undefined}>
          {header('interpretation', <FileText size={17} />, zh ? '结果解读' : 'Result interpretation', resultCount ? (zh ? '可继续' : 'Available') : undefined)}
          {openSections.has('interpretation') && <div className={css.drawerSectionBody}>
            <p className={css.drawerInterpretationCopy}>{resultCount ? (zh ? '查看图表、原始数据与已有解读，选择证据继续研究。' : 'Explore charts, source data, and interpretations to continue your research.') : (zh ? '结果完成后可在这里进入对应的解读页面。' : 'The interpretation page becomes available after results complete.')}</p>
            <button type="button" className={css.drawerDestination} disabled={!resultCount} onClick={() => onOpenResults('interpretation')}>{zh ? '打开结果解读' : 'Open interpretation'}<ArrowRight size={14} /></button>
          </div>}
        </section>
      </div>
    </aside>
  )
}
