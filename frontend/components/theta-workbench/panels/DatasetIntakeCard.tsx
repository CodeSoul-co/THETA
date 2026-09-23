import { ComputationNotice } from './WorkbenchNotice'
import { useEffect, useMemo, useRef, useState } from 'react'
import { listDatasets, uploadDataset, type WebAgentInteraction, type WebDataset } from '../api/client.ts'
import {
  Button,
  IconCheckOutline16,
  IconCopyOutline16,
  IconShareOutline16,
} from '../ui/index.ts'
import { usePreferences } from '../preferences.tsx'
import css from '../styles/app.module.css'

interface DatasetIntakeCardProps {
  interaction: WebAgentInteraction
  storageScope: string
  legacyStorageScope?: string
  knownDatasetRefs?: string[]
  onEnsureProject: (suggestedName: string) => Promise<string>
  onDatasetsReady: (datasets: WebDataset[]) => void | Promise<void>
}

const ACCEPTED_DATASET = /\.(csv|tsv|json|jsonl|ndjson|txt|md|xlsx|xls|parquet|pdf|docx)$/iu
const MAX_FILE_BYTES = 200 * 1024 * 1024
const PROJECT_ID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/iu

const readableBytes = (value: number): string => {
  if (value < 1024) return `${value} B`
  if (value < 1024 * 1024) return `${Math.round(value / 1024)} KB`
  if (value < 1024 * 1024 * 1024) return `${(value / 1024 / 1024).toFixed(1)} MB`
  return `${(value / 1024 / 1024 / 1024).toFixed(1)} GB`
}

const storageKeyFor = (scope: string): string =>
  `theta.frontend.datasets.v1.${encodeURIComponent(scope)}`

const readStoredDatasets = (scope: string): WebDataset[] => {
  try {
    const value = JSON.parse(localStorage.getItem(storageKeyFor(scope)) ?? '[]') as unknown
    return Array.isArray(value)
      ? (value as WebDataset[]).filter((dataset) => !dataset.datasetRef.startsWith('local-') && !dataset.datasetRef.startsWith('pending-'))
      : []
  } catch {
    return []
  }
}

export const DatasetIntakeCard = ({ interaction, storageScope, legacyStorageScope, knownDatasetRefs = [], onEnsureProject, onDatasetsReady }: DatasetIntakeCardProps): React.ReactElement => {
  const { locale } = usePreferences()
  const initialDatasetsRef = useRef<WebDataset[] | undefined>(undefined)
  if (initialDatasetsRef.current == null) {
    const knownRefs = new Set(knownDatasetRefs)
    const current = readStoredDatasets(storageScope).filter((dataset) => knownRefs.has(dataset.datasetRef))
    const legacy = legacyStorageScope
      ? readStoredDatasets(legacyStorageScope).filter((dataset) => knownRefs.has(dataset.datasetRef))
      : []
    initialDatasetsRef.current = current.length > 0 || !legacyStorageScope
      ? current
      : legacy
  }
  const [datasets, setDatasets] = useState<WebDataset[]>(initialDatasetsRef.current)
  const [datasetRef, setDatasetRef] = useState(() => initialDatasetsRef.current?.[0]?.datasetRef ?? '')
  const [processing, setProcessing] = useState(false)
  const [processed, setProcessed] = useState(() => (initialDatasetsRef.current?.length ?? 0) > 0)
  const [dragging, setDragging] = useState(false)
  const [copied, setCopied] = useState(false)
  const [error, setError] = useState<string>()
  const [pendingFiles, setPendingFiles] = useState<Array<{ datasetRef: string; file: File }>>([])
  const fileInput = useRef<HTMLInputElement>(null)

  useEffect(() => {
    localStorage.setItem(storageKeyFor(storageScope), JSON.stringify(
      datasets.filter((dataset) => !dataset.datasetRef.startsWith('pending-')),
    ))
  }, [datasets, storageScope])

  useEffect(() => {
    if (!PROJECT_ID_PATTERN.test(storageScope)) return
    let cancelled = false
    void listDatasets(storageScope)
      .then(({ datasets: storedDatasets }) => {
        if (cancelled || storedDatasets.length === 0) return
        setDatasets((current) => [
          ...current.filter((dataset) => dataset.datasetRef.startsWith('pending-')),
          ...storedDatasets,
        ])
        setDatasetRef((current) => storedDatasets.some((dataset) => dataset.datasetRef === current)
          ? current
          : storedDatasets[0]?.datasetRef ?? '')
        setProcessed(true)
      })
      .catch(() => undefined)
    return () => { cancelled = true }
  }, [storageScope])

  const selected = useMemo(
    () => datasets.find((dataset) => dataset.datasetRef === datasetRef),
    [datasets, datasetRef],
  )

  const selectFiles = (files: File[]): void => {
    if (datasets.length > 0 || processing || processed) return
    const oversized = files.find((file) => file.size > MAX_FILE_BYTES)
    if (oversized) {
      setError(locale === 'zh-CN' ? `${oversized.name} 超过单文件 200 MiB 限制。` : `${oversized.name} exceeds the 200 MiB limit.`)
      return
    }
    const accepted = files.filter((file) => ACCEPTED_DATASET.test(file.name)).slice(0, 1)
    if (accepted.length === 0) {
      setError(locale === 'zh-CN' ? '请选择 CSV、TSV、TXT、Markdown、JSON/JSONL、Excel、Parquet、PDF 或 DOCX 文件。' : 'Choose a supported dataset file.')
      return
    }
    const createdAt = new Date().toISOString()
    const selectedFiles = accepted.map((file) => ({ datasetRef: `pending-${crypto.randomUUID()}`, file }))
    const selectedDatasets = selectedFiles.map(({ datasetRef: pendingRef, file }): WebDataset => ({
      datasetRef: pendingRef,
      name: file.name,
      sizeBytes: file.size,
      suffix: file.name.split('.').pop()?.toLowerCase() ?? '',
      createdAt,
    }))
    setPendingFiles(selectedFiles)
    setDatasets(selectedDatasets)
    setDatasetRef(selectedDatasets[0]?.datasetRef ?? '')
    setProcessed(false)
    setError(undefined)
  }

  const startProcessing = async (): Promise<void> => {
    if (!selected || processing) return
    setProcessing(true)
    setProcessed(false)
    setError(undefined)
    try {
      const filesToUpload = pendingFiles.filter((item) =>
        datasets.some((dataset) => dataset.datasetRef === item.datasetRef),
      )
      const uploaded: WebDataset[] = []
      const projectId = await onEnsureProject(selected.name.replace(/\.[^.]+$/u, '') || '数据分析项目')
      for (const item of filesToUpload) uploaded.push(await uploadDataset(projectId, item.file))
      const ready = uploaded.length > 0 ? uploaded : [selected]
      setDatasets((current) => [
        ...ready,
        ...current.filter((dataset) =>
          !dataset.datasetRef.startsWith('pending-') &&
          !ready.some((next) => next.datasetRef === dataset.datasetRef || next.name === dataset.name),
        ),
      ])
      setDatasetRef(ready[0]?.datasetRef ?? '')
      setPendingFiles([])
      setProcessed(true)
      await onDatasetsReady(ready)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause))
    } finally {
      setProcessing(false)
    }
  }

  const copyReply = (): void => {
    const reply = '好的，您可以上传本地数据集文件，系统将协助您完成数据的处理、预览与管理。'
    void navigator.clipboard?.writeText(reply).catch(() => undefined)
    setCopied(true)
    window.setTimeout(() => setCopied(false), 1200)
  }

  return (
    <section className={`${css.datasetIntakePanel} ${processed ? css.datasetIntakePanelCompleted : ''} ${dragging ? css.contextCardDragging : ''}`} aria-label={interaction.card?.title ?? '上传本地数据集'} aria-disabled={processed}>
      <ComputationNotice sizeBytes={selected?.sizeBytes} locale={locale} />
      {processed ? (
        <div className={css.datasetUploadedSummary} role="status">
          <span><IconCheckOutline16 /></span>
          <div>
            <strong>{locale === 'zh-CN' ? '数据集已上传' : 'Dataset uploaded'}</strong>
            <small>{selected != null ? `${selected.name} · ${readableBytes(selected.sizeBytes)}` : (locale === 'zh-CN' ? '已绑定到当前项目' : 'Attached to this project')}</small>
          </div>
        </div>
      ) : (
        <>
          <p className={css.datasetIntro}>好的，您可以上传本地数据集文件，系统将协助您完成<br />数据的处理、预览与管理。</p>
          <div
            className={`${css.contextDropzone} ${datasets.length > 0 ? css.contextDropzoneLocked : ''}`}
            aria-disabled={datasets.length > 0 || processing}
            onDragEnter={(event) => { event.preventDefault(); if (datasets.length === 0 && !processing) setDragging(true) }}
            onDragOver={(event) => event.preventDefault()}
            onDragLeave={() => setDragging(false)}
            onDrop={(event) => {
              event.preventDefault()
              setDragging(false)
              if (datasets.length === 0 && !processing) selectFiles(Array.from(event.dataTransfer.files))
            }}
          >
            <Button size="sm" variant="primary" disabled={datasets.length > 0 || processing} onClick={() => fileInput.current?.click()}>
              <IconShareOutline16 />{locale === 'zh-CN' ? '从本地选择文件' : 'Choose local files'}
            </Button>
            <span>{locale === 'zh-CN' ? '支持 CSV、Excel、JSON、Parquet、PDF、DOCX、TXT 等格式' : 'CSV, Excel, JSON, Parquet, PDF, DOCX and text supported'}</span>
            <small>{locale === 'zh-CN' ? '单个文件最大 200 MiB' : 'Up to 200 MiB per file'}</small>
            <input
              ref={fileInput}
              type="file"
              hidden
              disabled={datasets.length > 0 || processing}
              accept=".csv,.tsv,.json,.jsonl,.ndjson,.txt,.md,.xlsx,.xls,.parquet,.pdf,.docx"
              onChange={(event) => {
                selectFiles(Array.from(event.target.files ?? []))
                event.target.value = ''
              }}
            />
          </div>
          {datasets.length > 0 && (
            <div className={css.contextCardActions}>
              <select value={datasetRef} aria-label={locale === 'zh-CN' ? '选择数据集' : 'Select dataset'} onChange={(event) => { setDatasetRef(event.target.value); setProcessed(false) }}>
                {datasets.map((dataset) => (
                  <option key={dataset.datasetRef} value={dataset.datasetRef}>{dataset.name} · {readableBytes(dataset.sizeBytes)}</option>
                ))}
              </select>
              {selected != null && <small>{selected.suffix.toUpperCase()} · {selected.datasetRef.startsWith('pending-') ? (locale === 'zh-CN' ? '等待上传' : 'waiting to upload') : (locale === 'zh-CN' ? '已上传到后端' : 'uploaded to backend')}</small>}
              <Button size="sm" variant="primary" disabled={!selected || processing} onClick={() => void startProcessing()}>
                {processing ? (locale === 'zh-CN' ? '处理中…' : 'Processing…') : (locale === 'zh-CN' ? '开始处理' : 'Start processing')}
              </Button>
            </div>
          )}
        </>
      )}
      {error != null && <div className={css.formError} role="alert">{error}</div>}
      <footer className={css.datasetFooter}>
        <time>{new Intl.DateTimeFormat(locale, { hour: '2-digit', minute: '2-digit' }).format(new Date())}</time>
        {!processed && <>
          <span />
          <button type="button" aria-label={copied ? '已复制' : '复制'} onClick={copyReply}><IconCopyOutline16 /></button>
        </>}
      </footer>
    </section>
  )
}
