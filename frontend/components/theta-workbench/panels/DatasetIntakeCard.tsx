import { DATASET_ACCEPT, datasetFilesError, documentCollectionError, isDocumentCollection } from '@/lib/dataset-files'
import { useFileDrop } from '@/lib/use-file-drop'
import { ComputationNotice } from './WorkbenchNotice'
import { useEffect, useMemo, useRef, useState } from 'react'
import { listDatasets, uploadDatasetFiles, type WebAgentInteraction, type WebDataset } from '../api/client.ts'
import {
  Button,
  IconCheckOutline16,
  IconCopyOutline16,
  IconShareOutline16,
} from '../ui/index.ts'
import { usePreferences } from '../preferences.tsx'
import css from '../styles/app.module.css'

interface DatasetIntakeCardProps {
  disabled?: boolean
  interaction: WebAgentInteraction
  storageScope: string
  legacyStorageScope?: string
  knownDatasetRefs?: string[]
  onEnsureProject: (suggestedName: string) => Promise<string>
  onDatasetsReady: (datasets: WebDataset[]) => void | Promise<void>
}

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

export const DatasetIntakeCard = ({ disabled = false, interaction, storageScope, legacyStorageScope, knownDatasetRefs = [], onEnsureProject, onDatasetsReady }: DatasetIntakeCardProps): React.ReactElement => {
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
  const [replacing, setReplacing] = useState(false)
  const previousSelection = useRef(datasetRef)
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
        setDatasetRef((current) => current.startsWith('pending-') || storedDatasets.some((dataset) => dataset.datasetRef === current)
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
    if (processing || disabled) return
    const problem = isDocumentCollection(files) ? documentCollectionError(files, locale) : datasetFilesError(files, locale)
    if (problem) { setError(problem); return }
    const accepted = files
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
    setDatasets(current => [...selectedDatasets, ...current.filter(item => !item.datasetRef.startsWith('pending-'))])
    setDatasetRef(selectedDatasets[0]?.datasetRef ?? '')
    setProcessed(false)
    setError(undefined)
  }

  const { dragging, dropProps } = useFileDrop(selectFiles, processing || disabled, setError)

  const startProcessing = async (): Promise<void> => {
    if (!selected || processing || disabled) return
    setProcessing(true)
    setProcessed(false)
    setError(undefined)
    try {
      const filesToUpload = pendingFiles.filter((item) =>
        selected.datasetRef.startsWith('pending-') && datasets.some(dataset => dataset.datasetRef === item.datasetRef),
      )
      const uploaded: WebDataset[] = []
      const projectId = await onEnsureProject(selected.name.replace(/\.[^.]+$/u, '') || '数据分析项目')
      if (filesToUpload.length) uploaded.push(await uploadDatasetFiles(projectId, filesToUpload.map(item => item.file), locale))
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
      await onDatasetsReady(ready)
      setProcessed(true); setReplacing(false)
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
    <section className={`${css.datasetIntakePanel} ${processed ? css.datasetIntakePanelCompleted : ''} ${dragging ? css.contextCardDragging : ''}`} aria-label={interaction.card?.title ?? '上传本地数据集'} aria-busy={processing}>
      <ComputationNotice sizeBytes={selected?.sizeBytes} locale={locale} />
      {processed && !replacing ? (
        <div className={css.datasetUploadedSummary} role="status">
          <span><IconCheckOutline16 /></span>
          <div>
            <strong>{locale === 'zh-CN' ? '数据集已上传' : 'Dataset uploaded'}</strong>
            <small>{selected != null ? `${selected.name} · ${readableBytes(selected.sizeBytes)}` : (locale === 'zh-CN' ? '已绑定到当前项目' : 'Attached to this project')}</small>
          </div>
          <Button size="sm" disabled={disabled} onClick={() => { previousSelection.current = datasetRef; setReplacing(true); setError(undefined) }}>{locale === 'zh-CN' ? '重新上传 / 更换数据' : 'Replace dataset'}</Button>
        </div>
      ) : (
        <>
          <p className={css.datasetIntro}>好的，您可以上传本地数据集文件，系统将协助您完成<br />数据的处理、预览与管理。</p>
          <div
            className={`${css.contextDropzone} ${dragging ? css.contextCardDragging : ''}`}
            aria-disabled={processing || disabled}
            {...dropProps}
          >
            <Button size="sm" variant="primary" disabled={processing || disabled} onClick={() => fileInput.current?.click()}>
              <IconShareOutline16 />{locale === 'zh-CN' ? '从本地选择文件' : 'Choose local files'}
            </Button>
            <strong>{locale === 'zh-CN' ? (dragging ? '松开即可添加文件' : '拖拽文件到这里，也可点击选择') : (dragging ? 'Drop your file here' : 'Drag a file here, or choose a file')}</strong>
            <span>{locale === 'zh-CN' ? '支持 CSV、Excel、JSON、Parquet、PDF、DOCX、TXT 等格式' : 'CSV, Excel, JSON, Parquet, PDF, DOCX and text supported'}</span>
            <small>{locale === 'zh-CN' ? '单文件或文档文件夹最大 200 MiB；按正文段落切分' : 'Up to 200 MiB per file'}</small>
            <input
              ref={fileInput}
              type="file"
              hidden
              disabled={processing || disabled}
              accept={DATASET_ACCEPT}
              onChange={(event) => {
                selectFiles(Array.from(event.target.files ?? []))
                event.target.value = ''
              }}
            />
          </div>
          {replacing && <Button size="sm" disabled={processing} onClick={() => { setDatasetRef(previousSelection.current); setDatasets(current => current.filter(item => !item.datasetRef.startsWith('pending-'))); setPendingFiles([]); setProcessed(true); setReplacing(false); setError(undefined) }}>{locale === 'zh-CN' ? '取消更换，保留原数据' : 'Cancel replacement'}</Button>}
          {datasets.length > 0 && (
            <div className={css.contextCardActions}>
              <select value={datasetRef} aria-label={locale === 'zh-CN' ? '选择数据集' : 'Select dataset'} onChange={(event) => { setDatasetRef(event.target.value); setProcessed(false) }}>
                {datasets.map((dataset) => (
                  <option key={dataset.datasetRef} value={dataset.datasetRef}>{dataset.name} · {readableBytes(dataset.sizeBytes)}</option>
                ))}
              </select>
              {selected != null && <small>{selected.suffix.toUpperCase()} · {selected.datasetRef.startsWith('pending-') ? (locale === 'zh-CN' ? '等待上传' : 'waiting to upload') : (locale === 'zh-CN' ? '已上传到后端' : 'uploaded to backend')}</small>}
              <Button size="sm" variant="primary" disabled={!selected || processing || disabled} onClick={() => void startProcessing()}>
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
