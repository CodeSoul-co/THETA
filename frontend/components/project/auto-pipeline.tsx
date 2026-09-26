"use client"

import { toast } from 'sonner'
import { ComputationNotice, SetupError } from '@/components/theta-workbench/panels/WorkbenchNotice'
import { useEffect, useRef, useState } from "react"
import { AlertCircle, Check, Loader2, Upload } from "lucide-react"
import { Button } from "@/components/ui/button"
import { ExecutionLog } from "./execution-log"
import type { TrainingWorkerState } from "@/lib/training-progress"
import { Progress } from "@/components/ui/progress"
import { apiFetch, API_BASE } from '@/lib/api/config'
import { BackendAPI, SimpleETMAPI, type TrainStatusResponse } from "@/lib/api/backend"
import { isUploadedFileId, uploadManualFiles } from "@/lib/manual-upload"
import { statusLabel, systemText } from "@/lib/presentation"
import { AnalysisConfigPanel, type AnalysisConfig } from "./analysis-config-panel"
import { ColumnSelectPanel, type ColumnSelection } from "./column-select-panel"
import { ETMAgentAPI } from "@/lib/api/etm-agent"
import { isTextDocument, type DatasetPreview } from "@/lib/dataset-input"
import { DATASET_ACCEPT, datasetFilesError, documentCollectionError, isDocumentCollection } from '@/lib/dataset-files'
import { useFileDrop } from '@/lib/use-file-drop'
import { useProjectDraft } from "@/lib/use-project-draft"

interface PipelineResult { success: boolean; taskId?: string; dataset?: string; metrics?: Record<string, number>; topicWords?: Record<string, string[]>; duration: number }
interface AutoPipelineProps {
  projectKey: string
  datasetName?: string
  projectName: string
  mode?: "zero_shot" | "unsupervised" | "supervised"
  numTopics?: number
  initialTaskId?: string | null
  pipelineStatus?: "running" | "completed" | "error" | "draft"
  onComplete?: (result: PipelineResult) => void
  onError?: (error: string) => void
  onUploadComplete?: (dataset: string) => void
  onTaskCreated?: (id: string) => void
  onConfigConfirmed?: (config: AnalysisConfig) => void
  onDlcStarted?: () => void
  onViewResults?: () => void
}
type JobState = TrainStatusResponse & { queued_models?: string[]; states?: TrainingWorkerState[]; workers?: { id: string; model: string }[] }
const phases = ["上传数据", "数据预处理", "模型训练", "模型评估", "生成可视化"]
const phaseIndex = (phase?: string) => ({ preparing: 1, preparing_data: 1, preprocessing: 1, training: 2, evaluating: 3, evaluation: 3, visualizing: 4, visualization: 4, publishing: 4, uploading: 4 }[phase ?? ""] ?? 1)

/** Independent manual lifecycle; every progress update comes from a real job. */
export function AutoPipeline(props: AutoPipelineProps) {
  const dataset = props.datasetName || props.projectName.trim().replace(/\s+/g, "_").replace(/[^\w\u4e00-\u9fa5-]/g, "").toLowerCase() || "dataset"
  const callbacks = useRef(props)
  callbacks.current = props
  const folderInput = useRef<HTMLInputElement>(null)
  useEffect(() => { folderInput.current?.setAttribute("webkitdirectory", ""); folderInput.current?.setAttribute("directory", "") })
  const [files, setFiles] = useState<File[]>([])
  const [replacing, setReplacing] = useState(false)
  const [uploads, setUploads] = useState<{ name: string; fileId: string; size: number; inputKind?: string }[]>([])
  const [fileId, setFileId] = useState<string | null>(null)
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [submitting, setSubmitting] = useState(false)
  const [recovering, setRecovering] = useState(true)
  const [taskId, setTaskId] = useState<string | null>(props.initialTaskId ?? null)
  const [task, setTask] = useState<JobState | null>(null)
  const [selection, setSelection] = useState<ColumnSelection | null>(null)
  const [draft, setDraft] = useProjectDraft<{ fileId: string | null; selection: ColumnSelection | null; columnsOpen: boolean; configOpen: boolean }>(props.projectKey + ':data', { fileId: null, selection: null, columnsOpen: false, configOpen: false })
  const [columnsOpen, setColumnsOpen] = useState(false)
  const [configOpen, setConfigOpen] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [pollError, setPollError] = useState<string | null>(null)
  const [logs, setLogs] = useState<string[]>([])
  const completed = useRef<string | null>(null)
  const lastObservation = useRef("")
  const busy = useRef(false)
  const log = (message: string) => setLogs(prev => [...prev.slice(-99), `[${new Date().toLocaleTimeString('zh-CN')}] ${message}`])

  useEffect(() => {
    let cancelled = false
    void (async () => {
      try {
        const [knownFiles, jobs] = await Promise.all([BackendAPI.getFiles(), BackendAPI.getTrainJobs()])
        if (cancelled) return
        const matching = knownFiles.filter(file => file.dataset_name === dataset && isUploadedFileId(file.id))
        setUploads(matching.map(file => ({ name: file.filename, fileId: String(file.id), size: Number(file.size) || 0, inputKind: file.input_kind })))
        if (matching[0]) { setFileId(String(matching[0].id)); setUploadProgress(100) }
        if (props.initialTaskId) { setTaskId(props.initialTaskId); return }
        // Recover a task whose HTTP response was lost instead of duplicating it.
        const existing = jobs.find(job => job.dataset_name === dataset && (props.pipelineStatus !== 'draft' || ['pending', 'creating', 'running', 'cancelling'].includes(job.status)))
        if (existing) { setTaskId(String(existing.job_id)); callbacks.current.onTaskCreated?.(String(existing.job_id)) }
      } catch (e) { if (!cancelled) setError(e instanceof Error ? e.message : '无法恢复项目状态') }
      finally { if (!cancelled) setRecovering(false) }
    })()
    return () => { cancelled = true }
  }, [dataset])

  useEffect(() => {
    if (recovering) return
    if (draft.fileId && uploads.some(file => file.fileId === draft.fileId)) {
      setFileId(draft.fileId); setSelection(draft.selection)
      setColumnsOpen(draft.columnsOpen); setConfigOpen(draft.configOpen)
    }
  }, [recovering])

  const selectedUpload = uploads.find(file => file.fileId === fileId)
  const textInput = selectedUpload?.inputKind === 'text' || isTextDocument(selectedUpload?.name ?? '')
  const [textPreview, setTextPreview] = useState<DatasetPreview>()
  const [previewError, setPreviewError] = useState('')
  const [previewAttempt, setPreviewAttempt] = useState(0)
  useEffect(() => {
    setTextPreview(undefined); setPreviewError('')
    if (recovering || !fileId || !textInput) return
    let cancelled = false
    setSelection(null); setColumnsOpen(false)
    void ETMAgentAPI.getDatasetPreview(dataset, fileId).then(preview => {
      if (cancelled) return
      const value = { textColumn: preview.textColumn || 'text', metaColumns: [] }
      setTextPreview(preview); setSelection(value)
      setDraft(prev => ({ ...prev, fileId, selection: value, columnsOpen: false }))
    }).catch(error => { if (!cancelled) setPreviewError(error instanceof Error ? error.message : '正文读取失败，请重试') })
    return () => { cancelled = true }
  }, [dataset, fileId, textInput, recovering, previewAttempt])

  const changeColumnsOpen = (open: boolean) => { setColumnsOpen(open); setDraft(prev => ({ ...prev, fileId, columnsOpen: open })) }
  const changeConfigOpen = (open: boolean) => { setConfigOpen(open); setDraft(prev => ({ ...prev, fileId, configOpen: open })) }

  useEffect(() => {
    if (!taskId) return
    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | undefined
    const poll = async () => {
      let terminal = false
      try {
        const next: JobState = await BackendAPI.getTrainStatus(Number(taskId))
        if (cancelled) return
        setTask(next); setPollError(null); setError(null)
        const observation = `${next.status}:${next.current_step}:${next.progress}:${next.error_message}`
        if (observation !== lastObservation.current) { lastObservation.current = observation; log(next.error_message || systemText(next.message || statusLabel(next.status))) }
        terminal = ['succeeded', 'failed', 'cancelled'].includes(next.status)
        if (next.status === 'succeeded' && completed.current !== taskId) {
          completed.current = taskId
          callbacks.current.onComplete?.({ success: true, taskId, dataset, duration: Math.max(0, Date.now() - Date.parse(next.created_at)) })
        }
        if (next.status === 'failed' || next.status === 'cancelled') {
          setError(next.status === 'cancelled' ? '任务已取消，原始数据与记录保留。' : next.error_message || '训练失败，请查看真实错误后调整参数。')
          callbacks.current.onError?.(next.error_message || '任务已结束，需处理')
        }
      } catch (e) { if (!cancelled) setPollError(`状态暂时无法同步，后台任务不会因此重启。${e instanceof Error ? e.message : ''}`) }
      if (!cancelled && !terminal) timer = setTimeout(poll, 2500)
    }
    void poll()
    return () => { cancelled = true; if (timer) clearTimeout(timer) }
  }, [taskId, dataset])

  const selectFiles = (next: File[]) => {
    if (busy.current) return
    const fromFolder = next.some(file => !!file.webkitRelativePath)
    const problem = (fromFolder || next.length > 1 && next.every(file => isTextDocument(file.name))) ? documentCollectionError(next) : datasetFilesError(next)
    if (problem) { setError(problem); return }
    setFiles(next); setError(null); setUploadProgress(0)
    toast.success(`已添加 ${next.length} 个文件，请点击“上传并配置分析”继续`)
  }
  const { dragging, dropProps } = useFileDrop(selectFiles, uploading, setError)

  const upload = async () => {
    if (busy.current || !files.length) return
    busy.current = true; setUploading(true); setError(null)
    try {
      log(`开始上传 ${files.length} 个文件`)
      let receipts: { name: string; fileId: string; size: number; inputKind?: string }[] = await uploadManualFiles(files, (file, progress) => SimpleETMAPI.uploadDataset(file, dataset, progress), setUploadProgress)
      if (isDocumentCollection(files) && files.every(file => isTextDocument(file.name))) {
        const combined = await apiFetch<{ file_id: number; name: string; size: number; inputKind: string }>(API_BASE, `/api/datasets/${encodeURIComponent(dataset)}/combine`, {
          method: 'POST', body: JSON.stringify({ fileIds: receipts.map(file => file.fileId), sourceNames: files.map(file => file.webkitRelativePath || file.name) }), timeoutMs: 300_000,
        })
        receipts = [{ ...combined, fileId: String(combined.file_id) }, ...receipts]
      }
      const directText = receipts[0].inputKind === 'text' || isTextDocument(receipts[0].name)
      setUploads(previous => [...receipts, ...previous.filter(file => !receipts.some(next => next.fileId === file.fileId))]); setFileId(receipts[0].fileId)
      setTaskId(null); setTask(null); setPollError(null); completed.current = null; lastObservation.current = ''
      setReplacing(false); setFiles([]); setConfigOpen(false)
      setSelection(null)
      setDraft({ fileId: receipts[0].fileId, selection: null, columnsOpen: receipts.length === 1 && !directText, configOpen: false })
      callbacks.current.onUploadComplete?.(dataset)
      log(directText ? '上传成功，正在直接读取正文，无需选择数据列。' : '上传成功，请选择本次分析的文本列与元数据。')
      toast.success(`上传成功，共 ${files.length} 个文件`, { description: directText ? '正在读取正文，随后可配置分析参数' : '请选择本次分析的正文列' })
      setColumnsOpen(receipts.length === 1 && !directText)
    } catch (e) { const message = e instanceof Error ? e.message : '上传失败'; setError(message); log(message) }
    finally { busy.current = false; setUploading(false) }
  }

  const start = (config: AnalysisConfig) => {
    if (!config.models.length) { setError('请至少选择一个模型后再继续。'); return false }
    if (busy.current || !fileId || !selection?.textColumn) { setError('请先上传数据，等待读取正文或选择表格中的正文列。'); return false }
    if (config.models.includes('dtm') && !selection.timeColumn) { setError('DTM 需要真实时间列，请上传包含正文和时间列的表格。'); return false }
    if (config.models.includes('stm') && !selection.metaColumns.length) { setError('STM 需要元数据列作为协变量，请上传包含正文和元数据的表格。'); return false }
    if (config.mode === 'supervised' && !selection.labelColumn) { setError('有监督嵌入需要标签列，请上传包含正文和标签列的表格。'); return false }
    busy.current = true; setSubmitting(true); setError(null)
    callbacks.current.onConfigConfirmed?.(config)
    void (async () => {
      try {
        const modelParams = Object.fromEntries(config.models.map(model => [model, config.parameters[model] ?? {}]))
        const response = await BackendAPI.startTraining({ file_id: Number(fileId), dataset_name: dataset, model_type: config.models.join(','), model_params: modelParams,
          num_topics: Number(modelParams[config.models[0]].num_topics ?? modelParams[config.models[0]].max_topics ?? 20), vocab_size: config.vocabSize,
          plot_language: config.plotLanguage, stopwords_id: config.stopwords ? Number(config.stopwords.id) : undefined, model_size: config.modelSize, mode: config.mode,
          embedding_provider: config.embeddingProvider, cloud_confirmed: config.cloudConfirmed, external_request_limit: config.externalRequestLimit,
          cloud_selection: config.cloudSelection,
          text_column: selection.textColumn, meta_columns: selection.metaColumns, time_column: selection.timeColumn, label_column: selection.labelColumn })
        setTaskId(String(response.id)); setTask(null)
        callbacks.current.onTaskCreated?.(String(response.id))
        log('任务已保存，服务正在检查数据和运行环境。')
      } catch (e) { const message = e instanceof Error ? e.message : '任务提交失败'; setError(message); log(message) }
      finally { busy.current = false; setSubmitting(false) }
    })()
    return true
  }

  const running = !!taskId && (!task || ['pending', 'creating', 'running', 'cancelling'].includes(task.status))
  const done = task?.status === 'succeeded'
  const current = taskId ? phaseIndex(task?.current_step) : 0
  return <div className="min-w-0 space-y-5 p-4 [overflow-wrap:anywhere] sm:p-6 lg:p-8">
    {props.onViewResults && <Button variant="outline" onClick={props.onViewResults}>查看已有结果（不影响后台训练）</Button>}
    <div className="flex flex-wrap items-center justify-between gap-3"><div><h1 className="text-2xl font-semibold text-slate-900">{props.projectName}</h1><p className="mt-2 text-sm text-slate-500">数据集：{dataset}</p></div>
      <span className="rounded-full bg-slate-100 px-3 py-1 text-xs text-slate-600">{recovering ? '正在恢复项目' : done ? '已完成' : error ? '需要处理' : running ? '分析中' : fileId ? '待配置' : '请上传数据'}</span></div>
    <ol className="flex flex-wrap gap-3 rounded-xl border bg-white p-4 text-sm">{phases.map((phase, index) => <li key={phase} className={`flex items-center gap-2 rounded-lg px-3 py-2 ${done || (index === 0 && fileId) || (running && index < current) ? 'bg-emerald-50 text-emerald-700' : running && current === index ? 'bg-blue-50 text-blue-700' : 'text-slate-400'}`}>
      {done || (index === 0 && fileId) || (running && index < current) ? <Check className="size-4" /> : running && current === index ? <Loader2 className="size-4 animate-spin" /> : <span>{index + 1}</span>}{phase}</li>)}</ol>
    {error && <div role="alert" className="rounded-xl border border-red-200 bg-red-50 p-5 text-sm text-red-700"><p className="flex items-center gap-2"><AlertCircle className="size-4" /></p><SetupError error={error} />{!running && fileId && <Button variant="outline" className="mt-3" onClick={() => { setTaskId(null); setTask(null); setError(null); textInput ? changeConfigOpen(true) : changeColumnsOpen(true) }}>调整配置后重试</Button>}</div>}
    {pollError && <p role="status" className="text-sm text-amber-700">{pollError}</p>}
    {!!task?.queued_models?.length && <p className="text-sm text-slate-600">排队模型：{task.queued_models.map(model => model.toUpperCase()).join('、')}。按顺序执行，关闭页面不会中断队列。</p>}
    {running && <Button variant="outline" onClick={async () => { try { const cancelled = await BackendAPI.cancelTraining(Number(taskId)); setTask(cancelled); setError(cancelled.status === 'cancelled' ? '任务已取消，可以调整数据与参数后重新开始。' : null) } catch (e) { setError(e instanceof Error ? e.message : '取消失败，请重试') } }}>取消本次训练</Button>}
    {(running || submitting) && <div role="status" className="space-y-4 rounded-2xl border bg-white p-6"><p className="flex items-center gap-2 text-sm font-medium"><Loader2 className="size-4 animate-spin" />{submitting ? '正在保存分析任务…' : systemText(task?.message || '正在检查数据与运行环境…')}</p><Progress value={task?.progress ?? 0} /><p className="text-xs leading-5 text-slate-500">{task?.progress ?? 0}% 为 worker 阶段进度，不是剩余时间估算。可以离开页面，返回后恢复真实任务状态。</p>{task?.states?.map(state => <p key={state.id} className="text-xs text-slate-600">{task.workers?.find(item => item.id === state.id)?.model.toUpperCase()} · {statusLabel(state.status)} · {systemText(state.phase)} · {state.percent}%</p>)}</div>}
    {!recovering && !running && !done && !submitting && <section className="space-y-4 rounded-2xl border bg-white p-6"><h2 className="font-semibold">{fileId ? '选择数据与分析参数' : '上传数据'}</h2>
      {(!fileId || replacing) && <><p className="text-sm text-slate-500">支持 Excel、CSV、TXT、PDF、DOCX、JSON 等格式。表格选择正文列；TXT、Markdown、PDF、Word 直接读取正文，无需选择数据列。</p><label {...dropProps} aria-busy={uploading} className={`flex cursor-pointer items-center justify-center gap-2 rounded-xl border-2 border-dashed p-8 text-sm text-blue-700 transition-colors focus-within:ring-2 focus-within:ring-blue-500 ${dragging ? "border-blue-500 bg-blue-50" : "border-slate-200"}`}><Upload className="size-5" />{dragging ? "松开即可添加文件" : "拖拽文件到这里，或点击选择文件"}<input className="sr-only" aria-label="选择文件" type="file" multiple accept={DATASET_ACCEPT} disabled={uploading} onChange={e => { selectFiles(Array.from(e.target.files || [])); e.target.value = "" }} /></label><input ref={folderInput} type="file" className="sr-only" aria-label="选择文件夹" disabled={uploading} onChange={event => { selectFiles(Array.from(event.target.files || []).filter(file => !file.webkitRelativePath.split("/").some(part => part.startsWith(".")))); event.target.value = "" }} /><Button variant="outline" disabled={uploading} onClick={() => folderInput.current?.click()}>选择文件夹（按内容合并文档）</Button>{replacing && <Button variant="outline" disabled={uploading} onClick={() => { setReplacing(false); setFiles([]); setError(null) }}>取消更换，保留原数据</Button>}<ComputationNotice sizeBytes={files.reduce((sum, file) => sum + file.size, 0)} />{files.map((file, index) => <p key={`${file.name}-${index}`} className="text-sm text-slate-600">{file.name} · {(file.size / 1024 / 1024).toFixed(1)} MB</p>)}{uploading ? <><Progress value={uploadProgress} /><p className="text-sm">正在上传 · {uploadProgress}%</p></> : <Button disabled={!files.length} onClick={() => void upload()}>上传并配置分析</Button>}</>}
      {fileId && !replacing && <><Button variant="outline" onClick={() => { setReplacing(true); setColumnsOpen(false); setConfigOpen(false); setError(null) }}>重新上传 / 更换数据</Button><p className="text-xs text-slate-500">新文件上传成功后切换，原始数据与已有结果保留。</p><label className="block space-y-2 text-sm">本次分析文件<select aria-label="本次分析文件" className="block w-full rounded-lg border p-2" value={fileId} onChange={e => { setFileId(e.target.value); setSelection(null); setDraft({ fileId: e.target.value, selection: null, columnsOpen: false, configOpen: false }) }}>{uploads.map(file => <option key={file.fileId} value={file.fileId}>{file.name}</option>)}</select></label><p className="text-xs text-slate-500">每个任务分析一个文件。多文件上传后，请明确选择本次文件。</p>{textInput ? <div className="space-y-3 rounded-xl border bg-slate-50 p-4">
        <h3 className="text-sm font-semibold">正文预览</h3><p className="text-xs text-slate-500">直接读取文本，无需选择数据列。按非空行、PDF 页面或 Word 段落生成分析记录。</p>
        {previewError ? <p role="alert" className="text-sm text-red-700">{previewError}<button className="ml-2 underline" onClick={() => setPreviewAttempt(value => value + 1)}>重新读取</button></p> : !textPreview ? <p role="status" className="text-sm">正在读取正文…</p> : <><p className="text-xs text-slate-500">共 {textPreview.totalRecords ?? textPreview.rows.length} 条正文记录，预览前 5 条</p><div className="max-h-72 space-y-3 overflow-y-auto">{(textPreview.segments ?? textPreview.rows.map(row => ({ text: row[0] }))).map((segment, index) => <article key={index} className="rounded-lg bg-white p-3"><p className="mb-2 text-xs text-slate-400">{'source_file' in segment && <span>{String(segment.source_file)} · </span>}{'page' in segment ? `第 ${segment.page} 页` : 'paragraph' in segment ? `第 ${segment.paragraph} 段` : `正文 ${index + 1}`}</p><p className="whitespace-pre-wrap text-sm leading-6 [overflow-wrap:anywhere]">{segment.text}</p></article>)}</div></>}
      </div> : <Button variant="outline" onClick={() => changeColumnsOpen(true)}>选择数据列</Button>}{selection && <Button className="ml-2" onClick={() => changeConfigOpen(true)}>配置分析参数</Button>}</>}
    </section>}
    <ExecutionLog states={task?.states} workers={task?.workers} logs={logs} running={running || submitting} />
    {!textInput && <ColumnSelectPanel key={fileId} projectKey={props.projectKey} open={columnsOpen} onReplace={() => { setReplacing(true); setConfigOpen(false); setError(null) }} onOpenChange={changeColumnsOpen} datasetName={dataset} jobId={fileId} onConfirm={value => { setSelection(value); setColumnsOpen(false); setConfigOpen(true); setDraft({ fileId, selection: value, columnsOpen: false, configOpen: true }) }} />}
    <AnalysisConfigPanel datasetSizeBytes={uploads.find(file => file.fileId === fileId)?.size} projectKey={props.projectKey} open={configOpen} onOpenChange={changeConfigOpen} datasetName={dataset} error={error} onConfirm={start} />
  </div>
}
