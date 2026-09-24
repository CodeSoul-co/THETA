import { DATASET_ACCEPT, datasetFilesError, documentCollectionError, isDocumentCollection } from '@/lib/dataset-files'
import { useFileDrop } from '@/lib/use-file-drop'
import { ComputationNotice, SetupError } from './WorkbenchNotice'
import { openSetup, errorGuidance } from '@/lib/workbench-guidance'
import { Fragment, useEffect, useMemo, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { Check, ChevronRight, Clock3, Pencil, X, AlertCircle } from 'lucide-react'
import approvalCss from './ActionApprovalCard.module.css'
import type {
  WebAgentInteraction,
  WebAttachment,
  WebDataset,
  WebMessage,
  WebReasoning,
  WebResultCatalog,
  WebRunStatus,
  WebTokenUsage,
} from '../api/client.ts'
import { uploadDatasetFiles } from '../api/client.ts'
import {
  Button,
  IconCopyOutline16,
  IconCloseFill14,
  IconPlusOutline16,
  IconSendOutline16,
  IconChevronDownOutline14,
  Menu,
  type MenuEntry,
  ThetaCubeMark,
  StateDot,
} from '../ui/index.ts'
import { usePreferences } from '../preferences.tsx'
import { useInferenceSettings } from '../inference-settings.tsx'
import { AgentActivityTrace } from './AgentActivityTrace.tsx'
import { ApprovalPanel } from './ApprovalPanel.tsx'
import { DatasetIntakeCard } from './DatasetIntakeCard.tsx'
import { ExistingDataArtwork, GuidedStartArtwork, ResearchIdeaArtwork } from './StarterCardArtworks.tsx'
import { TrainingStatusBar } from './TrainingStatusBar.tsx'
import { artifactLabel, conversationDisplayText } from '@/lib/presentation'
import { CollapsibleMessage } from './CollapsibleMessage.tsx'
import { MarkdownText } from '../ui/markdown/MarkdownText.tsx'
import css from '../styles/app.module.css'
import { InferenceSelector } from './InferenceSelector.tsx'
import { DatasetFileIcon } from '../ui/DatasetFileIcon.tsx'
import { useProjectDraft } from '@/lib/use-project-draft'

const WORKBENCH_PREFILL_KEY = 'theta.workspace.landing-prefill.v1'
const STARTER_TITLE = '今天想和我研究什么？'
const STARTER_DESCRIPTION = '告诉我研究问题，我会陪你理解数据、选择方法并验证结果。'

const analysisModeEntries = (locale: string): MenuEntry[] => [
  {
    id: 'topic',
    label: locale === 'zh-CN' ? '主题分析' : 'Topic analysis',
  },
  {
    id: 'free',
    label: locale === 'zh-CN' ? '自由分析' : 'Free analysis',
  },
]

const readLandingPrefill = (): string => {
  const value = localStorage.getItem(WORKBENCH_PREFILL_KEY) ?? ''
  localStorage.removeItem(WORKBENCH_PREFILL_KEY)
  return value
}

export interface QueuedChatMessage { id: string; text: string; attachments: WebAttachment[]; modelPreference?: string }

interface ConversationPaneProps {
  datasetSizeBytes?: number
  analysisMode?: 'topic' | 'free'
  onAnalysisModeChange?: (mode: 'topic' | 'free') => void | Promise<void>
  analysisModeDisabled?: boolean
  compact?: boolean
  projectStorageScope?: string
  projectName: string
  starterProjectName?: string
  messages: WebMessage[]
  sending: boolean
  inputDisabled: boolean
  onSend: (text: string, attachments: WebAttachment[], modelPreference?: string) => void
  onStop?: () => void
  workspaceSessionId?: string
  entryInteraction?: WebAgentInteraction
  workspaceActivity?: { proposal?: unknown; semanticDecision?: unknown; steps?: unknown; result?: unknown; evidenceRefs?: unknown }
  reasoning?: WebReasoning
  runId?: string
  status?: WebRunStatus
  resultCatalog?: WebResultCatalog
  onOpenTrainingPanel?: (tab: 'progress' | 'results') => void
  onApproved?: () => void
  attachments: WebAttachment[]
  onAttachmentsChange: (attachments: WebAttachment[]) => void
  onEnsureProject: (suggestedName: string) => Promise<string>
  onDatasetReady?: (datasets: WebDataset[]) => void | Promise<void>
  liveAssistantMessageId?: string
  onAssistantRendered?: (messageId: string) => void
  tokenUsage: WebTokenUsage
}

const formatTokenCount = (value: number): string =>
  value >= 1_000_000
    ? `${(value / 1_000_000).toFixed(value >= 10_000_000 ? 0 : 1)}M`
    : value >= 1_000
      ? `${(value / 1_000).toFixed(value >= 10_000 ? 0 : 1)}K`
      : String(value)

const messageTime = (value: string, locale: string): string => {
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? '' : new Intl.DateTimeFormat(locale, { hour: '2-digit', minute: '2-digit' }).format(date)
}

const artifactActivity = (message: WebMessage): WebAttachment[] | undefined => {
  if (message.messageKind !== 'activity.artifacts.viewed') return undefined
  try {
    const value = JSON.parse(message.content) as { attachments?: WebAttachment[] }
    return Array.isArray(value.attachments) ? value.attachments : []
  } catch {
    return []
  }
}

const messageAttachments = (message: WebMessage): WebAttachment[] | undefined => {
  if (message.messageKind !== 'conversation.attachment') return undefined
  try {
    const value = JSON.parse(message.content) as { attachments?: WebAttachment[] }
    return Array.isArray(value.attachments) ? value.attachments : []
  } catch {
    return []
  }
}

const hiddenActivityMarker = (message: WebMessage): boolean =>
  message.messageKind === 'activity.dataset.requested' ||
  message.messageKind === 'activity.dataset.request.cleared'

const LOCAL_DATASET_UPLOAD_INTERACTION: WebAgentInteraction = {
  source: 'fsm',
  state: 'AwaitDataset',
  status: 'waiting_human',
  reasoning: {
    goal: '接收用户本地数据集',
    observation: '用户希望上传数据集文件。',
    decision: '显示本地文件选择卡片。',
    nextStates: [],
    allowedTools: [],
    policyRefs: [],
  },
  card: {
    kind: 'dataset_upload',
    title: '上传本地数据集',
    description: '选择本地文件以继续。',
    actionRef: 'frontend.dataset.upload',
    requiresHumanAction: true,
  },
}

// Read historical operation receipts without coupling today's workbenches.
interface StoredManualOperation {
  id: string
  label: string
  detail: string
  startedAt: string
  state: 'running' | 'completed' | 'failed'
  completionDetail?: string
  error?: string
}

interface StoredToolActivity {
  toolId: string
  phases: Array<{ id: string; type: string; timestamp: string }>
  result?: unknown
}

interface StoredAgentProgress {
  phase: 'thinking' | 'tool'
  label: string
  status: 'ongoing'
  step: number
  toolId?: string
}

interface StoredInteractionSnapshot {
  resolution?: string
  feedback?: string
  error?: string
  runId: string
  interaction: WebAgentInteraction
}

const agentProgress = (message: WebMessage): StoredAgentProgress | undefined => {
  if (message.messageKind !== 'activity.agent.progress') return undefined
  try {
    const value = JSON.parse(message.content) as Partial<StoredAgentProgress>
    if ((value.phase !== 'thinking' && value.phase !== 'tool') || typeof value.label !== 'string') return undefined
    return {
      phase: value.phase,
      label: value.label,
      status: 'ongoing',
      step: typeof value.step === 'number' ? value.step : 1,
      ...(typeof value.toolId === 'string' ? { toolId: value.toolId } : {}),
    }
  } catch {
    return undefined
  }
}

const storedToolActivity = (message: WebMessage): StoredToolActivity | undefined => {
  if (message.messageKind !== 'activity.tool.trace') return undefined
  try {
    const value = JSON.parse(message.content) as StoredToolActivity
    return typeof value.toolId === 'string' && Array.isArray(value.phases) ? value : undefined
  } catch {
    return undefined
  }
}

const storedManualOperation = (message: WebMessage): StoredManualOperation | undefined => {
  if (message.messageKind !== 'activity.manual-operation') return undefined
  try {
    const value = JSON.parse(message.content) as Partial<StoredManualOperation>
    if (typeof value.id !== 'string' || typeof value.label !== 'string' || typeof value.detail !== 'string' ||
      typeof value.startedAt !== 'string' || (value.state !== 'running' && value.state !== 'completed' && value.state !== 'failed')) return undefined
    return value as StoredManualOperation
  } catch {
    return undefined
  }
}

const storedInteractionSnapshot = (message: WebMessage): StoredInteractionSnapshot | undefined => {
  if (message.messageKind !== 'activity.interaction.snapshot') return undefined
  try {
    const value = JSON.parse(message.content) as Partial<StoredInteractionSnapshot>
    return typeof value.runId === 'string' && value.interaction?.card != null
      ? value as StoredInteractionSnapshot
      : undefined
  } catch {
    return undefined
  }
}

const StarterDialogue = ({ children, projectName }: { children: ReactNode; projectName?: string }): React.ReactElement => {
  const [title, setTitle] = useState('')
  const [description, setDescription] = useState('')
  const [phase, setPhase] = useState<'title' | 'description' | 'complete'>('title')
  const normalizedProjectName = projectName?.trim()
  const projectTitle = normalizedProjectName ? STARTER_TITLE : undefined
  const renderedTitle = projectTitle ?? title
  const renderedDescription = projectTitle ? STARTER_DESCRIPTION : description
  const renderedPhase = projectTitle ? 'complete' : phase

  useEffect(() => {
    if (projectTitle) return

    setTitle('')
    setDescription('')
    setPhase('title')
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    if (reducedMotion) {
      setTitle(STARTER_TITLE)
      setDescription(STARTER_DESCRIPTION)
      setPhase('complete')
      return
    }

    const titleCharacters = Array.from(STARTER_TITLE)
    const descriptionCharacters = Array.from(STARTER_DESCRIPTION)
    const timers: number[] = []
    let titleIndex = 0
    let descriptionIndex = 0

    const schedule = (callback: () => void, delay: number): void => {
      timers.push(window.setTimeout(callback, delay))
    }
    const typeDescription = (): void => {
      descriptionIndex += 1
      setDescription(descriptionCharacters.slice(0, descriptionIndex).join(''))
      if (descriptionIndex < descriptionCharacters.length) {
        const character = descriptionCharacters[descriptionIndex - 1]
        schedule(typeDescription, character === '，' ? 105 : 30)
      } else {
        schedule(() => setPhase('complete'), 180)
      }
    }
    const typeTitle = (): void => {
      titleIndex += 1
      setTitle(titleCharacters.slice(0, titleIndex).join(''))
      if (titleIndex < titleCharacters.length) {
        const character = titleCharacters[titleIndex - 1]
        schedule(typeTitle, character === '，' ? 165 : 72)
      } else {
        schedule(() => {
          setPhase('description')
          typeDescription()
        }, 210)
      }
    }

    schedule(typeTitle, 460)
    return () => timers.forEach((timer) => window.clearTimeout(timer))
  }, [projectTitle])

  return (
    <>
      <div className={`${css.starterDialogue} ${renderedPhase === 'complete' ? css.starterDialogueComplete : ''}`}>
        <div className={css.starterScientist} aria-hidden="true">
          <img src="/theta-assets/brand/theta-scientist-avatar-final.png" alt="" draggable={false} />
        </div>
        <div className={css.starterSpeech}>
          <h1 aria-label={projectTitle ?? STARTER_TITLE}>
            <span aria-hidden="true">{renderedTitle}</span>
            {renderedPhase === 'title' && <i className={css.starterTypingCursor} aria-hidden="true" />}
          </h1>
          <p aria-label={STARTER_DESCRIPTION}>
            <span aria-hidden="true">{renderedDescription}</span>
            {renderedPhase === 'description' && <i className={css.starterTypingCursor} aria-hidden="true" />}
          </p>
        </div>
      </div>
      <div className={`${css.startCardStage} ${renderedPhase === 'complete' ? css.startCardStageReady : ''}`}>
        {children}
      </div>
    </>
  )
}

export const ConversationPane = ({
  analysisMode = 'topic',
  onAnalysisModeChange,
  analysisModeDisabled = false,
  compact = false,
  projectStorageScope,
  projectName,
  starterProjectName,
  messages,
  sending,
  inputDisabled,
  onSend,
  onStop,
  workspaceSessionId,
  entryInteraction,
  workspaceActivity,
  reasoning,
  runId,
  status,
  resultCatalog,
  onOpenTrainingPanel,
  datasetSizeBytes = 0,
  onApproved,
  attachments,
  onAttachmentsChange,
  onEnsureProject,
  onDatasetReady,
  liveAssistantMessageId,
  onAssistantRendered,
  tokenUsage,
}: ConversationPaneProps): React.ReactElement => {
  const { locale, t } = usePreferences()
  const { settings } = useInferenceSettings()
  const missingApi = !!settings && !settings.llm.apiKeyConfigured
  const [uploadSize, setUploadSize] = useProjectDraft<number>(`upload-size:${projectStorageScope ?? projectName}:${runId ?? workspaceSessionId ?? 'draft'}`, 0)
  const draftScope = `${projectStorageScope ?? projectName}:${runId ?? workspaceSessionId ?? 'draft'}`
  const [initialDraft] = useState(readLandingPrefill)
  const [draft, setDraft] = useProjectDraft(`conversation:${draftScope}:input`, initialDraft)
  const previousDraftScope = useRef(draftScope)
  // 从手动结果页"引用"回来的图/表：作为引用随下一条消息一起发出。
  const [pendingCitations, setPendingCitations] = useProjectDraft<Array<{ chartName: string; chartPath?: string; figure?: string; jobId?: string }>>(`conversation:${draftScope}:citations`, [])
  // Old versions could write a result quote into an unbound new-chat draft.
  // Keep the saved draft intact, but never display/send it without its source run.
  const activeCitations = runId ? pendingCitations : []
  // A chip is a one-message reference, NOT the project's dataset catalog.
  const [dismissedDatasets, setDismissedDatasets] = useProjectDraft<string[]>(`project:${projectStorageScope ?? projectName}:dismissed-dataset-references`, [])
  const composerAttachments = attachments.filter(item => item.kind !== 'dataset' || (!dismissedDatasets.includes(item.id) && !status?.datasetRefs?.includes(item.id)))
  const dismissDatasetReferences = () => setDismissedDatasets([...new Set([...dismissedDatasets, ...attachments.filter(item => item.kind === 'dataset').map(item => item.id)])])
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState<string>()
  const [addMenuOpen, setAddMenuOpen] = useState(false)
  const [analysisMenuOpen, setAnalysisMenuOpen] = useState(false)
  const threadRef = useRef<HTMLDivElement>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const folderInputRef = useRef<HTMLInputElement>(null)
  const followTail = useRef(true)
  const activeInteraction = runId ? status?.interaction : entryInteraction
  const decisionRunning = status?.status === 'running' && !sending
  const generationRunning = sending || decisionRunning
  const controlsDisabled = inputDisabled || sending || decisionRunning || uploading
  useEffect(() => {
    folderInputRef.current?.setAttribute('webkitdirectory', '')
    folderInputRef.current?.setAttribute('directory', '')
  }, [])

  useEffect(() => {
    const thread = threadRef.current
    if (messages.length === 0 && !sending) {
      const frame = window.requestAnimationFrame(() => {
        followTail.current = false
        threadRef.current?.scrollTo({ top: 0, behavior: 'auto' })
      })
      return () => window.cancelAnimationFrame(frame)
    }
    if (!followTail.current) return
    const frame = window.requestAnimationFrame(() => {
      const currentThread = threadRef.current
      if (currentThread == null) return
      currentThread.scrollTo({
        top: currentThread.scrollHeight,
        behavior: sending ? 'smooth' : 'auto',
      })
    })
    return () => window.cancelAnimationFrame(frame)
  }, [draftScope, messages.length, sending, status?.pendingActionRef, reasoning?.toolCalls.length, reasoning?.reasoningEvents.length])

  useEffect(() => {
    if (previousDraftScope.current === draftScope) return
    previousDraftScope.current = draftScope
    followTail.current = true
  }, [draftScope])

  useEffect(() => {
    const handleQuote = (event: Event): void => {
      const detail = (event as CustomEvent<{ chartName?: string; figure?: string; jobId?: string }>).detail
      if (!runId || (!detail?.chartName && !detail?.figure)) return
      setPendingCitations((current) => {
        const next = detail.figure ? { chartName: detail.chartName ?? detail.figure, figure: detail.figure, jobId: detail.jobId } : { chartName: detail.chartName! }
        return [...current.filter((item) => item.figure !== next.figure || item.jobId !== next.jobId), next].slice(-8)
      })
    }
    const handleReportRequest = (event: Event): void => {
      const detail = (event as CustomEvent<{ jobId?: string; modelId?: string }>).detail
      if (!runId || !detail?.jobId) return
      const label = (detail.modelId ?? '').toUpperCase()
      setPendingCitations([{ chartName: label ? `${label} 训练结果` : '选中训练结果', jobId: detail.jobId }])
      setDraft(`请整理${label ? ` ${label} ` : '选中任务'}的结果：调用原生可视化，交付图表与数据表并解释指标含义。先展示结果整理确认卡。`)
      textareaRef.current?.focus()
    }
    window.addEventListener('theta:chart-quote', handleQuote)
    window.addEventListener('theta:request-report', handleReportRequest)
    return () => {
      window.removeEventListener('theta:chart-quote', handleQuote)
      window.removeEventListener('theta:request-report', handleReportRequest)
    }
  }, [draftScope, runId])

  useEffect(() => {
    const raw = sessionStorage.getItem('theta.workspace.pending-citations.v1')
    if (!raw) return
    try {
      const parsed = JSON.parse(raw) as { runId?: string; citations?: Array<{ chartName: string; chartPath: string }>; analyses?: Array<{ chartName: string; analysis: string }> }
      // A mounted empty/new conversation must not consume another run's handoff.
      if (!runId || parsed.runId !== runId) return
      sessionStorage.removeItem('theta.workspace.pending-citations.v1')
      const citations = (parsed?.citations ?? []).slice(0, 8)
      if (!citations.length) return
      setPendingCitations(citations)
      const names = citations.map((item) => item.chartName).join('、')
      const analyses = (parsed?.analyses ?? []).filter((item) => item.analysis)
      const context = analyses.length
        ? '\n\n这些图表此前已经生成过一版解读，可以作为起点继续深化：\n' + analyses.map((item) => `【${item.chartName}】${item.analysis}`).join('\n\n')
        : ''
      setDraft(`请基于本次结果中已交付的图表继续分析：${names}。${context}\n\n请结合原始绘图数据与主题证据，说明这些结果能支持什么结论、不能支持什么结论；如果需要新的图表或数据表，请直接生成。`)
    } catch { /* 忽略损坏的交接数据 */ }
  }, [runId])

  useEffect(() => {
    const thread = threadRef.current
    if (thread == null) return
    let frame: number | undefined
    const scrollTail = (): void => {
      if (!followTail.current) return
      window.cancelAnimationFrame(frame ?? 0)
      frame = window.requestAnimationFrame(() => {
        thread.scrollTop = thread.scrollHeight
      })
    }
    const resizeObserver = new ResizeObserver(scrollTail)
    const observeChildren = (): void => {
      for (const child of thread.children) resizeObserver.observe(child)
    }
    const mutationObserver = new MutationObserver(() => {
      observeChildren()
      scrollTail()
    })
    observeChildren()
    mutationObserver.observe(thread, { childList: true, subtree: true })
    return () => {
      window.cancelAnimationFrame(frame ?? 0)
      mutationObserver.disconnect()
      resizeObserver.disconnect()
    }
  }, [])

  const refreshAfterDecision = (): void => {
    followTail.current = true
    threadRef.current?.scrollTo({ top: threadRef.current.scrollHeight, behavior: 'auto' })
    onApproved?.()
  }

  const submitText = (value = draft): void => {
    if (controlsDisabled) return
    if (missingApi) { openSetup('inference'); return }
    const text = value.trim()
    if (!text) return
    // 引用只作为模型可见的上下文；界面不展示这段文字，是否重绘由 Agent 依用户问题决定。
    const quoteBlock = activeCitations.length
      ? '\n\n[[cite]]' + JSON.stringify(activeCitations.map((item) => ({
        chartName: item.chartName,
        ...(item.figure ? { figure: item.figure } : {}),
        ...(item.jobId ? { jobId: item.jobId } : {}),
      }))) + '[[/cite]]'
      : ''
    const outgoing = text + quoteBlock
    setPendingCitations([])
    setDraft('')
    if (textareaRef.current != null) {
      textareaRef.current.value = ''
      textareaRef.current.style.height = 'auto'
    }
    followTail.current = true
    // Even a dismissed chip must bind the dataset on the first turn. Subsequent
    // messages use the durable session binding, without repeating the reference.
    onSend(outgoing, attachments.filter(item => item.kind !== 'dataset' || composerAttachments.includes(item) || !status?.datasetRefs?.includes(item.id)))
    dismissDatasetReferences()
    onAttachmentsChange(attachments.filter((attachment) => attachment.kind === 'dataset'))
  }

  const addFiles = async (files: FileList | File[]): Promise<void> => {
    if (files.length === 0 || uploading || controlsDisabled) return
    const file = Array.from(files)[0]
    const problem = isDocumentCollection(Array.from(files)) ? documentCollectionError(Array.from(files), locale) : datasetFilesError(Array.from(files), locale)
    if (problem) { setUploadError(problem); return }
    setUploadSize(file.size)
    setUploading(true)
    setUploadError(undefined)
    try {
      const projectId = await onEnsureProject(file.name.replace(/\.[^.]+$/u, '') || '数据分析项目')
      const dataset = await uploadDatasetFiles(projectId, Array.from(files), locale)
      await datasetsReady([dataset])
      textareaRef.current?.focus()
    } catch (cause) {
      const detail = cause instanceof Error ? cause.message : String(cause)
      setUploadError(/Unsupported dataset suffix/iu.test(detail)
        ? (locale === 'zh-CN'
            ? '该文件不是受支持的数据集格式。请上传 CSV、TSV、TXT、Markdown、JSON/JSONL、Excel、Parquet、PDF 或 DOCX 文件。'
            : 'This file is not a supported dataset format. Upload CSV, TSV, text, Markdown, JSON/JSONL, Excel, Parquet, PDF, or DOCX.')
        : detail)
    } finally {
      setUploading(false)
    }
  }

  const datasetsReady = async (datasets: WebDataset[]): Promise<void> => {
    const nextAttachments = [
      ...attachments.filter((attachment) => attachment.kind !== 'dataset'),
      ...datasets.slice(0, 1).map((dataset): WebAttachment => ({ kind: 'dataset', id: dataset.datasetRef, label: dataset.name })),
    ]
    if (onDatasetReady) await onDatasetReady(datasets.slice(0, 1))
    else onAttachmentsChange(nextAttachments)
  }

  const { dragging: draggingFiles, dropProps } = useFileDrop(files => { void addFiles(files) }, controlsDisabled, setUploadError)

  const dropAttachment = (event: React.DragEvent): void => {
    if (Array.from(event.dataTransfer.types).includes('Files')) { dropProps.onDrop(event); return }
    event.preventDefault()
    if (controlsDisabled) return
    const encoded = event.dataTransfer.getData('application/x-theta-artifact')
    if (!encoded && event.dataTransfer.files.length > 0) {
      void addFiles(event.dataTransfer.files)
      return
    }
    if (!encoded) return
    try {
      const attachment = JSON.parse(encoded) as WebAttachment
      if (!attachment.id || !attachment.label) return
      onAttachmentsChange([...attachments.filter((item) => item.id !== attachment.id), attachment].slice(-12))
      textareaRef.current?.focus()
    } catch {
      // Ignore unrelated drag payloads.
    }
  }

  const orderedMessages = useMemo(
    () => messages.map((message, index) => ({ message, index })).sort((left, right) => {
      const leftTime = Date.parse(left.message.createdAt)
      const rightTime = Date.parse(right.message.createdAt)
      if (Number.isFinite(leftTime) && Number.isFinite(rightTime) && leftTime !== rightTime) return leftTime - rightTime
      const leftSequence = Number.isFinite(left.message.sequenceNumber) ? left.message.sequenceNumber : Number.MAX_SAFE_INTEGER
      const rightSequence = Number.isFinite(right.message.sequenceNumber) ? right.message.sequenceNumber : Number.MAX_SAFE_INTEGER
      if (leftSequence !== rightSequence) return leftSequence - rightSequence
      if (left.message.role !== right.message.role) return left.message.role === 'user' ? -1 : 1
      return left.index - right.index
    }).map(({ message }) => message),
    [messages],
  )
  const cardSummaries = new Set(orderedMessages.map(storedInteractionSnapshot).map(snapshot => snapshot?.interaction.card?.description?.replace(/\s/gu, '')).filter(Boolean))
  const showStarter = !compact && orderedMessages.length === 0 && !sending
  const starterCards = [
    { kind: 'code', title: '了解 THETA Agent', description: '认识完整研究、训练与结果验证流程', prompt: '请介绍 THETA Agent 的能力，并说明一次完整研究任务会如何推进。' },
    { kind: 'build', title: '从研究想法开始', description: '先讨论研究方向，再决定需要什么数据', prompt: '我有一个研究想法，想先和你讨论研究问题，暂时不上传数据。' },
    { kind: 'review', title: '从已有数据开始', description: analysisMode === 'free' ? '上传表格或文本，按研究问题选择分析方法' : '上传语料，让 Agent 自主理解数据并推进研究', prompt: analysisMode === 'free' ? '我有一份数据，请从理解变量和研究设计开始引导我分析，不必先做主题建模。' : '我已经有一份文本数据，请从数据理解开始引导我完成研究。' },
    { kind: 'measure', title: '帮我找到研究起点', description: '还没有明确方向，由 Agent 提供可行的开始方式', prompt: analysisMode === 'free' ? '我想做实证分析或预测，还没有明确问题，请帮我找到可验证的研究起点。' : '我想进行主题研究，但还不确定研究问题、数据和模型，请帮我找到合适的起点。' },
  ] as const
  const hasDatasetCardMessage = orderedMessages.some((message) => message.messageKind === 'conversation.dataset-upload-card')
  const agentHeaderIds = useMemo(() => {
    const ids = new Set<string>()
    let inAgentTurn = false
    for (const message of orderedMessages) {
      if (hiddenActivityMarker(message) || agentProgress(message) || storedToolActivity(message) || storedInteractionSnapshot(message)) continue
      if (message.role === 'user') {
        inAgentTurn = false
        continue
      }
      if (!inAgentTurn) ids.add(message.messageId)
      inAgentTurn = true
    }
    return ids
  }, [orderedMessages])
  return (
    <div
      className={`${css.conversation} ${compact ? css.conversationCompact : ''} ${draggingFiles ? css.conversationDragging : ''}`}
      {...dropProps}
      onDragOver={event => { dropProps.onDragOver(event); if (event.dataTransfer.types.includes('application/x-theta-artifact')) event.preventDefault() }}
      onDrop={dropAttachment}
    >
      {!showStarter && !compact && onOpenTrainingPanel && (
        <TrainingStatusBar status={status} catalog={resultCatalog} onOpen={onOpenTrainingPanel} />
      )}
      <div
        ref={threadRef}
        className={`${css.thread} ${showStarter ? css.startThread : ''}`}
        aria-live="polite"
        onScroll={() => {
          const element = threadRef.current
          if (element == null) return
          followTail.current = element.scrollHeight - element.scrollTop - element.clientHeight < 120
        }}
      >
        {showStarter && (
          <div className={css.startSurface}>
            <StarterDialogue projectName={starterProjectName}>
              <div className={css.promptGrid}>
                {starterCards.map((card) => (
                  <button key={card.kind} type="button" onClick={() => { setDraft(card.prompt); textareaRef.current?.focus() }}>
                    <span className={css.featureCopy}>
                      <strong>{card.title}</strong>
                      <small>{card.description}</small>
                    </span>
                    <StarterVisual kind={card.kind} />
                  </button>
                ))}
              </div>
            </StarterDialogue>
          </div>
        )}
        {compact && orderedMessages.length === 0 && !sending && (
          <div className={css.compactConversationEmpty}>
            <ThetaCubeMark size={36} />
            <strong>THETA 项目指引</strong>
            <p>这里保留当前项目的完整对话。你可以说明目标，THETA 捕捉到的明确配置会同步到左侧手动工作台。</p>
          </div>
        )}

        {orderedMessages.map((message, index) => {
          // The decision row already records a plain acknowledgement. Keep substantive feedback as a user message.
          const precedingDecision = index > 0 ? storedInteractionSnapshot(orderedMessages[index - 1]) : undefined
          if (message.role === 'user' && (
            (message.content.trim() === '确认' && precedingDecision?.resolution === 'approved') ||
            (message.content.trim() === '拒绝本次操作，暂不执行。' && precedingDecision?.resolution === 'rejected')
          )) return null
          if (hiddenActivityMarker(message)) return null
          if (message.role === 'assistant' && message.messageKind === 'conversation.message' && cardSummaries.has(message.content.replace(/\s/gu, ''))) return null
          const nextInteraction = index + 1 < orderedMessages.length ? storedInteractionSnapshot(orderedMessages[index + 1]) : undefined
          if (message.role === 'assistant' && nextInteraction && /^(?:I'll start|I will start|Let me start|我(?:会|先)|让我先)/iu.test(message.content.trim())) return null
          const showAgentHeader = agentHeaderIds.has(message.messageId)
          const progress = agentProgress(message)
          if (progress) return null
          const manualOperation = storedManualOperation(message)
          if (manualOperation) {
            const running = manualOperation.state === 'running'
            const failed = manualOperation.state === 'failed'
            const label = running
              ? manualOperation.label
              : failed
                ? `操作失败：${manualOperation.label.replace(/^正在/u, '')}`
                : `已完成：${manualOperation.label.replace(/^正在/u, '')}`
            const detail = failed
              ? manualOperation.error ?? manualOperation.detail
              : running
                ? manualOperation.detail
                : manualOperation.completionDetail ?? '该操作已经完成，相关状态已同步到当前项目。'
            return (
              <StandardAssistantTurn key={message.messageId} createdAt={message.createdAt} locale={locale} copyText={!running ? `${label}\n${detail}` : undefined}>
                <div className={css.assistantOperationCopy}>
                  {running ? <span className={css.activitySpinner} /> : <StateDot size={8} state={failed ? 'error' : 'done'} />}
                  <span className={css.assistantOperationText}>
                    <strong>{label}</strong>
                    <small>{detail}</small>
                  </span>
                  {running && <span className={css.operationProgressTrack} aria-hidden="true"><i /></span>}
                </div>
              </StandardAssistantTurn>
            )
          }
          const interactionSnapshot = storedInteractionSnapshot(message)
          if (interactionSnapshot) {
            const snapshotCard = interactionSnapshot.interaction.card
            const currentCard = activeInteraction?.card
            const current = interactionSnapshot.runId === runId && currentCard != null &&
              currentCard.kind === snapshotCard?.kind && currentCard.actionRef === snapshotCard?.actionRef &&
              currentCard.contentHash === snapshotCard?.contentHash &&
              (!interactionSnapshot.resolution || interactionSnapshot.resolution === 'pending')
            if (current && onApproved && runId) return (
              <StandardAssistantTurn key={message.messageId} createdAt={message.createdAt} locale={locale}>
                <ApprovalPanel appearance="conversation" runId={runId} interaction={generationRunning ? { ...activeInteraction!, status: 'running' } : activeInteraction!} decisionError={interactionSnapshot.error} reasoning={reasoning} onApproved={refreshAfterDecision} />
              </StandardAssistantTurn>
            )
            const resolution = interactionSnapshot.resolution ?? 'resolved'
            const ResolutionIcon = resolution === 'approved' ? Check : resolution === 'rejected' ? X : resolution === 'revised' ? Pencil : resolution === 'failed' ? AlertCircle : Clock3
            return (
              <details key={message.messageId} className={approvalCss.history} data-state={resolution} open={resolution === 'failed'}>
                <summary>
                  <ResolutionIcon size={15} aria-hidden="true" />
                  <span className={approvalCss.historyTitle}>{snapshotCard?.title?.replace(/^确认/u, '') ?? '操作确认'}</span>
                  <span className={approvalCss.historyBadge}>{({ pending: '等待更新', submitting: '正在提交', approved: '已确认', rejected: '已拒绝', revised: '已提交修改', superseded: '已替换', expired: '已过期', failed: '执行失败' } as Record<string, string>)[resolution] ?? '已处理'}</span>
                  <time dateTime={message.createdAt}>{messageTime(message.createdAt, locale)}</time>
                  <ChevronRight size={13} className={approvalCss.chevron} aria-hidden="true" />
                </summary>
                <div className={approvalCss.historyBody}>
                  {resolution === 'failed' && <p role="alert">{interactionSnapshot.error || '本次执行未完成。'} 可说明修改要求，让 Agent 准备新方案；不会自动重跑。</p>}
                  {resolution === 'failed' ? <details className={approvalCss.details}>
                    <summary><ChevronRight size={14} aria-hidden="true" />查看失败方案</summary>
                    <p>{snapshotCard?.description}</p>
                  </details> : <p>{snapshotCard?.description}</p>}
                  {resolution === 'approved' && <p>{interactionSnapshot.interaction.card?.kind === 'embedding_choice' ? '已保存嵌入方式；选择本身不授权训练，请查看后续训练确认卡。' : '已确认此版本方案；执行结果见后续消息。'}</p>}
                  {resolution === 'rejected' && <p>本次操作未执行，你可以继续对话。</p>}
                  {resolution === 'revised' && <p>原方案未执行，更新后的方案需要重新确认。</p>}
                  {resolution === 'expired' && <p>确认已过期，可告诉 Agent 重新生成方案。</p>}
                  {interactionSnapshot.feedback && <p>你的反馈：{interactionSnapshot.feedback}</p>}
                  {interactionSnapshot.error && resolution !== 'failed' && <p role="alert">{interactionSnapshot.error}</p>}
                </div>
              </details>
            )
          }

          if (message.messageKind === 'conversation.dataset-upload-card') {
            return (
              <div key={message.messageId} className={`${css.interactionReply} ${css.datasetChatTurn}`}>
                <AgentTurnHeader createdAt={message.createdAt} locale={locale} />
                <DatasetIntakeCard
                  disabled={controlsDisabled}
                  key={`${message.messageId}:${projectStorageScope ?? projectName}`}
                  interaction={LOCAL_DATASET_UPLOAD_INTERACTION}
                  storageScope={projectStorageScope ?? projectName}
                  legacyStorageScope={projectStorageScope == null ? undefined : projectName}
                  knownDatasetRefs={attachments.filter((attachment) => attachment.kind === 'dataset').map((attachment) => attachment.id)}
                  onEnsureProject={onEnsureProject}
                  onDatasetsReady={datasetsReady}
                />
              </div>
            )
          }
          const uploaded = messageAttachments(message)
          if (uploaded) {
            return (
              <article key={message.messageId} className={`${css.message} ${css.messageUser}`}>
                <div className={css.messageBody}>
                  <div className={css.uploadedMessageList}>
                    {uploaded.map((attachment) => (
                      <div key={`${attachment.kind}-${attachment.id}`} className={css.uploadedMessageCard}>
                        <StateDot size={7} state="done" />
                        <span>
                          <strong>{attachment.label}</strong>
                          <small>{locale === 'zh-CN' ? '已上传到受管数据工作区' : 'Uploaded to the managed data workspace'}</small>
                        </span>
                        <code>{attachment.id}</code>
                      </div>
                    ))}
                  </div>
                </div>
              </article>
            )
          }
          const viewedArtifacts = artifactActivity(message)
          if (viewedArtifacts) {
            const visualizations = viewedArtifacts.filter((item) => item.kind === 'visualization').length
            return (
              <Fragment key={message.messageId}>
                {showAgentHeader && <AgentTurnHeader createdAt={message.createdAt} locale={locale} />}
                <div className={css.artifactActivity}>
                  <strong>{locale === 'zh-CN'
                    ? `已查看 ${viewedArtifacts.length} 个研究产物${visualizations > 0 ? `（${visualizations} 个可视化）` : ''}`
                    : `Viewed ${viewedArtifacts.length} research artifact${viewedArtifacts.length === 1 ? '' : 's'}${visualizations > 0 ? ` (${visualizations} visualization${visualizations === 1 ? '' : 's'})` : ''}`}</strong>
                  <small>{viewedArtifacts.map((item) => item.label).join(' · ')}</small>
                </div>
              </Fragment>
            )
          }
          const toolActivity = storedToolActivity(message)
          if (toolActivity) {
            return null
          }
          const human = message.role === 'user'
          const displayText = message.messageKind === 'conversation.error' ? errorGuidance(message.content, locale).message : conversationDisplayText(message.content)
          return (
            <Fragment key={message.messageId}>
              {showAgentHeader && <AgentTurnHeader createdAt={message.createdAt} locale={locale} />}
            <article className={`${css.message} ${human ? css.messageUser : css.messageAssistant}`}>
              <div className={css.messageBody}>
                {human && <div className={css.messageMeta}>
                  <strong>{human ? t('you') : t('agent')}</strong>
                  <time dateTime={message.createdAt}>{messageTime(message.createdAt, locale)}</time>
                </div>}
                <div className={`${css.bubble} ${human ? css.bubbleUser : css.bubbleAssistant}`}>
                  <CollapsibleMessage text={displayText} enabled={!human && displayText.length > 400} zh={locale === 'zh-CN'}>
                  <TypewriterMarkdown
                    text={displayText}
                    active={message.messageId === liveAssistantMessageId && settings?.llm.typewriter === true}
                    speedMs={settings?.llm.typewriterSpeedMs ?? 18}
                    onComplete={() => onAssistantRendered?.(message.messageId)}
                  />
                  </CollapsibleMessage>
                </div>
                {!human && (
                  <div className={css.messageActions}>
                    <time dateTime={message.createdAt}>{messageTime(message.createdAt, locale)}</time>
                    <CopyMessageButton text={displayText} />
                  </div>
                )}
              </div>
              {human && <span className={css.userMessageAvatar}>U</span>}
            </article>
            </Fragment>
          )
        })}

        {sending && (
          <div className={css.thinkingStatus} role="status" aria-live="polite">
            <span className={css.thinkingOrbit} aria-hidden="true"><i /><i /><i /></span>
            <span className={css.thinkingStatusCopy}>
              <strong>{locale === 'zh-CN' ? '猫咪科学家正在思考' : 'Cat scientist is thinking'}</strong>
              <small>{locale === 'zh-CN' ? '正在理解你的需求，分析数据特征并规划研究方案…' : 'Understanding your request, analyzing the data, and planning the study…'}</small>
            </span>
          </div>
        )}

        {!showStarter && activeInteraction?.card?.kind === 'dataset_upload' && !hasDatasetCardMessage && (
          <div className={`${css.interactionReply} ${css.datasetChatTurn}`}>
            <AgentTurnHeader createdAt={new Date().toISOString()} locale={locale} />
            <DatasetIntakeCard
                  disabled={controlsDisabled}
              interaction={activeInteraction}
              storageScope={projectStorageScope ?? projectName}
              legacyStorageScope={projectStorageScope == null ? undefined : projectName}
              knownDatasetRefs={attachments.filter((attachment) => attachment.kind === 'dataset').map((attachment) => attachment.id)}
              onEnsureProject={onEnsureProject}
              onDatasetsReady={datasetsReady}
            />
          </div>
        )}
        {reasoning?.toolCalls.length ? <details className={approvalCss.executionTrace}><summary>Agent 执行记录</summary><AgentActivityTrace reasoning={reasoning} working={sending} /></details> : null}
        {decisionRunning && <p className={approvalCss.runStatus} role="status"><Clock3 size={14} aria-hidden="true" />Agent 正在处理，进度会自动同步…</p>}
        <div ref={bottomRef} />
      </div>

      <div
        className={`${css.composerWrap} ${showStarter ? css.startComposerWrap : ''} ${draggingFiles ? css.composerWrapDragging : ''}`}
      >
        {draggingFiles && <p role="status">{locale === 'zh-CN' ? '松开即可上传文件或文件夹；更换数据将在当前项目中开始新对话，历史结果保留。' : 'Drop to upload and replace the current dataset. Existing results are kept.'}</p>}
        {uploading && <p role="status">{locale === 'zh-CN' ? '正在上传文件。上传后不会自动发送，你可以继续补充要求。' : 'Uploading your file. It will not be sent automatically, so you can keep adding instructions.'}</p>}
        <div className={`${css.composer} ${generationRunning ? css.composerBusy : ''}`} aria-busy={generationRunning || uploading}>
          <input
            ref={fileInputRef}
            className={css.visuallyHidden}
            type="file"
            disabled={controlsDisabled}
            accept={DATASET_ACCEPT}
            onChange={(event) => { if (event.target.files) void addFiles(Array.from(event.target.files).filter(file => !file.webkitRelativePath.split("/").some(part => part.startsWith(".")))); event.target.value = '' }}
          />
          <input
            ref={folderInputRef}
            className={css.visuallyHidden}
            type="file"
            disabled={controlsDisabled}
            accept={DATASET_ACCEPT}
            onChange={(event) => { if (event.target.files) void addFiles(Array.from(event.target.files).filter(file => !file.webkitRelativePath.split("/").some(part => part.startsWith(".")))); event.target.value = '' }}
          />
          <div className={css.composerContextRow}>
            <div className={css.addMenu}>
              <button className={css.addButton} type="button" disabled={uploading || controlsDisabled} onClick={() => setAddMenuOpen((current) => !current)} aria-expanded={addMenuOpen} title={locale === 'zh-CN' ? '添加文件或文件夹' : 'Add files or a folder'}>
                {uploading ? <span className={css.activitySpinner} /> : <IconPlusOutline16 />}
              </button>
              {addMenuOpen && (
                <div className={css.addMenuPopover}>
                  <button type="button" onClick={() => { setAddMenuOpen(false); fileInputRef.current?.click() }}>{locale === 'zh-CN' ? attachments.some(item => item.kind === 'dataset') ? '重新上传 / 更换数据' : '选择文件' : attachments.some(item => item.kind === 'dataset') ? 'Replace dataset' : 'Choose file'}</button>
                  <button type="button" onClick={() => { setAddMenuOpen(false); folderInputRef.current?.click() }}>{locale === 'zh-CN' ? '选择文件夹' : 'Choose folder'}</button>
                </div>
              )}
            </div>
            {onAnalysisModeChange && <Menu
              open={analysisMenuOpen}
              portal compact side="top"
              selectedId={analysisMode}
              items={analysisModeEntries(locale)}
              onClose={() => setAnalysisMenuOpen(false)}
              onSelect={(id) => {
                if (id !== 'topic' && id !== 'free') return
                void onAnalysisModeChange(id)
                setAnalysisMenuOpen(false)
              }}
              anchor={<button type="button" className={css.composerMode} aria-label={locale === 'zh-CN' ? '分析方式' : 'Analysis mode'} aria-haspopup="menu" aria-expanded={analysisMenuOpen} disabled={analysisModeDisabled || generationRunning || uploading || status?.status === 'waiting_human'} onClick={() => setAnalysisMenuOpen((current) => !current)}>
                <span>{analysisMode === 'topic' ? (locale === 'zh-CN' ? '主题分析' : 'Topic analysis') : (locale === 'zh-CN' ? '自由分析' : 'Free analysis')}</span><IconChevronDownOutline14 />
              </button>}
            />}
            <InferenceSelector disabled={controlsDisabled} />
            {composerAttachments.length > 0 && <div className={css.composerAttachments}>{composerAttachments.map((attachment) => (
              <button key={`${attachment.kind}-${attachment.id}`} type="button" disabled={controlsDisabled} title="移除本次引用，数据仍保存在当前项目中" onClick={() => attachment.kind === 'dataset' ? setDismissedDatasets([...new Set([...dismissedDatasets, attachment.id])]) : onAttachmentsChange(attachments.filter((item) => item !== attachment))}>
                {attachment.kind === 'dataset' ? <DatasetFileIcon filename={attachment.label} size={14} /> : <span className={css.attachmentFileIcon} aria-hidden="true">⌑</span>}{attachment.label}<IconCloseFill14 size={12} />
              </button>
            ))}</div>}
          </div>
          {activeCitations.length > 0 && (
            <div className={css.composerCitations} aria-label={locale === 'zh-CN' ? '引用中的结果' : 'Quoted results'}>
              {activeCitations.map((citation) => (
                <button key={(citation.jobId ?? '') + (citation.figure ?? citation.chartName)} type="button" disabled={controlsDisabled} onClick={() => setPendingCitations((current) => current.filter((item) => item !== citation))} title={locale === 'zh-CN' ? '移除这条引用' : 'Remove this reference'}>
                  <span>{locale === 'zh-CN' ? '引用 · ' : 'Quoted · '}{citation.chartName}</span><IconCloseFill14 size={12} />
                </button>
              ))}
            </div>
          )}
          <textarea
            ref={textareaRef}
            id="theta-composer"
            className={css.composerInput}
            rows={1}
            disabled={inputDisabled}
            aria-label={locale === 'zh-CN' ? '给 THETA Agent 发送消息' : 'Message THETA Agent'}
            placeholder={locale === 'zh-CN' ? (attachments.length ? '你还可以补充更多要求…' : '输入你的问题，Enter 发送，Shift+Enter 换行') : (attachments.length ? 'You can add more requirements…' : 'Ask a question; Enter to send, Shift+Enter for a new line')}
            value={draft}
            onChange={(event) => {
              setDraft(event.target.value)
              event.target.style.height = 'auto'
              event.target.style.height = `${Math.min(event.target.scrollHeight, 180)}px`
            }}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing && event.keyCode !== 229) { event.preventDefault(); submitText() }
            }}
          />
          <div className={css.composerBar}>
            <small className={css.composerMessageCount}>已发送 {status?.userMessageCount ?? 0} 条消息</small>
            <span className={css.composerSpacer} />
            <Button
              variant="primary"
              className={`${css.sendButton} ${generationRunning ? css.sendButtonRunning : ''}`}
              disabled={inputDisabled || uploading || (generationRunning ? !onStop : draft.trim().length === 0)}
              onClick={() => generationRunning ? onStop?.() : submitText()}
              aria-label={generationRunning ? (locale === 'zh-CN' ? '停止生成' : 'Stop generating') : t('send')}
            >
              {generationRunning ? <span className={css.stopGlyph} aria-hidden="true" /> : <IconSendOutline16 />}
            </Button>
          </div>
        </div>
        {missingApi && <SetupError error="API Key 未配置" locale={locale} />}
        <ComputationNotice sizeBytes={Math.max(uploadSize, datasetSizeBytes)} locale={locale} />
        {uploadError && <div className={css.composerUploadError} role="alert">{uploadError}</div>}
        <div className={css.composerHint}>
          <span>Enter · Shift + Enter</span>
          <div className={css.composerMeta}>
            <span
              className={css.tokenUsage}
              title={locale === 'zh-CN' ? '当前对话的累计语言模型 Token 用量' : 'Cumulative language-model token usage for this conversation'}
            >
              <b>Input</b> {formatTokenCount(tokenUsage.inputTokens)}
              <i aria-hidden="true">·</i>
              <b>Output</b> {formatTokenCount(tokenUsage.outputTokens)}
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}

const AgentTurnHeader = ({ createdAt, locale }: { createdAt: string; locale: string }): React.ReactElement => (
  <div className={css.agentTurnHeader}>
    <div className={css.messageAvatar} aria-hidden="true">
      <img src="/theta-assets/brand/theta-cat-scientist-compact-v1.png" alt="" />
    </div>
    <div className={css.agentTurnIdentity}>
      <strong>{locale === 'zh-CN' ? 'THETA猫咪科学家' : 'THETA Cat Scientist'}</strong>
      <time dateTime={createdAt}>{messageTime(createdAt, locale)}</time>
    </div>
  </div>
)

const StandardAssistantTurn = ({
  createdAt,
  locale,
  children,
  copyText,
}: {
  createdAt: string
  locale: string
  children: ReactNode
  copyText?: string
}): React.ReactElement => (
  <div className={css.standardAssistantTurn}>
    <AgentTurnHeader createdAt={createdAt} locale={locale} />
    <article className={`${css.message} ${css.messageAssistant}`}>
      <div className={css.messageBody}>
        <div className={`${css.bubble} ${css.bubbleAssistant}`}>{children}</div>
        <div className={css.messageActions}>
          <time dateTime={createdAt}>{messageTime(createdAt, locale)}</time>
          {copyText && <CopyMessageButton text={copyText} />}
        </div>
      </div>
    </article>
  </div>
)

const CopyMessageButton = ({ text }: { text: string }): React.ReactElement => {
  const [label, setLabel] = useState('复制')
  return <button type="button" aria-label={label} title={label} onClick={() => {
    void navigator.clipboard.writeText(text).then(() => setLabel('已复制'), () => setLabel('复制失败，请重试'))
  }}><IconCopyOutline16 />{label !== '复制' && <span>{label}</span>}</button>
}

const StarterVisual = ({ kind }: { kind: 'code' | 'build' | 'review' | 'measure' }): React.ReactElement => {
  const artwork = kind === 'code'
    ? <ThetaAgentOverviewArtwork />
    : kind === 'build'
      ? <ResearchIdeaArtwork />
      : kind === 'review'
        ? <ExistingDataArtwork />
        : <GuidedStartArtwork />
  return (
    <span className={`${css.starterVisual} ${css.starterVisualApproved}`} aria-hidden="true">
      {artwork}
    </span>
  )
}

const ThetaAgentOverviewArtwork = (): React.ReactElement => (
  <svg viewBox="0 0 320 200" fill="none" xmlns="http://www.w3.org/2000/svg" focusable="false">
    <defs>
      <linearGradient id="theta-card-panel" x1="90" y1="46" x2="266" y2="184" gradientUnits="userSpaceOnUse">
        <stop stopColor="#FFFFFF" />
        <stop offset="1" stopColor="#F7FAFF" />
      </linearGradient>
      <linearGradient id="theta-card-blue" x1="96" y1="93" x2="119" y2="115" gradientUnits="userSpaceOnUse">
        <stop stopColor="#78A7FF" />
        <stop offset="1" stopColor="#4D70E8" />
      </linearGradient>
      <linearGradient id="theta-card-blue-soft" x1="74" y1="35" x2="260" y2="151" gradientUnits="userSpaceOnUse">
        <stop stopColor="#EDF4FF" />
        <stop offset="1" stopColor="#DCE8FC" />
      </linearGradient>
      <filter id="theta-card-shadow-back" x="7" y="-8" width="286" height="199" filterUnits="userSpaceOnUse" colorInterpolationFilters="sRGB">
        <feDropShadow dx="0" dy="12" stdDeviation="12" floodColor="#45618D" floodOpacity="0.12" />
      </filter>
      <filter id="theta-card-shadow-front" x="48" y="25" width="272" height="174" filterUnits="userSpaceOnUse" colorInterpolationFilters="sRGB">
        <feDropShadow dx="0" dy="13" stdDeviation="11" floodColor="#36527E" floodOpacity="0.18" />
      </filter>
      <filter id="theta-card-node-glow" x="83" y="83" width="42" height="42" filterUnits="userSpaceOnUse" colorInterpolationFilters="sRGB">
        <feDropShadow dx="0" dy="4" stdDeviation="4" floodColor="#547BEA" floodOpacity="0.28" />
      </filter>
    </defs>

    <g filter="url(#theta-card-shadow-back)" transform="rotate(-5 145 88)">
      <rect x="31" y="17" width="230" height="142" rx="20" fill="url(#theta-card-blue-soft)" stroke="#C8D7EE" />
      <path d="M31 37C31 25.954 39.954 17 51 17H241C252.046 17 261 25.954 261 37V51H31V37Z" fill="#F8FBFF" fillOpacity="0.84" />
      <circle cx="49" cy="34" r="3" fill="#B5C7E4" />
      <circle cx="59" cy="34" r="3" fill="#CDDAEC" />
      <circle cx="69" cy="34" r="3" fill="#DDE6F2" />
      <rect x="49" y="70" width="63" height="7" rx="3.5" fill="#C9D8EE" />
      <rect x="49" y="88" width="147" height="6" rx="3" fill="#D7E2F1" />
      <rect x="49" y="103" width="121" height="6" rx="3" fill="#D7E2F1" />
      <rect x="49" y="118" width="137" height="6" rx="3" fill="#D7E2F1" />
    </g>

    <g filter="url(#theta-card-shadow-front)" transform="rotate(1.6 190 119)">
      <rect x="68" y="47" width="232" height="136" rx="20" fill="url(#theta-card-panel)" stroke="#C9D7EA" />
      <path d="M68 67C68 55.954 76.954 47 88 47H280C291.046 47 300 55.954 300 67V77H68V67Z" fill="#FBFCFF" />
      <circle cx="88" cy="62" r="3" fill="#9FB8E7" />
      <circle cx="98" cy="62" r="3" fill="#D2DDED" />
      <rect x="113" y="58" width="66" height="8" rx="4" fill="#DDE6F2" />
      <rect x="263" y="57" width="20" height="10" rx="5" fill="#E8F0FF" />
      <path d="M103 99V158" stroke="#C9D8EF" strokeWidth="2" strokeLinecap="round" />
      <g filter="url(#theta-card-node-glow)">
        <circle cx="103" cy="98" r="10" fill="url(#theta-card-blue)" />
        <path d="M103 92.8L107.5 95.4V100.6L103 103.2L98.5 100.6V95.4L103 92.8Z" stroke="white" strokeWidth="1.4" strokeLinejoin="round" />
        <path d="M100.5 98.1L102.2 99.8L105.6 96.4" stroke="white" strokeWidth="1.35" strokeLinecap="round" strokeLinejoin="round" />
      </g>
      <circle cx="103" cy="128" r="7" fill="#FFFFFF" stroke="#78A0EF" strokeWidth="2.5" />
      <circle cx="103" cy="158" r="7" fill="#FFFFFF" stroke="#A7BDE7" strokeWidth="2.5" />
      <rect x="123" y="92" width="112" height="8" rx="4" fill="#6F96EA" />
      <rect x="123" y="106" width="78" height="6" rx="3" fill="#DCE5F2" />
      <rect x="123" y="122" width="92" height="7" rx="3.5" fill="#CBD8EA" />
      <rect x="123" y="135" width="124" height="6" rx="3" fill="#E3E9F2" />
      <rect x="123" y="152" width="104" height="7" rx="3.5" fill="#D4DFEE" />
      <rect x="123" y="165" width="68" height="6" rx="3" fill="#E5EAF2" />
      <rect x="252" y="91" width="28" height="13" rx="6.5" fill="#EDF3FF" />
      <circle cx="260" cy="97.5" r="2.5" fill="#5D86E9" />
      <rect x="266" y="95.5" width="8" height="4" rx="2" fill="#AFC3EB" />
    </g>
  </svg>
)

const TypewriterMarkdown = ({
  text,
  active,
  speedMs,
  onComplete,
}: {
  text: string
  active: boolean
  speedMs: number
  onComplete?: () => void
}): React.ReactElement => {
  const prefersReducedMotion = typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches
  const animate = active && !prefersReducedMotion && speedMs > 0
  const [visible, setVisible] = useState(animate ? 1 : Number.POSITIVE_INFINITY)
  const completedTextRef = useRef('')
  const characters = Array.from(text)

  useEffect(() => {
    if (!animate) {
      setVisible(Number.POSITIVE_INFINITY)
      return
    }
    setVisible(1)
    const chunkSize = Math.max(1, Math.ceil(24 / Math.max(speedMs, 1)))
    const timer = window.setInterval(() => {
      setVisible((current) => {
        const next = current + chunkSize
        if (next >= characters.length) window.clearInterval(timer)
        return next
      })
    }, Math.max(8, speedMs))
    return () => window.clearInterval(timer)
  }, [animate, speedMs, text, characters.length])

  // When an existing message becomes the live assistant message, React can render
  // once before the effect above resets an earlier Infinity value. Clamp that
  // transition synchronously so the complete answer is never painted first.
  const renderedVisible = animate && !Number.isFinite(visible) ? 1 : visible
  const complete = renderedVisible >= characters.length
  useEffect(() => {
    if (!complete || completedTextRef.current === text) return
    completedTextRef.current = text
    onComplete?.()
  }, [complete, onComplete, text])
  return (
    <span className={!complete ? css.typewriterLive : undefined}>
      {complete ? <CompletedAnswer text={text} /> : <MarkdownText text={characters.slice(0, renderedVisible).join('')} />}
    </span>
  )
}

/** Collapse only the host's appended manifest; keep the answer and every link. */
const CompletedAnswer = ({ text }: { text: string }): React.ReactElement => {
  const marker = '\n\n本次可用文件：\n\n'
  const index = text.indexOf(marker)
  const remainder = index < 0 ? '' : text.slice(index + marker.length)
  const manifest = /^(?:- \[[^\n]+(?:\n|$))+/u.exec(remainder)?.[0]
  if (!manifest) return <MarkdownText text={text} />
  const count = manifest.split('\n').filter(line => line.startsWith('- [')).length
  return <>
    <MarkdownText text={text.slice(0, index)} />
    <details className={approvalCss.executionTrace}>
      <summary>可用文件（{count}）</summary>
      <MarkdownText text={manifest.replace(/\[([^\]\n]+)\](?=\()/gu, (_, name: string) => `[${artifactLabel(name)}]`)} />
    </details>
    <MarkdownText text={remainder.slice(manifest.length)} />
  </>
}
