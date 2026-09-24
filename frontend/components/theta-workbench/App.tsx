
import { OPEN_SOURCE_EDITION } from '@/lib/edition'
import { Activity, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import dynamic from 'next/dynamic'
import { ResizableSidebar } from '@/components/layout/resizable-sidebar'
import { ResultWorkspace, type ResultDestination } from '@/components/results/result-workspace'
import { useProjectDraft } from '@/lib/use-project-draft'
import { useRouter } from 'next/navigation'
import { toast } from 'sonner'
import {
  createProject as createProjectRequest,
  createWorkspaceSession,
  createRun,
  deleteProject as deleteProjectRequest,
  deleteRun,
  getConversation,
  getEvents,
  getReasoning,
  getResultCatalog,
  getRun,
  setAnalysisMode,
  getRuntimeProfile,
  getWorkspaceConversation,
  listProjects,
  listDatasets,
  listWorkspaceSessions,
  listRuns,
  openRunStream,
  postMessage,
  postWorkspaceMessage,
  stopRunGeneration,
  pinProject as pinProjectRequest,
  renameProject as renameProjectRequest,
  uploadDatasetFiles,
  type WebAgentInteraction,
  type WebAttachment,
  type WebConversationMemory,
  type WebDataset,
  type WebMessage,
  type WebReasoning,
  type WebResultCatalog,
  type WebRunDetail,
  type WebRunEvent,
  type WebRunStatus,
  type WebRunSummary,
  type WebRuntimeProfile,
  type WebRunResults,
  type WebTokenUsage,
  type WebWorkspaceSummary,
} from './api/client.ts'
import {
  Button,
  IconChevronDownOutline14,
  IconChevronLeftOutline14,
  IconChevronUpOutline14,
  IconEllipsisOutline16,
  IconEditOutline16,
  IconPanelLeftOutline16,
  IconSettingsOutline16,
  IconTrashOutline16,
  Modal,
} from './ui/index.ts'
import { CatBrandWordmark } from './ui/CatBrandWordmark.tsx'
import { ConversationPane, type QueuedChatMessage } from './panels/ConversationPane.tsx'
const ManualWorkbench = dynamic(() => import('./ManualWorkbench').then(module => module.ManualWorkbench))
import { DetailPane } from './panels/DetailPane.tsx'
import { SettingsDialog } from './panels/SettingsDialog.tsx'
import { usePreferences } from './preferences.tsx'
import { useAuth } from '@/contexts/auth-context'
import { accountStorageKey } from './storage-scope.ts'
import css from './styles/app.module.css'

type StreamState = 'idle' | 'connecting' | 'live' | 'reconnecting'
type WorkspaceMode = 'conversation' | 'manual'

interface LocalProject {
  id: string
  name: string
  createdAt: string
  pinned: boolean
  runIds: string[]
}

interface ProjectConversationMemory {
  messages: WebMessage[]
  attachments: WebAttachment[]
  interaction?: WebAgentInteraction
}

interface ProjectMemory extends ProjectConversationMemory {
  conversations?: Record<string, ProjectConversationMemory>
  workspaceSessionId?: string
  selectedRunId?: string
}

const PROJECTS_STORAGE_KEY = 'theta.workspace.projects.v1'
const PROJECT_ASSIGNMENTS_STORAGE_KEY = 'theta.workspace.project-assignments.v1'
const PROJECT_MEMORIES_STORAGE_KEY = 'theta.workspace.project-memories.v1'
const SIDEBAR_WIDTH_STORAGE_KEY = 'theta.workspace.sidebar-width.v1'
const ACCOUNT_NAME_STORAGE_KEY = 'theta.frontend.account-name.v1'
const WORKSPACE_MODE_STORAGE_KEY = 'theta.frontend.workspace-mode.v1'
const ACTIVE_CONVERSATION_STORAGE_KEY = 'theta.workspace.active-conversation.v1'
const HOME_URL = '/'
const DEFAULT_SIDEBAR_WIDTH = 232
const MIN_SIDEBAR_WIDTH = 176
const MAX_SIDEBAR_WIDTH = 360
const SIDEBAR_COLLAPSE_THRESHOLD = 120
const DRAFT_PROJECT_ID = '__theta_home_draft__'
const DEFAULT_PROJECTS: LocalProject[] = []
const conversationTitleFromText = (text: string, fallback = '新对话'): string => {
  const firstLine = text.trim().split(/\r?\n/u, 1)[0]?.trim() ?? ''
  const normalized = firstLine.replace(/\s+/gu, ' ')
  return normalized.slice(0, 120) || fallback
}

const isUuid = (value: string): boolean =>
  /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/iu.test(value)

const isTransientLocalError = (message: WebMessage): boolean =>
  message.messageKind === 'conversation.error' && message.messageId.startsWith('local.error.')

const withoutTransientLocalErrors = (messages: WebMessage[]): WebMessage[] =>
  messages.some(isTransientLocalError)
    ? messages.filter((message) => !isTransientLocalError(message))
    : messages

const compareMessageOrder = (left: WebMessage, right: WebMessage): number => {
  const leftTime = Date.parse(left.createdAt)
  const rightTime = Date.parse(right.createdAt)
  if (Number.isFinite(leftTime) && Number.isFinite(rightTime) && leftTime !== rightTime) return leftTime - rightTime
  const leftSequence = Number.isFinite(left.sequenceNumber) ? left.sequenceNumber : Number.MAX_SAFE_INTEGER
  const rightSequence = Number.isFinite(right.sequenceNumber) ? right.sequenceNumber : Number.MAX_SAFE_INTEGER
  if (leftSequence !== rightSequence) return leftSequence - rightSequence
  if (left.role !== right.role) return left.role === 'user' ? -1 : 1
  return left.messageId.localeCompare(right.messageId)
}

const mergeStoredMessages = (
  current: WebMessage[],
  incoming: WebMessage[],
  excludedIds: string[] = [],
): WebMessage[] => {
  const excluded = new Set(excludedIds)
  const byId = new Map(
    withoutTransientLocalErrors(current)
      .filter((message) => !excluded.has(message.messageId))
      .map((message) => [message.messageId, message]),
  )
  for (const message of incoming) byId.set(message.messageId, message)
  return [...byId.values()].sort(compareMessageOrder)
}

const upsertLocalMessage = (
  current: WebMessage[],
  message: Omit<WebMessage, 'sequenceNumber'>,
): WebMessage[] => {
  const existing = current.find((item) => item.messageId === message.messageId)
  const nextMessage: WebMessage = {
    ...message,
    sequenceNumber: existing?.sequenceNumber ?? Math.max(0, ...current.map((item) => Number.isFinite(item.sequenceNumber) ? item.sequenceNumber : 0)) + 0.5,
  }
  return [...current.filter((item) => item.messageId !== message.messageId), nextMessage]
    .sort(compareMessageOrder)
}

const readStoredProjects = (storageKey = PROJECTS_STORAGE_KEY): LocalProject[] => {
  try {
    const stored = JSON.parse(localStorage.getItem(storageKey) ?? 'null') as unknown
    if (!Array.isArray(stored)) return DEFAULT_PROJECTS
    const projects = stored.filter((item): item is Omit<LocalProject, 'createdAt'> & { createdAt?: string } =>
      item != null && typeof item === 'object' &&
      typeof (item as LocalProject).id === 'string' &&
      typeof (item as LocalProject).name === 'string',
    )
    return projects.map((project, index) => ({
      ...project,
      runIds: Array.isArray(project.runIds) ? project.runIds.filter((runId): runId is string => typeof runId === 'string') : [],
      createdAt: typeof project.createdAt === 'string' && Number.isFinite(Date.parse(project.createdAt))
        ? project.createdAt
        : new Date(index).toISOString(),
    }))
  } catch {
    return DEFAULT_PROJECTS
  }
}

const readProjectAssignments = (storageKey = PROJECT_ASSIGNMENTS_STORAGE_KEY): Record<string, string> => {
  try {
    const stored = JSON.parse(localStorage.getItem(storageKey) ?? '{}') as unknown
    return stored != null && typeof stored === 'object' && !Array.isArray(stored)
      ? stored as Record<string, string>
      : {}
  } catch {
    return {}
  }
}

const readProjectMemories = (storageKey = PROJECT_MEMORIES_STORAGE_KEY): Record<string, ProjectMemory> => {
  try {
    const stored = JSON.parse(localStorage.getItem(storageKey) ?? '{}') as unknown
    if (stored == null || typeof stored !== 'object' || Array.isArray(stored)) return {}
    return Object.fromEntries(Object.entries(stored as Record<string, ProjectMemory>).map(([projectId, memory]) => {
      const conversations = Object.fromEntries(Object.entries(memory.conversations ?? {}).map(([conversationId, conversation]) => [
        conversationId,
        { ...conversation, messages: withoutTransientLocalErrors(conversation.messages ?? []) },
      ]))
      return [projectId, {
        ...memory,
        messages: withoutTransientLocalErrors(memory.messages ?? []),
        ...(Object.keys(conversations).length > 0 ? { conversations } : {}),
      }]
    }))
  } catch {
    return {}
  }
}

const readSidebarWidth = (): number => {
  const raw = localStorage.getItem(SIDEBAR_WIDTH_STORAGE_KEY)
  if (raw == null) return DEFAULT_SIDEBAR_WIDTH
  const stored = Number(raw)
  if (!Number.isFinite(stored)) return DEFAULT_SIDEBAR_WIDTH
  return Math.min(MAX_SIDEBAR_WIDTH, Math.max(MIN_SIDEBAR_WIDTH, stored))
}

const EMPTY_TOKEN_USAGE: WebTokenUsage = {
  inputTokens: 0,
  outputTokens: 0,
  totalTokens: 0,
  calls: 0,
}

const DATASET_UPLOAD_INTERACTION: WebAgentInteraction = {
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

const normalizeDatasetIntentText = (text: string): string =>
  text.replace(/[\s，。！？,.!?：:；;“”"'（）()]/gu, '')

const isDatasetUploadPrompt = (text: string): boolean => {
  const normalized = normalizeDatasetIntentText(text)
  return /(?:上传|导入|选择|提供|提交).*(?:数据集|数据文件|文本数据|文件)/u.test(normalized) ||
    /(?:数据集|数据文件|文本数据|文件).*(?:上传|导入|选择|提供|提交)/u.test(normalized) ||
    /(?:是否有|有没有|有无|有现成的?).*(?:数据集|数据文件|文本数据|文件)/u.test(normalized)
}

const hasExplicitDatasetAvailability = (text: string): boolean => {
  const normalized = normalizeDatasetIntentText(text)
  return /^(?:我)?(?:这里|这边|手上|本地)?(?:已经|现在)?有(?:现成的?)?(?:一个|一些)?(?:文本)?(?:数据集|数据|数据文件|文件)(?:了)?$/u.test(normalized) ||
    /^(?:我)?(?:已经|现在)?(?:准备好|准备好了|拿到|拥有)(?:一个|一些)?(?:文本)?(?:数据集|数据|数据文件|文件)(?:了)?$/u.test(normalized) ||
    /^(?:数据集|数据文件|文本数据|文件)(?:已经|都)?(?:准备好|准备好了|有了|在本地)$/u.test(normalized)
}

const isDatasetUploadRequest = (text: string, previousMessages: WebMessage[] = []): boolean => {
  const normalized = normalizeDatasetIntentText(text)
  if (/(?:已|已经|刚|成功).{0,4}上传|上传.{0,4}(?:成功|完成|好了)/u.test(normalized)) return false
  const containsDatasetSubject = /(?:数据集|数据文件|文本数据|文件)/u.test(normalized)
  const isNegative = /(?:没有|没准备|暂无|无|不需要|不想|不要|暂时不|先不).{0,6}(?:数据集|数据|文件)/u.test(normalized) ||
    /(?:数据集|数据文件|文本数据|文件).{0,6}(?:没有|不存在|不上传|不用上传)/u.test(normalized)
  if (isNegative) return false

  const hasDirectUploadIntent = /(?:上传|导入|添加|选择|提供|提交).*(?:数据集|数据文件|文本数据|文件)/u.test(normalized) ||
    /(?:数据集|数据文件|文本数据|文件).*(?:上传|导入|添加|选择|提供|提交)/u.test(normalized) ||
    /(?:我要|我想|我准备|我需要|帮我)?传(?:一个|一下|我的)?(?:数据集|数据文件|文本数据|文件)/u.test(normalized)
  if (hasDirectUploadIntent) return true

  if (hasExplicitDatasetAvailability(text)) return true

  const isAffirmativeReply = /^(?:有|有的|我有|可以|好的?|是的?|要|需要)$/u.test(normalized)
  if (!isAffirmativeReply || containsDatasetSubject) return false
  const latestAssistantMessage = [...previousMessages].reverse().find((message) =>
    message.role === 'assistant' && message.messageKind === 'conversation.text',
  )
  return latestAssistantMessage != null && isDatasetUploadPrompt(latestAssistantMessage.content)
}

const createDatasetUploadCardMessage = (sequenceNumber: number): WebMessage => ({
  messageId: `local.dataset-card.${crypto.randomUUID()}`,
  role: 'assistant',
  messageKind: 'conversation.dataset-upload-card',
  content: DATASET_UPLOAD_INTERACTION.card?.description ?? '',
  sequenceNumber,
  createdAt: new Date().toISOString(),
})

const shouldRecoverDatasetUploadCard = (messages: WebMessage[], attachments: WebAttachment[]): boolean => {
  if (attachments.some((attachment) => attachment.kind === 'dataset') ||
    messages.some((message) => message.messageKind === 'conversation.dataset-upload-card')) return false

  const latestUserIndex = messages.findLastIndex((message) =>
    message.role === 'user' && message.messageKind === 'conversation.text',
  )
  if (latestUserIndex < 0) return false
  const latestUserMessage = messages[latestUserIndex]
  if (latestUserMessage == null || !isDatasetUploadRequest(latestUserMessage.content, messages.slice(0, latestUserIndex))) return false

  const assistantReply = messages.slice(latestUserIndex + 1).find((message) =>
    message.role === 'assistant' && message.messageKind === 'conversation.text',
  )
  return assistantReply != null && isDatasetUploadPrompt(assistantReply.content)
}

const readWorkspaceMode = (): WorkspaceMode => {
  const requested = new URLSearchParams(window.location.search).get('mode')
  if (requested === 'conversation' || requested === 'manual') return requested
  return localStorage.getItem(WORKSPACE_MODE_STORAGE_KEY) === 'manual' ? 'manual' : 'conversation'
}

const persistWorkspaceState = (key: string, value: unknown): void => {
  // localStorage 有硬配额：写不进去时只丢缓存，绝不让整个工作台崩掉。
  try {
    localStorage.setItem(key, JSON.stringify(value))
  } catch {
    try {
      localStorage.removeItem(key)
      localStorage.setItem(key, JSON.stringify(value))
    } catch { /* 仍然超限：本次不缓存，服务端数据不受影响 */ }
  }
}

/** 会话记忆按项目只保留最近的消息，避免 localStorage 无限增长。 */
const boundedMemories = (memories: Record<string, { messages?: unknown[]; attachments?: unknown[]; selectedRunId?: string }>): Record<string, unknown> =>
  Object.fromEntries(Object.entries(memories).map(([projectId, memory]) => [projectId, {
    ...memory,
    messages: Array.isArray(memory.messages) ? memory.messages.slice(-30) : [],
    attachments: Array.isArray(memory.attachments) ? memory.attachments.slice(-12) : [],
  }]))

export const AppRoot = ({ initialMode }: { initialMode?: WorkspaceMode }): React.ReactElement => {
  const { locale, t } = usePreferences()
  useEffect(() => {
    const preventFileNavigation = (event: DragEvent) => {
      if (Array.from(event.dataTransfer?.types ?? []).includes('Files')) event.preventDefault()
    }
    window.addEventListener('dragover', preventFileNavigation)
    window.addEventListener('drop', preventFileNavigation)
    return () => { window.removeEventListener('dragover', preventFileNavigation); window.removeEventListener('drop', preventFileNavigation) }
  }, [])
  const router = useRouter()
  const { user, loading: authLoading, logout } = useAuth()
  const accountScope = user?.id ?? 'anonymous'
  const navigationKey = accountStorageKey(ACTIVE_CONVERSATION_STORAGE_KEY, user?.id)
  const [navigationReadyKey, setNavigationReadyKey] = useState<string>()
  const storageKeys = useMemo(() => ({
    projects: accountStorageKey(PROJECTS_STORAGE_KEY, user?.id),
    assignments: accountStorageKey(PROJECT_ASSIGNMENTS_STORAGE_KEY, user?.id),
    memories: accountStorageKey(PROJECT_MEMORIES_STORAGE_KEY, user?.id),
    accountName: accountStorageKey(ACCOUNT_NAME_STORAGE_KEY, user?.id),
  }), [user?.id])
  const initialProjectId = useRef(DRAFT_PROJECT_ID).current
  const initialProjects = useRef(readStoredProjects(storageKeys.projects)).current
  const initialProjectMemories = useRef(readProjectMemories(storageKeys.memories)).current
  const [projects, setProjects] = useState<LocalProject[]>(initialProjects)
  const [projectAssignments, setProjectAssignments] = useState<Record<string, string>>(() => readProjectAssignments(storageKeys.assignments))
  const [projectMemories, setProjectMemories] = useState<Record<string, ProjectMemory>>(initialProjectMemories)
  const [activeProjectId, setActiveProjectId] = useState(initialProjectId)
  const [openProjectMenuId, setOpenProjectMenuId] = useState<string>()
  const [expandedProjectIds, setExpandedProjectIds] = useState<Set<string>>(() => new Set())
  const [analysisMode, setLocalAnalysisMode] = useState<'topic' | 'free'>('topic')
  const [analysisModeBusy, setAnalysisModeBusy] = useState(false)
  const [workspaceMode, setWorkspaceMode] = useState<WorkspaceMode>(() => initialMode ?? readWorkspaceMode())
  const [projectToDelete, setProjectToDelete] = useState<LocalProject>()
  const [projectDeleteBusy, setProjectDeleteBusy] = useState(false)
  const [projectDeleteError, setProjectDeleteError] = useState<string>()
  const [projectToRename, setProjectToRename] = useState<LocalProject>()
  const [projectRenameName, setProjectRenameName] = useState('')
  const [projectRenameBusy, setProjectRenameBusy] = useState(false)
  const [projectRenameError, setProjectRenameError] = useState<string>()
  const [projectCreateOpen, setProjectCreateOpen] = useState(false)
  const [projectCreateName, setProjectCreateName] = useState('')
  const [projectCreateBusy, setProjectCreateBusy] = useState(false)
  const [projectCreateError, setProjectCreateError] = useState<string>()
  const [runs, setRuns] = useState<WebRunSummary[]>([])
  const [workspaceSessions, setWorkspaceSessions] = useState<WebWorkspaceSummary[]>([])
  const [runsLoading, setRunsLoading] = useState(true)
  const [selectedRunId, setSelectedRunId] = useState<string | undefined>()
  const [messages, setMessages] = useState<WebMessage[]>([])
  const [status, setStatus] = useState<WebRunStatus>()
  const [runDetail, setRunDetail] = useState<WebRunDetail>()
  const [events, setEvents] = useState<WebRunEvent[]>([])
  const [reasoning, setReasoning] = useState<WebReasoning>()
  const [results, setResults] = useState<WebRunResults>()
  const [resultCatalog, setResultCatalog] = useState<WebResultCatalog>()
  const [memory, setMemory] = useState<WebConversationMemory>()
  const [tokenUsage, setTokenUsage] = useState<WebTokenUsage>(EMPTY_TOKEN_USAGE)
  const [sending, setSending] = useState(false)
  const [processingMessageId, setProcessingMessageId] = useState<string>()
  const [activeScopeKey, setActiveScopeKey] = useState(() => `draft:${crypto.randomUUID()}`)
  const [queued, setQueued] = useState<Array<QueuedChatMessage & {
    runId?: string
    workspaceSessionId?: string
    scopeKey: string
    analysisMode: 'topic' | 'free'
    projectId: string
    optimisticMessageId?: string
  }>>([])
  const [attachments, setAttachments] = useState<WebAttachment[]>([])
  const [loadError, setLoadError] = useState<string>()
  const [streamState, setStreamState] = useState<StreamState>('idle')
  const [runtimeProfile, setRuntimeProfile] = useState<WebRuntimeProfile>()
  const [workspaceSessionId, setWorkspaceSessionId] = useState<string | undefined>()
  const [workspaceInteraction, setWorkspaceInteraction] = useState<WebAgentInteraction | undefined>()
  const [workspaceActivity, setWorkspaceActivity] = useState<{ proposal?: unknown; semanticDecision?: unknown; steps?: unknown; result?: unknown; evidenceRefs?: unknown }>()
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [sidebarWidth, setSidebarWidth] = useState(readSidebarWidth)
  const [sidebarResizing, setSidebarResizing] = useState(false)
  const [detailOpen, setDetailOpen] = useState(true)
  const [detailTab, setDetailTab] = useState<'progress' | 'results'>('progress')
  const [manualVisited, setManualVisited] = useState(workspaceMode === 'manual')
  const switchWorkspace = (mode: WorkspaceMode): void => {
    if (mode === 'manual') setManualVisited(true)
    setWorkspaceMode(mode)
  }
  const [datasetSizes, setDatasetSizes] = useState<Record<string, number>>({})
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [settingsTab, setSettingsTab] = useState<'inference' | 'embedding'>()
  useEffect(() => {
    const open = (event: Event) => {
      const tab = (event as CustomEvent).detail
      setSettingsTab(tab === 'embedding' ? 'embedding' : 'inference'); setSettingsOpen(true)
    }
    window.addEventListener('theta:open-settings', open)
    return () => window.removeEventListener('theta:open-settings', open)
  }, [])
  useEffect(() => window.thetaDesktop?.onOpenSettings(() => setSettingsOpen(true)), [])
  const [accountMenuOpen, setAccountMenuOpen] = useState(false)
  const [accountName, setAccountName] = useState(() => localStorage.getItem(storageKeys.accountName) || 'user')
  const [liveAssistantMessageId, setLiveAssistantMessageId] = useState<string>()
  const knownMessageIds = useRef(new Set<string>())
  const activeScopeRef = useRef(activeScopeKey)
  const activeProjectIdRef = useRef(activeProjectId)
  const processingMessageRef = useRef<string | undefined>(undefined)
  const queuedSubmissionCountRef = useRef(0)
  const activeRequestControllerRef = useRef<AbortController | undefined>(undefined)
  const awaitingRenderRef = useRef<{ messageId: string; queueId: string } | undefined>(undefined)
  const finishedQueueIdsRef = useRef(new Set<string>())
  const preserveMessagesRunRef = useRef<string | undefined>(undefined)
  const projectMemoriesRef = useRef(projectMemories)
  const messagesRef = useRef(messages)
  const sidebarResizeStartRef = useRef<{ pointerX: number; width: number } | undefined>(undefined)
  const accountScopeRef = useRef(accountScope)
  const authRedirectingRef = useRef(false)
  const observedTerminalTrainingRef = useRef<string | undefined>(undefined)
  const activeConversationId = selectedRunId ?? workspaceSessionId
  const [resultDestination, setResultDestination] = useProjectDraft<ResultDestination | null>(`conversation:${activeConversationId ?? 'draft'}:result-destination`, null)

  useEffect(() => {
    setAccountName(user?.username ?? 'user')
  }, [user?.id, user?.username])

  useEffect(() => {
    if (!accountMenuOpen) return
    const closeMenu = (event: PointerEvent): void => {
      if (event.target instanceof Element && !event.target.closest('[data-account-menu]')) {
        setAccountMenuOpen(false)
      }
    }
    const closeOnEscape = (event: KeyboardEvent): void => {
      if (event.key === 'Escape') setAccountMenuOpen(false)
    }
    document.addEventListener('pointerdown', closeMenu)
    document.addEventListener('keydown', closeOnEscape)
    return () => {
      document.removeEventListener('pointerdown', closeMenu)
      document.removeEventListener('keydown', closeOnEscape)
    }
  }, [accountMenuOpen])

  useEffect(() => {
    if (accountScopeRef.current === accountScope) return
    accountScopeRef.current = accountScope
    const nextProjects = readStoredProjects(storageKeys.projects)
    setProjects(nextProjects)
    setProjectAssignments(readProjectAssignments(storageKeys.assignments))
    setProjectMemories(readProjectMemories(storageKeys.memories))
    setActiveProjectId(DRAFT_PROJECT_ID)
    activeProjectIdRef.current = DRAFT_PROJECT_ID
    setSelectedRunId(undefined)
    setWorkspaceSessionId(undefined)
    setMessages([])
    setAttachments([])
    setWorkspaceInteraction(undefined)
    setWorkspaceActivity(undefined)
    setLoadError(undefined)
    knownMessageIds.current.clear()
  }, [accountScope, storageKeys.assignments, storageKeys.memories, storageKeys.projects])

  useEffect(() => {
    persistWorkspaceState(storageKeys.projects, projects)
  }, [projects, storageKeys.projects])

  useEffect(() => {
    persistWorkspaceState(storageKeys.assignments, projectAssignments)
  }, [projectAssignments, storageKeys.assignments])

  useEffect(() => {
    activeProjectIdRef.current = activeProjectId
  }, [activeProjectId])

  useEffect(() => {
    projectMemoriesRef.current = projectMemories
    persistWorkspaceState(storageKeys.memories, boundedMemories(projectMemories))
  }, [projectMemories, storageKeys.memories])

  useEffect(() => {
    messagesRef.current = messages
  }, [messages])

  useEffect(() => {
    if (activeProjectId === DRAFT_PROJECT_ID) return
    let cancelled = false
    void listDatasets(activeProjectId).then(({ datasets }) => {
      if (cancelled || datasets.length === 0) return
      setDatasetSizes(current => ({ ...current, [activeProjectId]: datasets.reduce((sum, item) => sum + item.sizeBytes, 0) }))
      const restored = datasets.map((dataset): WebAttachment => ({
        kind: 'dataset',
        id: dataset.datasetRef,
        label: dataset.name,
      }))
      setAttachments((current) => {
        const merged = [...restored, ...current].filter((item, index, all) =>
          all.findIndex((candidate) => candidate.kind === item.kind && candidate.id === item.id) === index,
        ).slice(-12)
        return merged.length === current.length && merged.every((item, index) => item.id === current[index]?.id)
          ? current
          : merged
      })
      setProjectMemories((current) => {
        const previous = current[activeProjectId] ?? { messages: messagesRef.current, attachments: [] }
        const merged = [...restored, ...previous.attachments].filter((item, index, all) =>
          all.findIndex((candidate) => candidate.kind === item.kind && candidate.id === item.id) === index,
        ).slice(-12)
        return { ...current, [activeProjectId]: { ...previous, attachments: merged } }
      })
    }).catch(() => undefined)
    return () => { cancelled = true }
  }, [activeProjectId])

  useEffect(() => {
    if (activeProjectId === DRAFT_PROJECT_ID || !shouldRecoverDatasetUploadCard(messages, attachments)) return
    setMessages((current) => {
      if (!shouldRecoverDatasetUploadCard(current, attachments)) return current
      const cardMessage = createDatasetUploadCardMessage(
        Math.max(0, ...current.map((message) => message.sequenceNumber)) + 1,
      )
      knownMessageIds.current.add(cardMessage.messageId)
      const next = [...current, cardMessage]
      messagesRef.current = next
      return next
    })
  }, [activeProjectId, attachments, messages])

  useEffect(() => {
    localStorage.setItem(SIDEBAR_WIDTH_STORAGE_KEY, String(sidebarWidth))
  }, [sidebarWidth])

  useEffect(() => {
    localStorage.setItem(WORKSPACE_MODE_STORAGE_KEY, workspaceMode)
    const url = new URL(window.location.href)
    url.searchParams.set('mode', workspaceMode)
    window.history.replaceState(null, '', `${url.pathname}${url.search}`)
  }, [workspaceMode])

  useEffect(() => {
    if (!sidebarResizing) return
    const previousCursor = document.body.style.cursor
    const previousUserSelect = document.body.style.userSelect
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'

    const onPointerMove = (event: PointerEvent): void => {
      const start = sidebarResizeStartRef.current
      if (start == null) return
      const nextWidth = start.width + event.clientX - start.pointerX
      if (nextWidth <= SIDEBAR_COLLAPSE_THRESHOLD) {
        setSidebarOpen(false)
        return
      }
      setSidebarOpen(true)
      setSidebarWidth(Math.min(MAX_SIDEBAR_WIDTH, Math.max(MIN_SIDEBAR_WIDTH, nextWidth)))
    }
    const stopResizing = (): void => {
      sidebarResizeStartRef.current = undefined
      setSidebarResizing(false)
    }

    window.addEventListener('pointermove', onPointerMove)
    window.addEventListener('pointerup', stopResizing)
    window.addEventListener('pointercancel', stopResizing)
    return () => {
      window.removeEventListener('pointermove', onPointerMove)
      window.removeEventListener('pointerup', stopResizing)
      window.removeEventListener('pointercancel', stopResizing)
      document.body.style.cursor = previousCursor
      document.body.style.userSelect = previousUserSelect
    }
  }, [sidebarResizing])

  useEffect(() => {
    const conversationId = workspaceSessionId ?? selectedRunId
    if (activeProjectId === DRAFT_PROJECT_ID || conversationId == null) return
    const conversation: ProjectConversationMemory = {
      messages: withoutTransientLocalErrors(messages),
      attachments,
      ...(workspaceInteraction ? { interaction: workspaceInteraction } : {}),
    }
    const next: ProjectMemory = {
      messages: conversation.messages,
      attachments: conversation.attachments,
      ...(conversation.interaction ? { interaction: conversation.interaction } : {}),
      conversations: {
        ...(projectMemoriesRef.current[activeProjectId]?.conversations ?? {}),
        [conversationId]: conversation,
      },
      ...(workspaceSessionId ? { workspaceSessionId } : {}),
      ...(selectedRunId ? { selectedRunId } : {}),
    }
    setProjectMemories((current) => {
      const previous = current[activeProjectId]
      const previousConversation = previous?.conversations?.[conversationId]
      if (previousConversation?.messages === conversation.messages && previousConversation.attachments === attachments &&
        previousConversation.interaction === workspaceInteraction && previous?.workspaceSessionId === workspaceSessionId &&
        previous?.selectedRunId === selectedRunId) return current
      return { ...current, [activeProjectId]: next }
    })
  }, [activeProjectId, attachments, messages, selectedRunId, workspaceInteraction, workspaceSessionId])

  useEffect(() => {
    const interaction = status?.interaction
    const card = interaction?.card
    if (activeProjectId === DRAFT_PROJECT_ID || selectedRunId == null || interaction == null || card == null ||
      card.kind === 'action_review' || card.kind === 'dataset_upload' || card.kind === 'research_question') return
    const eventKey = (card.actionRef ?? card.kind).replace(/[^a-zA-Z0-9_.-]/gu, '-')
    const localMessage = {
      messageId: `local.interaction.${selectedRunId}.${eventKey}`,
      role: 'assistant' as const,
      messageKind: 'activity.interaction.snapshot',
      content: JSON.stringify({ runId: selectedRunId, interaction }),
      createdAt: status?.lastEventAt ?? new Date().toISOString(),
    }
    knownMessageIds.current.add(localMessage.messageId)
    setMessages((current) => {
      const next = upsertLocalMessage(current, localMessage)
      messagesRef.current = next
      return next
    })
  }, [activeProjectId, selectedRunId, status?.interaction, status?.lastEventAt])

  const projectIdFor = useCallback((itemId: string): string => {
    const serverProjectId = runs.find((run) => run.runId === itemId)?.projectId
    if (serverProjectId) return serverProjectId
    const assignedProjectId = projectAssignments[itemId]
    return assignedProjectId && projects.some((project) => project.id === assignedProjectId)
      ? assignedProjectId
      : DRAFT_PROJECT_ID
  }, [projectAssignments, projects, runs])

  const assignItemToProject = useCallback((itemId: string, projectId: string): void => {
    setProjectAssignments((current) => {
      if (projectId === DRAFT_PROJECT_ID) {
        if (!(itemId in current)) return current
        const next = { ...current }
        delete next[itemId]
        return next
      }
      return current[itemId] === projectId ? current : { ...current, [itemId]: projectId }
    })
    if (projectId !== DRAFT_PROJECT_ID) {
      setProjects((current) => current.map((project) => project.id === projectId && !project.runIds.includes(itemId)
        ? { ...project, runIds: [...project.runIds, itemId] }
        : project,
      ))
    }
  }, [])

  const cacheProjectResult = useCallback((
    projectId: string,
    incoming: WebMessage[],
    excludedIds: string[],
    identity: { selectedRunId?: string; workspaceSessionId?: string } = {},
  ): void => {
    if (projectId === DRAFT_PROJECT_ID) return
    setProjectMemories((current) => {
      const previous = current[projectId] ?? { messages: [], attachments: [] }
      const conversationId = identity.selectedRunId ?? identity.workspaceSessionId
      const previousConversation = conversationId
        ? previous.conversations?.[conversationId]
        : undefined
      const nextConversation = conversationId
        ? {
            ...(previousConversation ?? { messages: [], attachments: previous.attachments }),
            messages: mergeStoredMessages(previousConversation?.messages ?? [], incoming, excludedIds),
          }
        : undefined
      return {
        ...current,
        [projectId]: {
          ...previous,
          messages: nextConversation?.messages ?? mergeStoredMessages(previous.messages, incoming, excludedIds),
          attachments: nextConversation?.attachments ?? previous.attachments,
          ...(nextConversation?.interaction ? { interaction: nextConversation.interaction } : {}),
          ...(conversationId && nextConversation ? {
            conversations: {
              ...(previous.conversations ?? {}),
              [conversationId]: nextConversation,
            },
          } : {}),
          ...identity,
        },
      }
    })
  }, [])

  const finishQueuedMessage = useCallback((queueId: string): void => {
    if (finishedQueueIdsRef.current.has(queueId)) return
    finishedQueueIdsRef.current.add(queueId)
    setQueued((current) => current.filter((item) => item.id !== queueId))
    queuedSubmissionCountRef.current = Math.max(0, queuedSubmissionCountRef.current - 1)
    if (processingMessageRef.current === queueId) processingMessageRef.current = undefined
    if (awaitingRenderRef.current?.queueId === queueId) awaitingRenderRef.current = undefined
    activeRequestControllerRef.current = undefined
    setProcessingMessageId(undefined)
    setSending(false)
  }, [])

  const optimisticMessages = useCallback((
    id: string,
    text: string,
    nextAttachments: WebAttachment[],
    sequence: number,
  ): WebMessage[] => {
    const createdAt = new Date().toISOString()
    return [
      {
        messageId: `optimistic.${id}`,
        role: 'user',
        messageKind: 'conversation.text',
        content: text,
        sequenceNumber: sequence,
        createdAt,
      },
      ...(nextAttachments.length > 0 ? [{
        messageId: `optimistic.${id}.attachments`,
        role: 'user' as const,
        messageKind: 'conversation.attachment',
        content: JSON.stringify({ attachments: nextAttachments }),
        sequenceNumber: sequence + 1,
        createdAt,
      }] : []),
    ]
  }, [])

  const activateScope = useCallback((scopeKey: string): void => {
    activeScopeRef.current = scopeKey
    setActiveScopeKey(scopeKey)
  }, [])

  const refreshRuns = useCallback(async () => {
    try {
      const data = await listRuns()
      setRuns(data.runs)
      setProjectAssignments((current) => ({
        ...current,
        ...Object.fromEntries(data.runs.map((run) => [run.runId, run.projectId])),
      }))
      setSelectedRunId((current) => {
        if (current != null && data.runs.some((run) => run.runId === current)) return current
        return undefined
      })
      setLoadError(undefined)
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : String(error))
    } finally {
      setRunsLoading(false)
    }
  }, [])

  const refreshProjects = useCallback(async () => {
    const data = await listProjects()
    const serverProjects = data.projects.map(({ id, name, createdAt, pinned, runIds }) => ({ id, name, createdAt, pinned, runIds }))
    const serverProjectIds = new Set(serverProjects.map((project) => project.id))
    setProjects(serverProjects)
    setProjectAssignments((current) => ({
      ...Object.fromEntries(Object.entries(current).filter(([, projectId]) => serverProjectIds.has(projectId))),
      ...Object.fromEntries(serverProjects.flatMap((project) => project.runIds.map((runId) => [runId, project.id]))),
    }))
    setProjectMemories((current) => Object.fromEntries(
      Object.entries(current).filter(([projectId]) => serverProjectIds.has(projectId)),
    ))
  }, [])

  const refreshWorkspaceSessions = useCallback(async () => {
    try {
      setWorkspaceSessions((await listWorkspaceSessions()).sessions)
    } catch {
      setWorkspaceSessions([])
    }
  }, [])

  useEffect(() => {
    if (authLoading) return
    if (!user && !OPEN_SOURCE_EDITION) {
      setRuns([])
      setWorkspaceSessions([])
      setRunsLoading(false)
      return
    }
    void refreshRuns()
    void refreshProjects().catch((error) => {
      setLoadError(error instanceof Error ? error.message : String(error))
    })
    void refreshWorkspaceSessions()
  }, [authLoading, refreshProjects, refreshRuns, refreshWorkspaceSessions, user?.id])

  const requireAuthentication = useCallback((): boolean => {
    if (OPEN_SOURCE_EDITION) return true
    if (authLoading) return false
    if (user) return true
    if (authRedirectingRef.current) return false
    authRedirectingRef.current = true
    toast.error('您当前暂未登录，登陆后进行后续使用。', { duration: 5000 })
    router.replace('/?auth=login')
    return false
  }, [authLoading, router, user])

  useEffect(() => {
    if (!authLoading && !user) requireAuthentication()
  }, [authLoading, requireAuthentication, user])

  const ensureProject = useCallback(async (suggestedName: string): Promise<string> => {
    if (!requireAuthentication()) throw new Error('请先登录。')
    const currentProjectId = activeProjectIdRef.current
    if (currentProjectId !== DRAFT_PROJECT_ID && (
      projects.some((project) => project.id === currentProjectId) || isUuid(currentProjectId)
    )) {
      return currentProjectId
    }
    const created = await createProjectRequest(suggestedName.trim().replace(/\s+/gu, ' ') || '新分析项目')
    setProjects((current) => current.some((project) => project.id === created.id)
      ? current
      : [...current, { id: created.id, name: created.name, createdAt: created.createdAt, pinned: created.pinned, runIds: created.runIds }])
    setActiveProjectId(created.id)
    activeProjectIdRef.current = created.id
    return created.id
  }, [projects, requireAuthentication])

  useEffect(() => {
    void getRuntimeProfile()
      .then((profile) => {
        setRuntimeProfile(profile)
      })
      .catch(() => setRuntimeProfile(undefined))
  }, [])

  useEffect(() => {
    const narrow = window.matchMedia('(max-width: 900px)')
    const syncPanels = (matches: boolean): void => {
      setSidebarOpen(!matches)
      setDetailOpen(!matches)
    }
    syncPanels(narrow.matches)
    const onChange = (event: MediaQueryListEvent): void => syncPanels(event.matches)
    narrow.addEventListener('change', onChange)
    return () => narrow.removeEventListener('change', onChange)
  }, [])

  const mergeMessages = useCallback((incoming: WebMessage[]) => {
    setMessages((current) => {
      const next = [...current]
      for (const message of incoming) {
        if (knownMessageIds.current.has(message.messageId)) {
          const index = next.findIndex(item => item.messageId === message.messageId)
          if (index >= 0) next[index] = message
          continue
        }
        if (message.role === 'user') {
          const optimisticIndex = next.findIndex((item) =>
            item.messageId.startsWith('optimistic.') && item.role === 'user' && item.content === message.content,
          )
          if (optimisticIndex >= 0) {
            const optimistic = next[optimisticIndex]
            if (optimistic) knownMessageIds.current.delete(optimistic.messageId)
            next.splice(optimisticIndex, 1)
          }
        }
        knownMessageIds.current.add(message.messageId)
        next.push(message)
      }
      next.sort(compareMessageOrder)
      messagesRef.current = next
      return next
    })
  }, [])

  const refreshRun = useCallback(async () => {
    if (!activeConversationId) return
    try {
      const [detail, conversation, eventData, reasoningData, catalog] = await Promise.all([
        getRun(activeConversationId),
        getConversation(activeConversationId),
        getEvents(activeConversationId, { limit: 300 }),
        getReasoning(activeConversationId),
        getResultCatalog(activeConversationId),
      ])
      setStatus(detail.status)
      setRunDetail(detail)
      mergeMessages(conversation.messages)
      setEvents(eventData.events)
      setReasoning(reasoningData)
      setResults(detail.results)
      setResultCatalog(catalog)
      setMemory(conversation.memory)
      setTokenUsage(conversation.tokenUsage)
      setLoadError(undefined)
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : String(error))
    }
    void refreshRuns()
  }, [activeConversationId, mergeMessages, refreshRuns])

  useEffect(() => {
    if (!activeConversationId) {
      setStreamState('idle')
      setResultCatalog(undefined)
      return
    }
    let cancelled = false
    let reasoningTimer: number | undefined
    const projectMemory = projectMemoriesRef.current[projectIdFor(activeConversationId)]
    const cachedMessages = projectMemory?.selectedRunId === activeConversationId || projectMemory?.workspaceSessionId === activeConversationId
      ? withoutTransientLocalErrors(projectMemory.messages)
      : []
    const preserveCurrentMessages = preserveMessagesRunRef.current === activeConversationId
    if (preserveCurrentMessages) preserveMessagesRunRef.current = undefined
    setLoadError(undefined)
    if (!preserveCurrentMessages) setMessages(cachedMessages)
    setAttachments(projectMemory?.attachments ?? [])
    setEvents([])
    setReasoning(undefined)
    setResults(undefined)
    setResultCatalog(undefined)
    setMemory(undefined)
    setTokenUsage(EMPTY_TOKEN_USAGE)
    setStatus(undefined)
    setRunDetail(undefined)
    setStreamState('connecting')
    knownMessageIds.current.clear()
    const visibleMessages = preserveCurrentMessages ? messagesRef.current : cachedMessages
    visibleMessages.forEach((message) => knownMessageIds.current.add(message.messageId))

    void (async () => {
      try {
        const [detail, conversation, eventData, reasoningData, catalog] = await Promise.all([
          getRun(activeConversationId),
          getConversation(activeConversationId),
          getEvents(activeConversationId, { limit: 300 }),
          getReasoning(activeConversationId),
          getResultCatalog(activeConversationId),
        ])
        if (cancelled) return
        setStatus(detail.status)
        setRunDetail(detail)
        mergeMessages(conversation.messages)
        setEvents(eventData.events)
        setReasoning(reasoningData)
        setResults(detail.results)
        setResultCatalog(catalog)
        setMemory(conversation.memory)
        setTokenUsage(conversation.tokenUsage)
      } catch (error) {
        if (!cancelled) setLoadError(error instanceof Error ? error.message : String(error))
      }
    })()

    const scheduleReasoningRefresh = (): void => {
      window.clearTimeout(reasoningTimer)
      reasoningTimer = window.setTimeout(() => {
        void getReasoning(activeConversationId)
          .then((data) => {
            if (!cancelled) setReasoning(data)
          })
          .catch(() => undefined)
      }, 250)
    }
    const source = openRunStream(activeConversationId, {
      onOpen: () => setStreamState('live'),
      onSnapshot: (data) => {
        setStatus(data.status)
        setStreamState('live')
      },
      onStatus: (data) => {
        setStatus(data.status)
        const terminal = data.status.trainingStatus === 'completed' || data.status.trainingStatus === 'failed'
        if (!terminal) observedTerminalTrainingRef.current = undefined
        const terminalKey = terminal ? `${activeConversationId}:${data.status.trainingStatus}:${data.status.trainingPercent ?? ''}` : undefined
        if (terminalKey && observedTerminalTrainingRef.current !== terminalKey) {
          observedTerminalTrainingRef.current = terminalKey
          void refreshRun()
        }
      },
      onEvents: (data) => {
        setEvents((current) => {
          const merged = new Map(current.map(event => [event.id, event]))
          for (const event of data.events) merged.set(event.id, event)
          return [...merged.values()].sort(
            (left, right) => left.timestamp.localeCompare(right.timestamp),
          )
        })
        scheduleReasoningRefresh()
      },
      onMessages: (data) => {
        const latest = [...data.messages].reverse().find((message) => message.role === 'assistant' && !message.messageKind.startsWith('activity.'))
        const queueId = processingMessageRef.current
        const receivedNewAssistant = latest != null && !knownMessageIds.current.has(latest.messageId)
        if (latest) setLiveAssistantMessageId(latest.messageId)
        mergeMessages(data.messages)
        void getResultCatalog(activeConversationId).then(setResultCatalog).catch(() => undefined)
        // The message request owns completion; stream snapshots may contain earlier replies.
      },
      onError: () => setStreamState('reconnecting'),
    })
    return () => {
      cancelled = true
      window.clearTimeout(reasoningTimer)
      source.close()
    }
  }, [activeConversationId, activeScopeKey, finishQueuedMessage, mergeMessages, projectIdFor, refreshRun])

  const enqueue = useCallback((text: string, nextAttachments: WebAttachment[], modelPreference?: string, projectId = activeProjectId) => {
    const id = crypto.randomUUID()
    const scopeKey = activeScopeRef.current
    const displayImmediately = processingMessageRef.current == null && queuedSubmissionCountRef.current === 0
    queuedSubmissionCountRef.current += 1
    const optimisticMessageId = displayImmediately ? `optimistic.${id}` : undefined
    if (optimisticMessageId) {
      knownMessageIds.current.add(optimisticMessageId)
      if (nextAttachments.length > 0) knownMessageIds.current.add(`${optimisticMessageId}.attachments`)
      setMessages((current) => [
        ...current,
        ...optimisticMessages(
          id,
          text,
          nextAttachments,
          Math.max(0, ...current.map((message) => message.sequenceNumber)) + 1,
        ),
      ])
    }
    setQueued((current) => [...current, {
      id,
      text,
      attachments: nextAttachments,
      ...(modelPreference ? { modelPreference } : {}),
      scopeKey,
      analysisMode,
      projectId,
      ...(optimisticMessageId ? { optimisticMessageId } : {}),
      ...(selectedRunId ? { runId: selectedRunId } : {}),
      ...(workspaceSessionId ? { workspaceSessionId } : {}),
    }])
  }, [activeProjectId, analysisMode, optimisticMessages, selectedRunId, workspaceSessionId])

  const sendMessage = useCallback(async (text: string, nextAttachments: WebAttachment[], modelPreference?: string) => {
    if (!requireAuthentication()) return
    if (queuedSubmissionCountRef.current > 0 || processingMessageRef.current != null) return
    let projectId: string | undefined
    try {
      projectId = await ensureProject(conversationTitleFromText(text))
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error)
      const id = crypto.randomUUID()
      const createdAt = new Date().toISOString()
      const sequenceNumber = Math.max(0, ...messagesRef.current.map((message) => message.sequenceNumber)) + 1
      const failedMessage = {
        messageId: `local.error.${crypto.randomUUID()}`,
        role: 'assistant' as const,
        messageKind: 'conversation.error' as const,
        content: locale === 'zh-CN'
          ? `抱歉，无法创建项目或连接本地 THETA 服务。\n\n原因：${detail}\n\n请确认后端服务已在 127.0.0.1:4318 启动后重试。`
          : `Sorry, the project could not be created or the local THETA service could not be reached.\n\nReason: ${detail}\n\nStart the backend service on 127.0.0.1:4318 and try again.`,
        sequenceNumber: sequenceNumber + Math.max(1, nextAttachments.length + 1),
        createdAt,
      }
      const failedConversationMessages = [...optimisticMessages(id, text, nextAttachments, sequenceNumber), failedMessage]
      setMessages((current) => {
        const next = [...current, ...failedConversationMessages]
        messagesRef.current = next
        return next
      })
      if (projectId && projectId !== DRAFT_PROJECT_ID) {
        const failedProjectId = projectId
        setProjectMemories((current) => {
          const previous = current[failedProjectId] ?? { messages: [], attachments: [] }
          return {
            ...current,
            [failedProjectId]: {
              ...previous,
              messages: mergeStoredMessages(previous.messages, failedConversationMessages),
              attachments: nextAttachments,
            },
          }
        })
      }
      setLoadError(undefined)
      return
    }

    enqueue(text, nextAttachments, modelPreference, projectId)
  }, [enqueue, ensureProject, locale, optimisticMessages, requireAuthentication])

  useEffect(() => {
    const next = queued[0]
    if (!next || processingMessageRef.current) return
    processingMessageRef.current = next.id
    setProcessingMessageId(next.id)
    setSending(true)
    setLoadError(undefined)
    const controller = new AbortController()
    activeRequestControllerRef.current = controller
    const activeOptimisticMessageId = next.optimisticMessageId ?? `optimistic.${next.id}`
    if (!next.optimisticMessageId && activeScopeRef.current === next.scopeKey) {
      knownMessageIds.current.add(activeOptimisticMessageId)
      if (next.attachments.length > 0) knownMessageIds.current.add(`${activeOptimisticMessageId}.attachments`)
      setMessages((current) => [
        ...current,
        ...optimisticMessages(
          next.id,
          next.text,
          next.attachments,
          Math.max(0, ...current.map((message) => message.sequenceNumber)) + 1,
        ),
      ])
    }
    void (async () => {
      try {
        if (next.runId) {
          const result = await postMessage(next.runId, next.text, true, next.attachments, controller.signal, next.modelPreference)
          cacheProjectResult(next.projectId, result.messages, [
            activeOptimisticMessageId,
            `${activeOptimisticMessageId}.attachments`,
          ], { selectedRunId: next.runId, workspaceSessionId: undefined })
          if (activeProjectIdRef.current === next.projectId) {
            knownMessageIds.current.delete(activeOptimisticMessageId)
            knownMessageIds.current.delete(`${activeOptimisticMessageId}.attachments`)
            setMessages((current) => current.filter((message) =>
              message.messageId !== activeOptimisticMessageId &&
              message.messageId !== `${activeOptimisticMessageId}.attachments` &&
              !isTransientLocalError(message),
            ))
            setStatus(result.status)
            const latest = [...result.messages].reverse().find((message) => message.role === 'assistant' && !message.messageKind.startsWith('activity.'))
            if (latest) {
              setLiveAssistantMessageId(latest.messageId)
            }
            mergeMessages(result.messages)
            setTokenUsage(result.tokenUsage)
          }
          void refreshRuns()
        } else {
          const datasetAttachment = next.attachments.find((attachment) => attachment.kind === 'dataset')
          if (datasetAttachment) {
            const createdRun = await createRun({
              projectId: next.projectId,
              datasetRef: datasetAttachment.id,
              analysisMode: next.analysisMode,
              useLanguageProvider: true,
              ...(next.workspaceSessionId ? { sourceSessionId: next.workspaceSessionId } : {}),
            })
            assignItemToProject(createdRun.runId, next.projectId)
            cacheProjectResult(next.projectId, [], [], { selectedRunId: createdRun.runId, workspaceSessionId: undefined })
            if (activeProjectIdRef.current === next.projectId) {
              activateScope(`run:${createdRun.runId}:${crypto.randomUUID()}`)
              preserveMessagesRunRef.current = createdRun.runId
              setSelectedRunId(createdRun.runId)
              setWorkspaceSessionId(undefined)
              setWorkspaceActivity(undefined)
              setWorkspaceInteraction(undefined)
              setTokenUsage(EMPTY_TOKEN_USAGE)
            }
            const result = await postMessage(createdRun.runId, next.text, true, next.attachments, controller.signal, next.modelPreference)
            cacheProjectResult(next.projectId, result.messages, [
              activeOptimisticMessageId,
              `${activeOptimisticMessageId}.attachments`,
            ], { selectedRunId: createdRun.runId, workspaceSessionId: undefined })
            if (activeProjectIdRef.current === next.projectId) {
              knownMessageIds.current.delete(activeOptimisticMessageId)
              knownMessageIds.current.delete(`${activeOptimisticMessageId}.attachments`)
              setMessages((current) => current.filter((message) =>
                message.messageId !== activeOptimisticMessageId &&
                message.messageId !== `${activeOptimisticMessageId}.attachments` &&
                !isTransientLocalError(message),
              ))
              setStatus(result.status)
              const latest = [...result.messages].reverse().find((message) => message.role === 'assistant' && !message.messageKind.startsWith('activity.'))
              if (latest) {
                setLiveAssistantMessageId(latest.messageId)
              }
              mergeMessages(result.messages)
              setTokenUsage(result.tokenUsage)
            }
            void refreshRuns()
            void refreshWorkspaceSessions()
            return
          }

          let sessionId = next.workspaceSessionId ?? workspaceSessionId
          let result: Awaited<ReturnType<typeof getWorkspaceConversation>>
          if (!sessionId) {
            const created = await createWorkspaceSession(next.projectId, conversationTitleFromText(next.text), next.text, next.analysisMode)
            sessionId = created.sessionId
            assignItemToProject(created.sessionId, next.projectId)
            const createdAt = new Date().toISOString()
            setWorkspaceSessions((current) => [{
              sessionId: created.sessionId,
              projectId: next.projectId,
              title: conversationTitleFromText(next.text),
              messageCount: 1,
              createdAt,
              updatedAt: createdAt,
              pinned: false,
            }, ...current.filter((session) => session.sessionId !== created.sessionId)])
            setProjects((current) => current.map((project) => project.id === next.projectId && !project.runIds.includes(created.sessionId)
              ? { ...project, runIds: [...project.runIds, created.sessionId] }
              : project))
            cacheProjectResult(next.projectId, [], [], { workspaceSessionId: sessionId, selectedRunId: undefined })
            setQueued((current) => current.map((item) =>
              item.scopeKey === next.scopeKey && !item.runId
                ? { ...item, workspaceSessionId: created.sessionId }
                : item,
            ))
            if (activeProjectIdRef.current === next.projectId) {
              setWorkspaceSessionId(sessionId)
              setWorkspaceInteraction(created.interaction)
            }
            result = await postWorkspaceMessage(sessionId, next.text, true, next.attachments, controller.signal, next.modelPreference)
            void refreshWorkspaceSessions()
            void refreshRuns()
          } else {
            result = await postWorkspaceMessage(sessionId, next.text, true, next.attachments, controller.signal, next.modelPreference)
          }
          cacheProjectResult(next.projectId, result.messages, [
            activeOptimisticMessageId,
            `${activeOptimisticMessageId}.attachments`,
          ], result.runId
            ? { selectedRunId: result.runId, workspaceSessionId: undefined }
            : { workspaceSessionId: sessionId, selectedRunId: undefined })
          if (activeProjectIdRef.current === next.projectId) {
            knownMessageIds.current.delete(activeOptimisticMessageId)
            knownMessageIds.current.delete(`${activeOptimisticMessageId}.attachments`)
            setMessages((current) => current.filter((message) =>
              message.messageId !== activeOptimisticMessageId &&
              message.messageId !== `${activeOptimisticMessageId}.attachments` &&
              !isTransientLocalError(message),
            ))
            const latest = [...result.messages].reverse().find((message) => message.role === 'assistant' && !message.messageKind.startsWith('activity.'))
            if (latest) {
              setLiveAssistantMessageId(latest.messageId)
            }
            mergeMessages(result.messages)
            setWorkspaceInteraction(result.interaction)
            setWorkspaceActivity(result.activity)
            setMemory(result.memory)
            setTokenUsage(result.tokenUsage)
            if (result.runId) {
              assignItemToProject(result.runId, next.projectId)
              preserveMessagesRunRef.current = result.runId
              setSelectedRunId(result.runId)
              setStatus(result.status)
              setWorkspaceSessionId(undefined)
            }
          }
          void refreshWorkspaceSessions()
        }
      } catch (error) {
        if (!(error instanceof DOMException && error.name === 'AbortError')) {
          const detail = error instanceof Error ? error.message : String(error)
          if (activeProjectIdRef.current === next.projectId) {
            const createdAt = new Date().toISOString()
            const errorMessage: WebMessage = {
              messageId: `local.error.${crypto.randomUUID()}`,
              role: 'assistant',
              messageKind: 'conversation.error',
              content: locale === 'zh-CN'
                ? `抱歉，这条消息暂时无法处理。\n\n原因：${detail === 'Failed to fetch' ? '无法连接到 THETA 本地服务，请确认服务已启动并允许当前页面访问。' : detail}\n\n你可以检查服务连接后重新发送。`
                : `Sorry, this message could not be processed.\n\nReason: ${detail}\n\nCheck the service connection and try again.`,
              sequenceNumber: 0,
              createdAt,
            }
            knownMessageIds.current.add(errorMessage.messageId)
            setMessages((current) => [...current, { ...errorMessage, sequenceNumber: Math.max(0, ...current.map((message) => message.sequenceNumber)) + 1 }])
          }
          setLoadError(undefined)
        }
      } finally {
        finishQueuedMessage(next.id)
      }
    })()
  }, [activateScope, assignItemToProject, cacheProjectResult, finishQueuedMessage, locale, mergeMessages, optimisticMessages, queued, refreshRuns, refreshWorkspaceSessions, selectedRunId, workspaceSessionId])

  useEffect(() => {
    if (!sending || selectedRunId || !workspaceSessionId) return
    let cancelled = false
    const poll = (): void => {
      const expectedScope = activeScopeRef.current
      void getWorkspaceConversation(workspaceSessionId)
        .then((conversation) => {
          if (cancelled || activeScopeRef.current !== expectedScope) return
          const activeItem = queued.find((item) => item.id === processingMessageRef.current)
          if (activeItem && conversation.messages.some((message) => message.role === 'user' && message.content === activeItem.text)) {
            const optimisticMessageId = activeItem.optimisticMessageId ?? `optimistic.${activeItem.id}`
            knownMessageIds.current.delete(optimisticMessageId)
            setMessages((current) => current.filter((message) => message.messageId !== optimisticMessageId))
          }
          const latest = [...conversation.messages].reverse().find((message) =>
            message.role === 'assistant' && !message.messageKind.startsWith('activity.'),
          )
          const queueId = processingMessageRef.current
          const receivedNewAssistant = latest != null && !knownMessageIds.current.has(latest.messageId)
          if (latest && receivedNewAssistant) setLiveAssistantMessageId(latest.messageId)
          mergeMessages(conversation.messages)
          if (conversation.status) setStatus(conversation.status)
          // The message request owns completion; stream snapshots may contain earlier replies.
          setMemory(conversation.memory)
          setTokenUsage(conversation.tokenUsage)
        })
        .catch(() => undefined)
    }
    poll()
    const timer = window.setInterval(poll, 400)
    return () => { cancelled = true; window.clearInterval(timer) }
  }, [finishQueuedMessage, sending, selectedRunId, workspaceSessionId, mergeMessages, queued])

  const assistantRendered = useCallback((messageId: string): void => {
    const pending = awaitingRenderRef.current
    if (!pending || pending.messageId !== messageId) return
    finishQueuedMessage(pending.queueId)
  }, [finishQueuedMessage])

  const stopSending = useCallback((): void => {
    const runId = selectedRunId ?? workspaceSessionId
    if (runId) {
      void stopRunGeneration(runId).catch((error) => {
        setLoadError(error instanceof Error ? error.message : String(error))
      })
    }
    activeRequestControllerRef.current?.abort()
    const queueId = processingMessageRef.current
    if (queueId) finishQueuedMessage(queueId)
  }, [finishQueuedMessage, selectedRunId, workspaceSessionId])

  useEffect(() => {
    const id = selectedRunId ?? workspaceSessionId
    if (!id) { setAnalysisModeBusy(false); return }
    let stale = false
    setAnalysisModeBusy(true)
    void getRun(id).then(detail => { if (!stale) setLocalAnalysisMode(detail.analysisMode ?? 'topic') })
      .catch(error => { if (!stale) toast.error(error instanceof Error ? error.message : String(error)) })
      .finally(() => { if (!stale) setAnalysisModeBusy(false) })
    return () => { stale = true }
  }, [selectedRunId, workspaceSessionId])

  const changeAnalysisMode = async (mode: 'topic' | 'free') => {
    const id = selectedRunId ?? workspaceSessionId
    if (!id) { setLocalAnalysisMode(mode); return }
    const scope = activeScopeRef.current
    setAnalysisModeBusy(true)
    try { const result = await setAnalysisMode(id, mode); if (activeScopeRef.current === scope) setLocalAnalysisMode(result.analysisMode) }
    catch (error) { toast.error(error instanceof Error ? error.message : String(error)) }
    finally { setAnalysisModeBusy(false) }
  }

  // Completed manual projects enter here only after the server has copied their results.
  useEffect(() => {
    const openConversation = (event: Event): void => {
      const detail = (event as CustomEvent<{ runId?: string; projectId?: string; project?: LocalProject; jobId?: string; model?: string; citations?: Array<{ chartName: string; chartPath: string }>; analyses?: Array<{ chartName: string; analysis: string }> }>).detail
      if (!detail?.runId) return
      if (detail.citations?.length || detail.analyses?.length) sessionStorage.setItem('theta.workspace.pending-citations.v1', JSON.stringify(detail))
      if (detail.project) setProjects(current => [detail.project!, ...current.filter(project => project.id !== detail.project!.id)])
      if (detail.projectId) {
        setActiveProjectId(detail.projectId)
        activeProjectIdRef.current = detail.projectId
        setProjectAssignments(current => ({ ...current, [detail.runId!]: detail.projectId! }))
        setExpandedProjectIds(current => new Set([...current, detail.projectId!]))
      }
      setResultDestination(null)
      setWorkspaceMode('conversation')
      activateScope(`run:${detail.runId}:${crypto.randomUUID()}`)
      setMessages([])
      setAttachments([])
      setStatus(undefined)
      setRunDetail(undefined)
      setResults(undefined)
      setResultCatalog(undefined)
      setEvents([])
      setReasoning(undefined)
      setWorkspaceInteraction(undefined)
      setWorkspaceActivity(undefined)
      setMemory(undefined)
      setTokenUsage(EMPTY_TOKEN_USAGE)
      knownMessageIds.current.clear()
      setSelectedRunId(detail.runId)
      setWorkspaceSessionId(undefined)
      setDetailOpen(true)
      void refreshRuns()
    }
    window.addEventListener('theta:open-conversation', openConversation)
    return () => window.removeEventListener('theta:open-conversation', openConversation)
  }, [activateScope, refreshRuns])

  const startNewConversation = useCallback((projectId = activeProjectId) => {
    setResultDestination(null)
    setWorkspaceMode('conversation')
    setLocalAnalysisMode('topic')
    activateScope(`draft:${crypto.randomUUID()}`)
    setActiveProjectId(projectId)
    activeProjectIdRef.current = projectId
    setSelectedRunId(undefined)
    setWorkspaceSessionId(undefined)
    setMessages([])
    setEvents([])
    setReasoning(undefined)
    setResults(undefined)
    setMemory(undefined)
    setTokenUsage(EMPTY_TOKEN_USAGE)
    setStatus(undefined)
    setRunDetail(undefined)
    setWorkspaceActivity(undefined)
    setWorkspaceInteraction(undefined)
    setAttachments([])
    setLiveAssistantMessageId(undefined)
    setLoadError(undefined)
    knownMessageIds.current.clear()
  }, [activateScope, activeProjectId, activeConversationId])

  const restoreLocalProject = useCallback((projectId: string): void => {
    const snapshot = projectMemoriesRef.current[projectId]
    const cachedMessages = withoutTransientLocalErrors(snapshot?.messages ?? [])
    activateScope(`draft:${projectId}:${crypto.randomUUID()}`)
    setActiveProjectId(projectId)
    setSelectedRunId(undefined)
    setWorkspaceSessionId(undefined)
    setMessages(cachedMessages)
    setEvents([])
    setReasoning(undefined)
    setResults(undefined)
    setMemory(undefined)
    setTokenUsage(EMPTY_TOKEN_USAGE)
    setStatus(undefined)
    setRunDetail(undefined)
    setWorkspaceActivity(undefined)
    setWorkspaceInteraction(snapshot?.interaction)
    setAttachments(snapshot?.attachments ?? [])
    setLiveAssistantMessageId(undefined)
    setLoadError(undefined)
    knownMessageIds.current.clear()
    cachedMessages.forEach((message) => knownMessageIds.current.add(message.messageId))
  }, [activateScope])

  const selectWorkspaceHistory = async (sessionId: string): Promise<void> => {
    const scopeKey = `workspace:${sessionId}:${crypto.randomUUID()}`
    const projectId = projectIdFor(sessionId)
    const snapshot = projectMemoriesRef.current[projectId]
    const conversationSnapshot = snapshot?.conversations?.[sessionId]
    const snapshotMatches = conversationSnapshot != null || snapshot?.workspaceSessionId === sessionId
    const cachedMessages = snapshotMatches
      ? withoutTransientLocalErrors(conversationSnapshot?.messages ?? snapshot?.messages ?? [])
      : []
    activateScope(scopeKey)
    setActiveProjectId(projectId)
    setSelectedRunId(undefined)
    setWorkspaceSessionId(sessionId)
    setMessages(cachedMessages)
    setStatus(undefined)
    setRunDetail(undefined)
    setEvents([])
    setReasoning(undefined)
    setResults(undefined)
    setWorkspaceActivity(undefined)
    setWorkspaceInteraction(undefined)
    setAttachments(snapshotMatches ? conversationSnapshot?.attachments ?? snapshot?.attachments ?? [] : [])
    setTokenUsage(EMPTY_TOKEN_USAGE)
    knownMessageIds.current.clear()
    cachedMessages.forEach((message) => knownMessageIds.current.add(message.messageId))
    try {
      const conversation = await getWorkspaceConversation(sessionId)
      if (activeScopeRef.current !== scopeKey) return
      setWorkspaceSessionId(sessionId)
      setWorkspaceInteraction(conversation.interaction)
      setMemory(conversation.memory)
      setTokenUsage(conversation.tokenUsage)
      mergeMessages(conversation.messages)
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : String(error))
    }
  }

  const selectRunHistory = (run: Pick<WebRunSummary, 'runId'>, projectId: string): void => {
    const snapshot = projectMemoriesRef.current[projectId]
    const conversationSnapshot = snapshot?.conversations?.[run.runId]
    const snapshotMatches = conversationSnapshot != null || snapshot?.selectedRunId === run.runId
    const cachedMessages = snapshotMatches
      ? withoutTransientLocalErrors(conversationSnapshot?.messages ?? snapshot?.messages ?? [])
      : []
    activateScope(`run:${run.runId}:${crypto.randomUUID()}`)
    setActiveProjectId(projectId)
    setWorkspaceSessionId(undefined)
    setMessages(cachedMessages)
    setAttachments(snapshotMatches ? conversationSnapshot?.attachments ?? snapshot?.attachments ?? [] : [])
    setWorkspaceActivity(undefined)
    setWorkspaceInteraction(conversationSnapshot?.interaction ?? snapshot?.interaction)
    setSelectedRunId(run.runId)
    knownMessageIds.current.clear()
    cachedMessages.forEach((message) => knownMessageIds.current.add(message.messageId))
  }

  const openProject = (project: LocalProject): void => {
    const snapshot = projectMemoriesRef.current[project.id]
    const projectRunIds = new Set([
      ...project.runIds,
      ...Object.keys(snapshot?.conversations ?? {}),
    ])
    const projectRuns = runs
      .filter((run) => run.projectId === project.id || project.runIds.includes(run.runId))
      .sort((left, right) => Date.parse(right.updatedAt) - Date.parse(left.updatedAt))
    if (snapshot?.selectedRunId && projectRunIds.has(snapshot.selectedRunId)) {
      selectRunHistory({ runId: snapshot.selectedRunId }, project.id)
      return
    }
    if (snapshot?.workspaceSessionId && projectRunIds.has(snapshot.workspaceSessionId)) {
      void selectWorkspaceHistory(snapshot.workspaceSessionId)
      return
    }
    const session = workspaceSessions.find((item) => item.projectId === project.id || projectRunIds.has(item.sessionId))
    if (session) {
      void selectWorkspaceHistory(session.sessionId)
      return
    }
    const run = projectRuns[0]
    if (run) {
      selectRunHistory(run, project.id)
      return
    }
    const storedRunId = [...projectRunIds].at(-1)
    if (storedRunId) {
      selectRunHistory({ runId: storedRunId }, project.id)
      return
    }
    restoreLocalProject(project.id)
  }

  useEffect(() => {
    if (authLoading || runsLoading || navigationReadyKey === navigationKey) return
    // Read after account hydration; the anonymous storage key is not the user's.
    const runId = sessionStorage.getItem(navigationKey)
    const projectId = sessionStorage.getItem(`${navigationKey}:project`)
    setNavigationReadyKey(navigationKey)
    if (activeProjectIdRef.current !== DRAFT_PROJECT_ID) return
    const run = runs.find((item) => item.runId === runId)
    if (run) selectRunHistory(run, run.projectId)
    else if (projectId && projects.some(project => project.id === projectId)) restoreLocalProject(projectId)
    else sessionStorage.removeItem(navigationKey)
  }, [authLoading, runsLoading, runs, navigationKey, navigationReadyKey, projects])

  useEffect(() => {
    if (authLoading || runsLoading || navigationReadyKey !== navigationKey) return
    const runId = selectedRunId ?? workspaceSessionId
    if (runId) sessionStorage.setItem(navigationKey, runId)
    else sessionStorage.removeItem(navigationKey)
    sessionStorage.setItem(`${navigationKey}:project`, activeProjectId)
  }, [authLoading, runsLoading, navigationReadyKey, navigationKey, activeProjectId, selectedRunId, workspaceSessionId])

  const deleteProject = async (): Promise<void> => {
    const target = projectToDelete
    if (!target || projectDeleteBusy) return
    setProjectDeleteBusy(true)
    setProjectDeleteError(undefined)
    try {
      await deleteProjectRequest(target.id)
      const targetRunIds = new Set([
        ...target.runIds,
        ...runs.filter((run) => run.projectId === target.id).map((run) => run.runId),
        ...workspaceSessions.filter((session) => session.projectId === target.id).map((session) => session.sessionId),
        ...Object.entries(projectAssignments)
          .filter(([, projectId]) => projectId === target.id)
          .map(([itemId]) => itemId),
      ])
      await Promise.all([...targetRunIds].map((runId) => deleteRun(runId)))
      setProjects((current) => current.filter((project) => project.id !== target.id))
      setProjectAssignments((current) => Object.fromEntries(
        Object.entries(current).filter(([, projectId]) => projectId !== target.id),
      ))
      setProjectMemories((current) => {
        const next = { ...current }
        delete next[target.id]
        return next
      })
      setProjectToDelete(undefined)
      if (activeProjectId === target.id) startNewConversation(DRAFT_PROJECT_ID)
    } catch (error) {
      setProjectDeleteError(error instanceof Error ? error.message : String(error))
    } finally {
      setProjectDeleteBusy(false)
    }
  }

  const renameProject = async (): Promise<void> => {
    const target = projectToRename
    const displayName = projectRenameName.trim().replace(/\s+/gu, ' ')
    if (!target || !displayName || projectRenameBusy) return
    setProjectRenameBusy(true)
    setProjectRenameError(undefined)
    try {
      await renameProjectRequest(target.id, displayName)
      setProjects((current) => current.map((project) =>
        project.id === target.id ? { ...project, name: displayName } : project,
      ))
      setProjectToRename(undefined)
      setProjectRenameName('')
    } catch (error) {
      setProjectRenameError(error instanceof Error ? error.message : String(error))
    } finally {
      setProjectRenameBusy(false)
    }
  }

  const toggleProjectPin = async (project: LocalProject): Promise<void> => {
    setOpenProjectMenuId(undefined)
    try {
      const updated = await pinProjectRequest(project.id, !project.pinned)
      setProjects((current) => current.map((item) => item.id === project.id
        ? { ...item, pinned: updated.pinned }
        : item))
      toast.success(locale === 'zh-CN'
        ? (updated.pinned ? '项目已置顶' : '已取消项目置顶')
        : (updated.pinned ? 'Project pinned' : 'Project unpinned'))
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      setLoadError(message)
      toast.error(message)
    }
  }

  const submitCreateProject = async (): Promise<void> => {
    const name = projectCreateName.trim().replace(/\s+/gu, ' ')
    if (!name || projectCreateBusy || !requireAuthentication()) return
    setProjectCreateBusy(true)
    setProjectCreateError(undefined)
    try {
      const created = await createProjectRequest(name)
      setProjects((current) => current.some((project) => project.id === created.id)
        ? current
        : [...current, { id: created.id, name: created.name, createdAt: created.createdAt, pinned: created.pinned, runIds: created.runIds }])
      setExpandedProjectIds((current) => new Set(current).add(created.id))
      setProjectCreateOpen(false)
      setProjectCreateName('')
      startNewConversation(created.id)
      toast.success(locale === 'zh-CN' ? '项目已创建' : 'Project created')
    } catch (error) {
      setProjectCreateError(error instanceof Error ? error.message : String(error))
    } finally {
      setProjectCreateBusy(false)
    }
  }

  const submitCreateConversation = (targetProjectId = activeProjectId): void => {
    if (!requireAuthentication()) return
    if (targetProjectId !== DRAFT_PROJECT_ID) {
      setExpandedProjectIds((current) => new Set(current).add(targetProjectId))
    }
    startNewConversation(targetProjectId)
  }

  const selectedRun = useMemo(
    () => runs.find((run) => run.runId === selectedRunId),
    [runs, selectedRunId],
  )
  const activeProject = useMemo(
    () => projects.find((project) => project.id === activeProjectId),
    [activeProjectId, projects],
  )
  const sortedProjects = useMemo(
    () => [...projects].sort((left, right) => {
      if (left.pinned !== right.pinned) return left.pinned ? -1 : 1
      return Date.parse(right.createdAt) - Date.parse(left.createdAt)
    }),
    [projects],
  )
  const conversationsByProject = useMemo(() => {
    const grouped = new Map<string, WebWorkspaceSummary[]>()
    for (const session of workspaceSessions) {
      if (!session.projectId) continue
      const existing = grouped.get(session.projectId) ?? []
      existing.push(session)
      grouped.set(session.projectId, existing)
    }
    for (const sessions of grouped.values()) {
      sessions.sort((left, right) => Date.parse(right.updatedAt) - Date.parse(left.updatedAt))
    }
    return grouped
  }, [workspaceSessions])
  const processingItem = queued.find((item) => item.id === processingMessageId)
  const activeSending = processingItem?.projectId === activeProjectId
  const conversationInputDisabled = (processingMessageId != null || queued.length > 0) && !activeSending

  const toggleSidebar = (): void => {
    setSidebarOpen((current) => {
      if (!current && window.matchMedia('(max-width: 900px)').matches) setDetailOpen(false)
      return !current
    })
  }

  const startSidebarResize = (event: React.PointerEvent<HTMLDivElement>): void => {
    event.preventDefault()
    event.currentTarget.setPointerCapture(event.pointerId)
    sidebarResizeStartRef.current = { pointerX: event.clientX, width: sidebarWidth }
    setSidebarResizing(true)
  }

  const resizeSidebar = (event: React.PointerEvent<HTMLDivElement>): void => {
    const start = sidebarResizeStartRef.current
    if (start == null) return
    const nextWidth = start.width + event.clientX - start.pointerX
    if (nextWidth <= SIDEBAR_COLLAPSE_THRESHOLD) {
      setSidebarOpen(false)
      return
    }
    setSidebarOpen(true)
    setSidebarWidth(Math.min(MAX_SIDEBAR_WIDTH, Math.max(MIN_SIDEBAR_WIDTH, nextWidth)))
  }

  const stopSidebarResize = (): void => {
    sidebarResizeStartRef.current = undefined
    setSidebarResizing(false)
  }

  const closeProjectMenu = (projectId: string): void => {
    setOpenProjectMenuId((current) => current === projectId ? undefined : current)
  }

  const selectProject = (project: LocalProject): void => {
    if (!requireAuthentication()) return
    setOpenProjectMenuId(undefined)
    const isExpanded = expandedProjectIds.has(project.id)
    setExpandedProjectIds((current) => {
      const next = new Set(current)
      if (next.has(project.id)) next.delete(project.id)
      else next.add(project.id)
      return next
    })
    if (isExpanded) return
    if (activeProjectId === project.id && messagesRef.current.length > 0) return
    openProject(project)
  }

  const toggleDetail = (): void => {
    setDetailOpen((current) => {
      if (!current && window.matchMedia('(max-width: 900px)').matches) setSidebarOpen(false)
      return !current
    })
  }

  const inspectorAvailable = workspaceMode === 'conversation'
  const activeTitle = activeProject?.name ?? selectedRun?.identity?.displayName ?? selectedRun?.presentation?.title ??
    workspaceSessions.find((session) => session.sessionId === workspaceSessionId)?.title ??
    t('newChat')

  const datasetReady = async (datasets: WebDataset[]): Promise<void> => {
    const dataset = datasets[0]
    if (!dataset) return
    const projectId = dataset.projectId ?? await ensureProject(dataset.name.replace(/\.[^.]+$/u, '') || '数据分析项目')

    // An existing run retains its original dataset and approvals. Replacement starts
    // a fresh conversation inside the same project, leaving its history/results intact.
    const replacing = !!(selectedRunId || workspaceSessionId) && !attachments.some(item => item.kind === 'dataset' && item.id === dataset.datasetRef)
    if (replacing && activeProjectIdRef.current === projectId) startNewConversation(projectId)
    setDatasetSizes(current => ({ ...current, [projectId]: dataset.sizeBytes }))
    const datasetAttachments: WebAttachment[] = [{ kind: 'dataset', id: dataset.datasetRef, label: dataset.name }]
    if (activeProjectIdRef.current === projectId) setAttachments(datasetAttachments)
    setProjectMemories((current) => {
      const previous = current[projectId] ?? { messages: messagesRef.current, attachments: [] }
      return { ...current, [projectId]: { ...previous, ...(replacing ? { messages: [], selectedRunId: undefined, workspaceSessionId: undefined } : {}), attachments: datasetAttachments } }
    })
    if (activeProjectIdRef.current === projectId) setDetailOpen(true)
  }

  const uploadDatasetFromDrawer = async (files: File[]): Promise<void> => {
    const projectId = await ensureProject(files[0].name.replace(/\.[^.]+$/u, '') || '数据分析项目')
    const dataset = await uploadDatasetFiles(projectId, files, locale)
    await datasetReady([dataset])
  }

  return (
    <div className={css.shell}>
      <div className={css.body}>
        {sidebarOpen && workspaceMode === 'conversation' && !resultDestination && (
          <aside
            className={`${css.sidebar} ${sidebarResizing ? css.sidebarResizing : ''}`}
            style={{ width: sidebarWidth, minWidth: sidebarWidth }}
          >
            <div className={css.sidebarBrand}>
              <CatBrandWordmark className={css.sidebarBrandLogo} />
            </div>
            <div className={css.sidebarCreateActions}>
              <button
                type="button"
                className={css.projectCreateButton}
                onClick={() => {
                  if (!requireAuthentication()) return
                  setProjectCreateError(undefined)
                  setProjectCreateName('')
                  setProjectCreateOpen(true)
                }}
              >
                <img className={css.newProjectFolderIcon} src="/ui/project-folder-new.png" alt="" aria-hidden="true" />
                <span>{locale === 'zh-CN' ? '新建项目' : 'New project'}</span>
              </button>
              <button
                type="button"
                className={css.conversationCreateButton}
                onClick={() => void submitCreateConversation()}
              >
                <img className={css.newConversationIcon} src="/ui/workspace-conversation.png" alt="" aria-hidden="true" />
                <span>{locale === 'zh-CN' ? '新建对话' : 'New conversation'}</span>
              </button>
            </div>
            <div className={css.projectSectionHeading}>
              <span>{locale === 'zh-CN' ? '项目' : 'Projects'}</span>
              <small>{sortedProjects.length}</small>
            </div>
            <nav className={css.projectList} aria-label={locale === 'zh-CN' ? '项目' : 'Projects'}>
              {runsLoading && <div className={css.projectSyncing}>{locale === 'zh-CN' ? '正在同步项目…' : 'Syncing projects…'}</div>}
              {sortedProjects.map((project) => {
                const projectConversations = conversationsByProject.get(project.id) ?? []
                const projectExpanded = expandedProjectIds.has(project.id)
                return <div
                  key={project.id}
                  className={`${css.projectNode} ${activeProjectId === project.id ? css.projectNodeActive : ''} ${openProjectMenuId === project.id ? css.projectNodeMenuOpen : ''}`}
                  onMouseLeave={() => closeProjectMenu(project.id)}
                >
                  <div className={css.projectRow}>
                    <button
                      type="button"
                      className={css.projectEntry}
                      aria-current={activeProjectId === project.id ? 'page' : undefined}
                      aria-expanded={projectExpanded}
                      onClick={() => selectProject(project)}
                      title={project.name}
                    >
                      <img
                        className={css.projectIcon}
                        src="/ui/project-folder-new.png"
                        alt=""
                        aria-hidden="true"
                      />
                      <span>{project.name}</span>
                      {project.pinned && (
                        <span className={css.projectPinnedMark} aria-hidden="true">
                          <svg viewBox="0 0 16 16"><path d="M5 2h6l-1 4 2 2v1H9v5l-1 1-1-1V9H4V8l2-2-1-4Z" /></svg>
                        </span>
                      )}
                      <IconChevronDownOutline14 className={`${css.projectExpandChevron} ${projectExpanded ? css.projectExpandChevronOpen : ''}`} />
                    </button>
                    <div className={`${css.projectMenu} ${openProjectMenuId === project.id ? css.projectMenuOpen : ''}`}>
                      <button
                        type="button"
                        className={css.projectMenuTrigger}
                        aria-expanded={openProjectMenuId === project.id}
                        aria-label={`${project.name} - ${locale === 'zh-CN' ? '项目操作' : 'Project actions'}`}
                        onClick={() => setOpenProjectMenuId((current) => current === project.id ? undefined : project.id)}
                      >
                        <IconEllipsisOutline16 />
                      </button>
                      {openProjectMenuId === project.id && (
                        <div className={css.projectMenuPopover} onMouseLeave={() => closeProjectMenu(project.id)}>
                          <button
                            type="button"
                            className={css.projectRenameAction}
                            onClick={() => void toggleProjectPin(project)}
                          >
                            <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="M5 2h6l-1 4 2 2v1H9v5l-1 1-1-1V9H4V8l2-2-1-4Z" /></svg>
                            {locale === 'zh-CN' ? (project.pinned ? '取消置顶' : '置顶项目') : (project.pinned ? 'Unpin project' : 'Pin project')}
                          </button>
                          <button
                            type="button"
                            className={css.projectRenameAction}
                            onClick={() => {
                              setOpenProjectMenuId(undefined)
                              setProjectRenameError(undefined)
                              setProjectRenameName(project.name)
                              setProjectToRename(project)
                            }}
                          >
                            <IconEditOutline16 />{locale === 'zh-CN' ? '修改名称' : 'Rename'}
                          </button>
                          <button type="button" onClick={() => { setOpenProjectMenuId(undefined); setProjectDeleteError(undefined); setProjectToDelete(project) }}>
                            <IconTrashOutline16 />{locale === 'zh-CN' ? '删除项目' : 'Delete project'}
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                  {projectExpanded && <div className={css.projectConversationList}>
                    {projectConversations.map((session) => {
                      const active = workspaceSessionId === session.sessionId || selectedRunId === session.sessionId
                      return (
                        <div
                          key={session.sessionId}
                          className={`${css.conversationNode} ${active ? css.conversationNodeActive : ''}`}
                        >
                          <button
                            type="button"
                            className={css.conversationEntry}
                            aria-current={active ? 'page' : undefined}
                            onClick={() => void selectWorkspaceHistory(session.sessionId)}
                            title={session.title}
                          >
                            <span>{session.title}</span>
                            <small>{session.messageCount}</small>
                          </button>
                        </div>
                      )
                    })}
                    <div className={css.conversationNode}>
                      <button
                        type="button"
                        className={css.conversationEntry}
                        onClick={() => void submitCreateConversation(project.id)}
                      >
                        <span>{locale === 'zh-CN' ? '新建对话' : 'New conversation'}</span>
                        <small>＋</small>
                      </button>
                    </div>
                  </div>}
                </div>
              })}
            </nav>
            <div className={css.sidebarFooter}>
              {OPEN_SOURCE_EDITION ? <span className={css.sidebarIdentity}><strong>THETA Open Source</strong><small>{locale === 'zh-CN' ? '本地工作台' : 'Local workbench'}</small></span> : <div className={css.accountMenu} data-account-menu>
                <button
                  type="button"
                  className={css.accountMenuTrigger}
                  aria-expanded={accountMenuOpen}
                  aria-haspopup="menu"
                  onClick={() => setAccountMenuOpen((current) => !current)}
                >
                  <span className={css.userAvatar}>U</span>
                  <span className={css.sidebarIdentity}>
                    <strong>{user?.username ?? accountName}</strong>
                    <small>{locale === 'zh-CN' ? '账号管理' : 'Account management'}</small>
                  </span>
                  <IconChevronDownOutline14 className={css.accountMenuChevron} />
                </button>
                {accountMenuOpen && (
                  <div className={css.accountMenuPopover} role="menu">
                    <button
                      type="button"
                      role="menuitem"
                      onClick={() => {
                        setAccountMenuOpen(false)
                        toast.info(locale === 'zh-CN' ? '该功能正在开发中' : 'This feature is under development', { duration: 5000 })
                      }}
                    >
                      {locale === 'zh-CN' ? '账号详情' : 'Account details'}
                    </button>
                    <button type="button" role="menuitem" onClick={() => { void logout() }}>
                      {locale === 'zh-CN' ? '退出登录' : 'Log out'}
                    </button>
                  </div>
                )}
              </div>}
              <button type="button" className={css.sidebarSettings} onClick={() => setSettingsOpen(true)} aria-label={locale === 'zh-CN' ? '打开设置' : 'Open settings'}>
                <IconSettingsOutline16 />
              </button>
            </div>
            <div
              className={css.sidebarResizeHandle}
              role="separator"
              aria-label={locale === 'zh-CN' ? '调整侧边栏宽度' : 'Resize sidebar'}
              aria-orientation="vertical"
              onPointerDown={startSidebarResize}
              onPointerMove={resizeSidebar}
              onPointerUp={stopSidebarResize}
              onPointerCancel={stopSidebarResize}
              onDoubleClick={() => setSidebarWidth(DEFAULT_SIDEBAR_WIDTH)}
            />
          </aside>
        )}

        <main className={css.center}>
          <div className={css.workspaceModeBar}>
            <div className={css.workspaceNavigation}>
              {workspaceMode === 'conversation' && !sidebarOpen && !resultDestination && (
                <Button
                  size="sm"
                  variant="ghost"
                  className={css.iconButton}
                  aria-label={locale === 'zh-CN' ? '展开侧边栏' : 'Show sidebar'}
                  onClick={toggleSidebar}
                >
                  <IconPanelLeftOutline16 />
                </Button>
              )}
              <a className={css.homeLink} href={HOME_URL} aria-label={locale === 'zh-CN' ? '返回首页' : 'Back to home'}>
                <IconChevronLeftOutline14 />
                <span>{locale === 'zh-CN' ? '返回首页' : 'Home'}</span>
              </a>
            </div>
            <div className={css.workspaceModeSwitch} role="group" aria-label="工作台模式">
              <button type="button" aria-pressed={workspaceMode === 'conversation'} className={workspaceMode === 'conversation' ? css.workspaceModeActive : undefined} onClick={() => switchWorkspace('conversation')}>
                <span className={css.workspaceModeIcon} aria-hidden="true">
                  <img src="/ui/workspace-conversation.png" alt="" />
                </span>
                <span className={css.workspaceModeCopy}>
                  <strong>对话</strong>
                  <small>探索、分析与生成</small>
                </span>
              </button>
              <button type="button" aria-pressed={workspaceMode === 'manual'} className={workspaceMode === 'manual' ? css.workspaceModeActive : undefined} onClick={() => switchWorkspace('manual')}>
                <span className={css.workspaceModeIcon} aria-hidden="true">
                  <img src="/ui/workspace-manual.png" alt="" />
                </span>
                <span className={css.workspaceModeCopy}>
                  <strong>手动</strong>
                  <small>流程化精细控制</small>
                </span>
              </button>
            </div>
          </div>
          {workspaceMode === 'conversation' && !resultDestination && (
            <div className={css.centerHeader}>
              {sidebarOpen && (
                <Button size="sm" variant="ghost" className={css.iconButton} aria-label="收起任务列表" onClick={toggleSidebar}>
                  <IconPanelLeftOutline16 />
                </Button>
              )}
              <strong>{activeTitle || t('newChat')}</strong>
              <IconChevronUpOutline14 className={css.headerChevron} />
              <div className={css.headerActions} />
              {inspectorAvailable && (
                <Button size="sm" variant="ghost" className={`${css.iconButton} ${detailOpen ? css.detailToggleActive : ''}`} aria-label={detailOpen ? '收起训练详情' : '展开训练详情'} onClick={toggleDetail}>
                  <IconPanelLeftOutline16 className={css.flipIcon} />
                </Button>
              )}
            </div>
          )}
          {manualVisited && <Activity mode={workspaceMode === 'manual' ? 'visible' : 'hidden'}>
            <div className={css.manualDashboardHost}><ManualWorkbench /></div>
          </Activity>}
          <Activity mode={workspaceMode === 'conversation' && !resultDestination ? 'visible' : 'hidden'}>
          {runsLoading && selectedRunId == null
            ? (
              <div className={css.catalogLoading}>
                <span />
                <strong>正在恢复研究工作区</strong>
                <small>读取本地任务、对话与 Agent 状态</small>
              </div>
            )
            : (
              <ConversationPane
                datasetSizeBytes={datasetSizes[activeProjectId] ?? 0}
                  analysisMode={analysisMode}
                  onAnalysisModeChange={changeAnalysisMode}
                  analysisModeDisabled={analysisModeBusy || queued.length > 0}
                projectStorageScope={activeProjectId}
                projectName={activeProject?.name ?? t('newChat')}
                starterProjectName={activeProject?.name}
                messages={messages}
                sending={activeSending}
                inputDisabled={conversationInputDisabled}
                onSend={sendMessage}
                onStop={stopSending}
                workspaceSessionId={workspaceSessionId}
                entryInteraction={workspaceInteraction ?? runtimeProfile?.entryInteraction}
                workspaceActivity={workspaceActivity}
                runId={activeConversationId}
                status={status}
                resultCatalog={resultCatalog}
                onOpenTrainingPanel={(tab) => {
                  if (tab === 'results') { setResultDestination('results'); return }
                  setDetailTab(tab); setDetailOpen(true)
                  if (window.matchMedia('(max-width: 900px)').matches) setSidebarOpen(false)
                }}
                reasoning={reasoning}
                onApproved={() => void refreshRun()}
                attachments={attachments}
                onAttachmentsChange={setAttachments}
                onEnsureProject={ensureProject}
                onDatasetReady={datasetReady}
                liveAssistantMessageId={liveAssistantMessageId}
                onAssistantRendered={assistantRendered}
                tokenUsage={tokenUsage}
              />
            )}
          </Activity>
          {resultDestination && activeConversationId && <Activity mode={workspaceMode === 'conversation' ? 'visible' : 'hidden'}><ResultWorkspace
            source={{ kind: 'run', runId: activeConversationId, projectId: activeProjectId === DRAFT_PROJECT_ID ? undefined : activeProjectId, projectName: activeTitle, datasetName: attachments.find(item => item.kind === 'dataset')?.label, catalog: resultCatalog }}
            initialDestination={resultDestination}
            onBack={() => setResultDestination(null)}
          /></Activity>}
        </main>

        {workspaceMode === 'conversation' && !resultDestination && detailOpen && inspectorAvailable && (
          <ResizableSidebar storageKey="theta.workspace.research-panel-ratio.v1" label="调整研究助手宽度">
          <DetailPane
            runId={activeConversationId}
            status={status}
            events={events}
            results={results}
            resultCatalog={resultCatalog}
            plan={runDetail?.plan}
            attachments={attachments}
            initialSection={detailTab}
            onCollapse={() => setDetailOpen(false)}
            onOpenResults={setResultDestination}
            onAttach={(attachment) => {
              setAttachments((current) => [...current.filter((item) => item.id !== attachment.id), attachment].slice(-12))
            }}
            onUploadDataset={uploadDatasetFromDrawer}
          />
          </ResizableSidebar>
        )}
      </div>
      <SettingsDialog initialTab={settingsTab} open={settingsOpen} onClose={() => { setSettingsOpen(false); setSettingsTab(undefined) }} onAccountNameChange={setAccountName} />
      <Modal
        open={projectCreateOpen}
        onClose={() => { if (!projectCreateBusy) setProjectCreateOpen(false) }}
        title={locale === 'zh-CN' ? '新建项目' : 'Create project'}
        description={locale === 'zh-CN' ? '创建一个独立项目，用于组织相关对话、数据和运行记录。' : 'Create a project for related conversations, data, and runs.'}
        className={css.createProjectDialog}
        footer={(
          <>
            <Button variant="ghost" disabled={projectCreateBusy} onClick={() => setProjectCreateOpen(false)}>{locale === 'zh-CN' ? '取消' : 'Cancel'}</Button>
            <Button variant="primary" className={css.createProjectConfirm} disabled={projectCreateBusy || !projectCreateName.trim()} onClick={() => void submitCreateProject()}>
              {projectCreateBusy ? '…' : (locale === 'zh-CN' ? '创建' : 'Create')}
            </Button>
          </>
        )}
      >
        <label className={css.createProjectField}>
          <span>{locale === 'zh-CN' ? '项目名称' : 'Project name'}</span>
          <input
            autoFocus
            maxLength={120}
            value={projectCreateName}
            onChange={(event) => setProjectCreateName(event.target.value)}
            onKeyDown={(event) => { if (event.key === 'Enter') void submitCreateProject() }}
          />
        </label>
        {projectCreateError && <p className={css.projectDeleteError} role="alert">{projectCreateError}</p>}
      </Modal>
      <Modal
        open={projectToRename != null}
        onClose={() => { if (!projectRenameBusy) setProjectToRename(undefined) }}
        title={locale === 'zh-CN' ? '修改项目名称' : 'Rename project'}
        description={locale === 'zh-CN' ? '项目名称仅用于左侧项目列表和当前工作台标题。' : 'The name is used in the project list and workspace title.'}
        className={css.createProjectDialog}
        footer={(
          <>
            <Button variant="ghost" disabled={projectRenameBusy} onClick={() => setProjectToRename(undefined)}>{locale === 'zh-CN' ? '取消' : 'Cancel'}</Button>
            <Button variant="primary" className={css.createProjectConfirm} disabled={projectRenameBusy || !projectRenameName.trim()} onClick={() => void renameProject()}>
              {projectRenameBusy ? '…' : (locale === 'zh-CN' ? '保存' : 'Save')}
            </Button>
          </>
        )}
      >
        <label className={css.createProjectField}>
          <span>{locale === 'zh-CN' ? '项目名称' : 'Project name'}</span>
          <input
            autoFocus
            maxLength={120}
            value={projectRenameName}
            onChange={(event) => setProjectRenameName(event.target.value)}
            onKeyDown={(event) => { if (event.key === 'Enter') void renameProject() }}
          />
        </label>
        {projectRenameError && <p className={css.projectDeleteError} role="alert">{projectRenameError}</p>}
      </Modal>
      <Modal
        open={projectToDelete != null}
        onClose={() => { if (!projectDeleteBusy) setProjectToDelete(undefined) }}
        title={locale === 'zh-CN' ? '删除项目' : 'Delete project'}
         description={locale === 'zh-CN' ? '项目及其下的全部对话、消息和运行记录将被永久删除。' : 'The project and all conversations, messages, and run records in it will be permanently deleted.'}
        className={css.createProjectDialog}
        footer={(
          <>
            <Button variant="ghost" disabled={projectDeleteBusy} onClick={() => setProjectToDelete(undefined)}>{locale === 'zh-CN' ? '取消' : 'Cancel'}</Button>
            <Button variant="primary" className={css.projectDeleteConfirm} disabled={projectDeleteBusy} onClick={() => void deleteProject()}>
              {projectDeleteBusy ? '…' : (locale === 'zh-CN' ? '删除' : 'Delete')}
            </Button>
          </>
        )}
      >
        <div className={css.projectDeleteBody}>
          <strong>{projectToDelete?.name}</strong>
          <p>{locale === 'zh-CN' ? '删除后无法恢复，服务端和当前浏览器中的记录都会清除。' : 'This cannot be undone. Server and browser records will be removed.'}</p>
          {projectDeleteError && <p className={css.projectDeleteError} role="alert">{projectDeleteError}</p>}
        </div>
      </Modal>
    </div>
  )
}
