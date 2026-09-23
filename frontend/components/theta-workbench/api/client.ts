/**
 * Typed client for the THETA 2.0 API (apps/api). All shapes mirror
 * agent/src/api/contracts.ts so the web UI and the API stay in sync.
 */

import type { EditorTrainingPlan } from '@/lib/analysis-config'
import type { AnalysisConfigService } from '@/components/project/analysis-config-panel'

export interface WebMessage {
  messageId: string;
  role: 'user' | 'assistant';
  messageKind: string;
  content: string;
  sequenceNumber: number;
  createdAt: string;
}

export interface WebTaskReceipt {
  id: string; status: 'queued' | 'running' | 'waiting_human' | 'completed' | 'failed' | 'cancelled' | 'interrupted';
  progress?: {completed:number;total:number};
  phase: string; objective: string; createdAt: string; updatedAt: string; completedAt?: string; error?: string; lastOperation?: string;
}

export interface WebRunStatus {
  userMessageCount?: number;
  datasetRefs?: string[];
  task?: WebTaskReceipt;
  runId: string;
  status: string;
  currentState?: string;
  datasetRef?: string;
  datasetHash?: string;
  datasetWorkspaceHash?: string;
  planWorkspaceHash?: string;
  candidatePlanHash?: string;
  canonicalPlanId?: string;
  canonicalPlanHash?: string;
  trainingRunId?: string;
  trainingStatus?: string;
  trainingPhase?: string;
  trainingPercent?: number;
  pendingReason?: string;
  pendingActionRef?: string;
  statePath?: string[];
  lastEventAt?: string;
  presentation?: WebPresentation;
  interaction?: WebAgentInteraction;
}

export interface WebAgentInteraction {
  source: 'fsm' | 'agent';
  state: string;
  status: string;
  reasoning: {
    goal: string;
    observation: string;
    decision: string;
    nextStates: string[];
    allowedTools: string[];
    policyRefs: string[];
  };
  card?: {
    kind:
      | 'dataset_upload'
      | 'research_question'
      | 'dataset_review'
      | 'column_review'
      | 'research_intent_review'
      | 'model_selection'
      | 'plan_review'
      | 'training_review'
      | 'embedding_choice'
      | 'action_review';
    cloudAvailable?: boolean;
    trainingEditable?: boolean;
    title: string;
    description: string;
    actionRef: string;
    contentHash?: string;
    requiresHumanAction: true;
  };
}

export interface WebPresentation {
  kind?: string;
  title: string;
  summary: string;
  progress?: { current: number; total: number; label: string; percent?: number };
  sections?: Array<{ title: string; lines: string[] }>;
  nextActions: Array<{
    id: string;
    label: string;
    description: string;
    recommended?: boolean;
    destructive?: boolean;
    command?: string;
  }>;
}

export interface WebRunSummary {
  runId: string;
  projectId: string;
  updatedAt: string;
  status: string;
  currentState?: string;
  pendingReason?: string;
  identity?: { displayName?: string; datasetName?: string; researchQuestion?: string };
  presentation?: WebPresentation;
  pinned?: boolean;
  messageCount?: number;
}

export interface WebRunEvent {
  id: string;
  source: 'orchestration' | 'tool';
  type: string;
  title: string;
  detail?: string;
  timestamp: string;
  payload?: unknown;
}

export interface WebReasoningToolCall {
  eventId: string;
  invocationId?: string;
  toolId: string;
  phase: 'requested' | 'started' | 'policy' | 'completed' | 'failed' | 'validated';
  label: string;
  timestamp: string;
  payload: unknown;
}

export interface WebReasoning {
  runId: string;
  researchIntent?: Record<string, unknown>;
  intentSummary?: Record<string, unknown>;
  currentDecisionGap?: string;
  decisionGaps: Array<{
    question: string;
    answers: Array<{ content: string; createdAt: string }>;
    resolved: boolean;
  }>;
  recommendation?: Record<string, unknown>;
  modelChoices?: Array<{
    modelId: string;
    recommended: boolean;
    rationale: string;
    advantages: string[];
    tradeoffs: string[];
    parameterDefaults: Record<string, string | number | boolean | null>;
  }>;
  parameterConfiguration?: {
    modelId: string;
    parameters: Array<{
      field: string;
      value: string | number | boolean | null;
      rationale: string;
      source: string;
    }>;
  };
  plan?: { state: string; presentation?: WebPresentation };
  toolCalls: WebReasoningToolCall[];
  reasoningEvents: WebRunEvent[];
}

export interface WebRunDetail {
  runId: string;
  analysisMode?: 'topic' | 'free';
  status: WebRunStatus;
  identity?: Record<string, unknown>;
  plan?: Record<string, unknown>;
  results?: WebRunResults;
}

export interface WebResultVisualization {
  id: string;
  label: string;
  relativePath: string;
  format: 'image' | 'interactive';
  scope: 'global' | 'topic';
  topicId?: string;
  sizeBytes: number;
}

export interface WebRunResults {
  status: string;
  progress: number;
  message: string;
  visualizations: WebResultVisualization[];
  metrics: Record<string, unknown>;
  topics: Array<{ id: string; name: string; strength?: number; keywords: string[] }>;
  warnings: string[];
  topicTable?: string;
  researchStatus?: string;
}

export interface WebResultSummary {
  jobId: string;
  runId: string;
  modelId: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
  phase?: string;
  percent?: number;
  selected: boolean;
  reportStatus: 'not_requested' | 'ready' | 'incomplete';
  artifacts?: {
    reportUrl: string;
    fileCount: number;
    figureCount: number;
    tableCount: number;
    matrixCount: number;
    archiveUrl?: string;
    files: Array<{ name: string; kind: string; url: string }>;
  };
}

export interface WebResultCatalog {
  results: WebResultSummary[];
  selection: {
    mode: 'manual' | 'automatic';
    strategy?: 'latest_completed' | 'all_completed';
    resolvedJobIds: string[];
    resolvedAt: string;
  } | null;
}

export interface WebAttachment {
  kind: 'visualization' | 'topic' | 'metric' | 'table' | 'dataset';
  id: string;
  label: string;
}

export interface WebConversationMemory {
  sessionId: string;
  summary: string;
  recentUserGoals: string[];
  sourceMessageCount: number;
  updatedAt: string;
}

export interface WebTokenUsage {
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
  calls: number;
}

export interface WebWorkspaceTurn {
  sessionId: string;
  runId?: string;
  messages: WebMessage[];
  memory?: WebConversationMemory;
  tokenUsage: WebTokenUsage;
  interaction: WebAgentInteraction;
  status?: WebRunStatus;
  activity?: {
    proposal?: unknown;
    semanticDecision?: unknown;
    steps?: unknown;
    result?: unknown;
    evidenceRefs?: unknown;
  };
}

export interface WebWorkspaceSummary {
  sessionId: string;
  projectId?: string;
  title: string;
  messageCount: number;
  createdAt: string;
  updatedAt: string;
  pinned: boolean;
}

export interface WebPostMessageResult {
  runId: string;
  activeRunId: string;
  messages: WebMessage[];
  status: WebRunStatus;
  tokenUsage: WebTokenUsage;
}

export interface WebDataset {
  datasetRef: string;
  name: string;
  sizeBytes: number;
  suffix: string;
  status?: string;
  createdAt: string;
}

export interface WebProject {
  id: string;
  name: string;
  createdAt: string;
  updatedAt: string;
  pinned: boolean;
  runIds: string[];
}

export interface WebDatasetWorkspace {
  workspaceType: 'dataset';
  runId: string;
  datasetHash: string;
  revision: number;
  narrative: string;
  columnRoles: Array<{
    column: string;
    proposedRole: string;
    confidence: number;
    epistemicStatus?: string;
  }>;
  risks: string[];
  workspaceHash: string;
  updatedAt: string;
}

export interface WebResearchWorkspace {
  workspaceType: 'research';
  runId: string;
  datasetHash: string;
  revision: number;
  narrative: string;
  questions: Array<{ id: string; question: string; whyItMatters: string; status: string; blocking: boolean }>;
  preferences: Array<{ id: string; preference: string }>;
  boundaries: Array<{ id: string; boundary: string }>;
  workspaceHash: string;
  updatedAt: string;
}

export interface WebRunWorkspaces {
  runId: string;
  dataset: WebDatasetWorkspace | null;
  research: WebResearchWorkspace | null;
  plan: Record<string, unknown> | null;
}

export interface WebToolInvocation<T = Record<string, unknown>> {
  runId: string;
  state: string;
  toolId: string;
  status: string;
  output: T;
}

export interface WebInferenceProvider {
  id: string;
  displayName: string;
  baseUrl: string;
  credentialConfigured: boolean;
  configured: boolean;
  configuredModel: string | null;
  selected: boolean;
  local: boolean;
  category: 'direct' | 'router' | 'local' | 'compatible';
  models: string[];
  capabilities: {
    streaming: boolean;
    reasoning: boolean;
    reasoningEffort: boolean;
  };
}

export interface WebInferenceCatalog {
  kind: 'inference.provider.list';
  providers: WebInferenceProvider[];
  selection: { providerId: string; model: string; source: string } | null;
}

export type WebReasoningMode = 'auto' | 'chat' | 'reasoning';
export type WebReasoningEffort = 'low' | 'medium' | 'high' | 'xhigh';

export interface WebInferenceSettings {
  readOnly?: boolean;
  llm: {
    providerId: string | null;
    model: string;
    baseUrl: string;
    apiKeyConfigured: boolean;
    reasoningMode: WebReasoningMode;
    reasoningEffort: WebReasoningEffort;
    reasoningBudgetTokens: number | null;
    temperature: number;
    maxTokens: number;
    timeoutMs: number;
    streaming: boolean;
    typewriter: boolean;
    typewriterSpeedMs: number;
  };
  embedding: {
    enabled: boolean;
    providerId: string;
    model: string;
    baseUrl: string;
    dimensions: number | null;
    apiKeyConfigured: boolean;
  };
}

export interface WebInferenceSettingsUpdate {
  llm?: Partial<Omit<WebInferenceSettings['llm'], 'apiKeyConfigured'>> & {
    providerId?: string;
    apiKey?: string;
    clearApiKey?: boolean;
    models?: string[];
  };
  embedding?: Partial<Omit<WebInferenceSettings['embedding'], 'apiKeyConfigured'>> & {
    apiKey?: string;
    clearApiKey?: boolean;
  };
}

export interface WebRuntimeProfile {
  service: 'theta-agent-runtime';
  version: 'v2' | 'v3';
  compute: {
    backend: 'local';
    defaultDevice: 'cpu' | 'gpu';
    scheduler: { supported: false; enabled: false };
  };
  capabilities: {
    domains: number;
    tools: number;
    skills: number;
  };
  entryInteraction: WebAgentInteraction;
}

export interface WebEnvelope<T> {
  ok: boolean;
  data?: T;
  error?: { code: string; message: string };
}

const BASE_URL =
  process.env.NEXT_PUBLIC_THETA_AGENT_API_BASE ??
  '';

const request = async <T>(route: string, init: RequestInit = {}): Promise<T> => {
  const isFormData = init.body instanceof FormData;
  const response = await fetch(`${BASE_URL}${route}`, {
    ...init,
    headers: {
      ...(!isFormData ? { 'Content-Type': 'application/json' } : {}),
      ...(init.headers ?? {}),
    },
  });
  const payload = await response.json().catch(() => undefined) as WebEnvelope<T> | undefined;
  if (payload === undefined) throw new Error(`API returned an invalid response (HTTP ${response.status}).`);
  if (!response.ok || !payload.ok) {
    throw new Error(payload.error?.message ?? `HTTP ${response.status}`);
  }
  if (payload.data === undefined) throw new Error('API response is missing the data field.');
  return payload.data;
};

// The server owns work after acknowledgement. Polling is only a view; aborting
// this browser wait never resubmits or implicitly cancels an approved effect.
const submitBackgroundTurn = async (runId: string, action: '/messages' | '/checkpoint-decision', body: Record<string, unknown>, signal?: AbortSignal) => {
  const accepted = await request<{task:WebTaskReceipt}>(v3Path(runId, action), {method:'POST', body:JSON.stringify({...body, async:true, requestId:crypto.randomUUID()}), signal});
  let task = accepted.task;
  while (task.status === 'queued' || task.status === 'running') {
    await new Promise<void>((resolve, reject) => {
      const abort = () => { clearTimeout(timer); reject(new DOMException('Aborted', 'AbortError')); };
      const timer = setTimeout(() => { signal?.removeEventListener('abort', abort); resolve(); }, 1000);
      signal?.addEventListener('abort', abort, {once:true});
      if (signal?.aborted) abort();
    });
    try { task = await request<WebTaskReceipt>(v3Path(runId, `/tasks/${encodeURIComponent(task.id)}`), {signal}); }
    catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') throw error;
      throw new Error('任务已提交并保存，当前连接中断。刷新对话查看任务记录，先不要重复提交。');
    }
  }
  if (task.status === 'failed') throw new Error(task.error ?? '任务未完成，记录已保存');
  return task;
};

type ApiVersion = 'v2' | 'v3';

export const stopRunGeneration = (runId: string): Promise<{ stopped: boolean }> =>
  request(`/api/v3/runs/${encodeURIComponent(runId)}/stop`, { method: 'POST' });

interface V3RunSnapshot extends Record<string, unknown> {
  task?: WebTaskReceipt;
  runId: string;
  analysisMode?: 'topic' | 'free';
  status: string;
  currentState?: string;
  eventCount: number;
  datasetRef?: string;
  pendingReason?: string;
  pendingActionRef?: string;
  trainingStatus?: string;
  trainingPercent?: number;
  artifactManifestHash?: string;
  lastEventAt?: string;
  lastMessageAt?: string;
  conversationTitle?: string;
  messageCount?: number;
  progress?: { completedGates: number; totalGates: number; percent: number; label: string };
}

interface V3Checkpoint {
  checkpointId: string;
  kind: 'dataset' | 'research' | 'plan' | 'training' | 'action' | 'embedding_choice';
  contentHash: string;
  content: Record<string, unknown>;
  summaryForUser: string;
  warnings: string[];
  view?: { title: string; summary: string; sections: Array<{ title: string; content: string | string[] }> };
}

interface V3Conversation {
  runId: string;
  messageCount: number;
  messages: Array<{ messageId: string; role: 'user' | 'assistant'; content: string; createdAt: string; messageKind?: string }>;
}

interface V3Activity {
  eventId: string;
  kind: string;
  toolId?: string;
  displayName: string;
  userMessage: string;
  status: string;
  startedAt: string;
  completedAt?: string;
  safeInputSummary?: string;
  safeOutputSummary?: string;
}

interface V3ActivitySnapshot {
  runId: string;
  phase?: string;
  current?: V3Activity;
  recent: V3Activity[];
  progress: { completedGates: number; totalGates: number; percent: number; label: string };
}

interface LocalV3Run {
  runId: string;
  title: string;
  updatedAt: string;
  pinned: boolean;
}

const V3_RUNS_STORAGE_KEY = 'theta.frontend.v3-runs.v1';
const v3RunsStorageKey = (): string => {
  try {
    const stored = JSON.parse(localStorage.getItem('user') ?? 'null') as { id?: string | number } | null;
    if (stored?.id !== undefined && stored.id !== null && String(stored.id).trim()) {
      return `${V3_RUNS_STORAGE_KEY}.${encodeURIComponent(String(stored.id))}`;
    }
  } catch {
    // Ignore malformed cached auth state and use the anonymous scope.
  }
  return `${V3_RUNS_STORAGE_KEY}.anonymous`;
};
const detectedApiVersion: ApiVersion = 'v3';
const apiVersion = async (): Promise<ApiVersion> => detectedApiVersion;

const readV3Runs = (): LocalV3Run[] => {
  try {
    const value = JSON.parse(localStorage.getItem(v3RunsStorageKey()) ?? '[]') as unknown;
    return Array.isArray(value) ? value.filter((item): item is LocalV3Run =>
      item != null && typeof item === 'object' && typeof (item as LocalV3Run).runId === 'string') : [];
  } catch {
    return [];
  }
};

const writeV3Runs = (runs: LocalV3Run[]): void =>
  localStorage.setItem(v3RunsStorageKey(), JSON.stringify(runs.slice(0, 50)));

const rememberV3Run = (runId: string, title?: string): void => {
  const current = readV3Runs();
  const previous = current.find((run) => run.runId === runId);
  writeV3Runs([{
    runId,
    title: title?.trim() || previous?.title || 'THETA 研究任务',
    updatedAt: new Date().toISOString(),
    pinned: previous?.pinned ?? false,
  }, ...current.filter((run) => run.runId !== runId)]);
};

const v3Path = (runId: string, suffix = ''): string =>
  `/api/v3/runs/${encodeURIComponent(runId)}${suffix}`;

const v3Checkpoint = async (runId: string): Promise<V3Checkpoint | null> =>
  request<V3Checkpoint | null>(v3Path(runId, '/checkpoint'));

export interface TrainingEditorData { datasetSizeBytes?: number; plans: EditorTrainingPlan[]; datasetName: string; columns: string[]; goal: string; stopwords?: { id: string; name: string; count: number } }
export const getTrainingEditor = (runId: string) => request<TrainingEditorData>(v3Path(runId, '/training-configuration'))
export const submitTrainingEditor = (runId: string, value: { checkpointId: string; expectedContentHash: string; plans: EditorTrainingPlan[]; cloudConfirmed: boolean; cloudSelection?: { provider: string; endpoint: string; model: string }; stopwordsId?: string | number }) =>
  request(v3Path(runId, '/training-configuration'), { method: 'POST', body: JSON.stringify(value) })
export const trainingEditorService = (runId: string): AnalysisConfigService => ({
  runtime: () => request('/api/v3/training-runtime'),
  model: id => request(`/api/v3/training-models/${encodeURIComponent(id)}`),
  uploadStopwords: file => request(v3Path(runId, `/training-stopwords?filename=${encodeURIComponent(file.name)}`), { method: 'POST', headers: { 'Content-Type': 'text/plain; charset=utf-8' }, body: file }),
  stopwordsDownloadUrl: `${BASE_URL}/api/v3/training-stopwords/default`,
})

const checkpointCardKind = (kind: V3Checkpoint['kind']): NonNullable<WebAgentInteraction['card']>['kind'] => {
  const kinds: Record<V3Checkpoint['kind'], NonNullable<WebAgentInteraction['card']>['kind']> = {
    dataset: 'dataset_review',
    research: 'research_intent_review',
    plan: 'plan_review',
    training: 'training_review',
    action: 'action_review',
    embedding_choice: 'embedding_choice',
  };
  return kinds[kind];
};

const v3Interaction = (snapshot: V3RunSnapshot, checkpoint?: V3Checkpoint | null): WebAgentInteraction => ({
  source: 'agent',
  state: snapshot.currentState ?? 'Intake',
  status: snapshot.status,
  reasoning: {
    goal: snapshot.progress?.label ?? '推进 THETA 研究任务',
    observation: snapshot.pendingReason ?? `当前阶段：${snapshot.currentState ?? 'Intake'}`,
    decision: checkpoint ? '等待你确认本次操作。' : '可以继续对话。',
    nextStates: [],
    allowedTools: [],
    policyRefs: [],
  },
  ...(checkpoint ? {
    card: {
      kind: checkpointCardKind(checkpoint.kind),
      title: checkpoint.view?.title ?? `${checkpoint.kind} 检查点`,
      description: checkpoint.view?.summary ?? checkpoint.summaryForUser,
      actionRef: checkpoint.checkpointId,
      contentHash: checkpoint.contentHash,
      cloudAvailable: checkpoint.content.cloudAvailable === true,
      trainingEditable: checkpoint.content.trainingEditable === true,
      requiresHumanAction: true,
    },
  } : {}),
});

const v3Status = (snapshot: V3RunSnapshot, checkpoint?: V3Checkpoint | null): WebRunStatus & Record<string, unknown> => ({
  ...snapshot,
  runId: snapshot.runId,
  status: snapshot.status,
  currentState: snapshot.currentState,
  pendingReason: snapshot.pendingReason,
  pendingActionRef: snapshot.pendingActionRef,
  interaction: v3Interaction(snapshot, checkpoint),
  checkpoint,
  presentation: {
    title: snapshot.currentState ?? 'THETA 研究任务',
    summary: snapshot.pendingReason ?? snapshot.progress?.label ?? 'Agent 正在处理研究流程。',
    progress: snapshot.progress ? {
      current: snapshot.progress.completedGates,
      total: snapshot.progress.totalGates,
      label: snapshot.progress.label,
      percent: snapshot.progress.percent,
    } : undefined,
    nextActions: [],
  },
});

const mapV3Messages = (conversation: V3Conversation): WebMessage[] =>
  conversation.messages.map((message, index) => ({
    messageId: message.messageId,
    role: message.role,
    messageKind: message.messageKind ?? 'conversation.message',
    content: message.content,
    sequenceNumber: index + 1,
    createdAt: message.createdAt,
  }));

const emptyTokenUsage = (): WebTokenUsage => ({ inputTokens: 0, outputTokens: 0, totalTokens: 0, calls: 0 });

const mapV3Event = (event: V3Activity): WebRunEvent => ({
  id: event.eventId,
  source: event.toolId ? 'tool' : 'orchestration',
  type: event.kind,
  title: event.displayName,
  detail: event.safeOutputSummary ?? event.safeInputSummary ?? event.userMessage,
  timestamp: event.completedAt ?? event.startedAt,
  payload: event,
});

export const listRuns = async (): Promise<{ runs: WebRunSummary[] }> => {
  if (await apiVersion() === 'v2') return request('/api/v2/runs?limit=30');
  const data = await request<{ runs: Array<V3RunSnapshot & { projectId: string }> }>('/api/v3/runs');
  const localRuns = new Map(readV3Runs().map((run) => [run.runId, run]));
  return { runs: data.runs.map((snapshot) => {
    const local = localRuns.get(snapshot.runId);
    return {
      runId: snapshot.runId,
      projectId: snapshot.projectId,
      updatedAt: snapshot.lastMessageAt ?? snapshot.lastEventAt ?? local?.updatedAt ?? new Date(0).toISOString(),
      status: snapshot.status,
      currentState: snapshot.currentState,
      pendingReason: snapshot.pendingReason,
      identity: {
        displayName: snapshot.conversationTitle?.trim() || local?.title,
        datasetName: snapshot.datasetRef,
      },
      pinned: local?.pinned ?? false,
      messageCount: snapshot.messageCount ?? 0,
    };
  }) };
};

export const createRun = async (input: {
  analysisMode?: 'topic' | 'free';
  projectId: string;
  datasetRef?: string;
  researchGoal?: string;
  useLanguageProvider?: boolean;
  sourceSessionId?: string;
  allowRemoteSamples?: boolean;
}): Promise<{ runId: string }> => {
  if (await apiVersion() === 'v2') return request('/api/v2/runs', { method: 'POST', body: JSON.stringify(input) });
  const snapshot = await request<V3RunSnapshot>('/api/v3/runs', {
    method: 'POST',
    body: JSON.stringify({
      projectId: input.projectId,
      analysisMode: input.analysisMode,
      datasetRef: input.datasetRef,
      sourceSessionId: input.sourceSessionId,
      researchGoal: input.researchGoal,
      allowRemoteSamples: input.allowRemoteSamples ?? false,
    }),
  });
  rememberV3Run(snapshot.runId, input.researchGoal);
  return { runId: snapshot.runId };
};

export const deleteRun = async (runId: string): Promise<{ runId: string }> => {
  if (await apiVersion() === 'v2') return request(`/api/v2/runs/${encodeURIComponent(runId)}`, { method: 'DELETE' });
  await request(v3Path(runId), { method: 'DELETE' });
  writeV3Runs(readV3Runs().filter((run) => run.runId !== runId));
  return { runId };
};

export const renameRun = async (runId: string, displayName: string): Promise<{ runId: string; displayName: string }> => {
  if (await apiVersion() === 'v2') return request(`/api/v2/runs/${encodeURIComponent(runId)}`, { method: 'PATCH', body: JSON.stringify({ displayName }) });
  await request(v3Path(runId), { method: 'PATCH', body: JSON.stringify({ displayName }) });
  rememberV3Run(runId, displayName);
  return { runId, displayName };
};

export const pinRun = async (runId: string, pinned: boolean): Promise<{ runId: string; pinned: boolean }> => {
  if (await apiVersion() === 'v2') return request(`/api/v2/runs/${encodeURIComponent(runId)}`, { method: 'PATCH', body: JSON.stringify({ pinned }) });
  writeV3Runs(readV3Runs().map((run) => run.runId === runId ? { ...run, pinned } : run));
  return { runId, pinned };
};

export const createWorkspaceSession = async (
  projectId: string,
  displayName?: string,
  initialMessage?: string,
  analysisMode: 'topic' | 'free' = 'topic',
): Promise<{ sessionId: string; interaction: WebAgentInteraction; session?: WebWorkspaceSummary }> => {
  if (await apiVersion() === 'v2') return request('/api/v2/workspace/sessions', { method: 'POST', body: JSON.stringify(displayName ? { displayName } : {}) });
  const snapshot = await request<V3RunSnapshot>('/api/v3/runs', {
    method: 'POST',
    body: JSON.stringify({
      projectId,
      analysisMode,
      ...(initialMessage?.trim() ? { researchGoal: initialMessage.trim() } : {}),
    }),
  });
  rememberV3Run(snapshot.runId, displayName);
  return { sessionId: snapshot.runId, interaction: v3Interaction(snapshot) };
};

export const listWorkspaceSessions = async (): Promise<{ sessions: WebWorkspaceSummary[] }> => {
  if (await apiVersion() === 'v2') return request('/api/v2/workspace/sessions?limit=30');
  const { runs } = await request<{
    runs: Array<{
      projectId: string;
      runId: string;
      conversationTitle?: string;
      messageCount: number;
      lastMessageAt?: string;
    }>;
  }>('/api/v3/runs?summary=1');
  const localRuns = new Map(readV3Runs().map((run) => [run.runId, run]));
  return {
    sessions: runs.map((run) => {
      const local = localRuns.get(run.runId);
      const updatedAt = run.lastMessageAt ?? local?.updatedAt ?? new Date(0).toISOString();
      return {
        sessionId: run.runId,
        projectId: run.projectId,
        title: run.conversationTitle?.trim() || local?.title || '新对话',
        messageCount: run.messageCount,
        createdAt: updatedAt,
        updatedAt,
        pinned: local?.pinned ?? false,
      };
    }),
  };
};

export const renameWorkspaceSession = async (sessionId: string, displayName: string): Promise<WebWorkspaceSummary> => {
  if (await apiVersion() === 'v2') return request(`/api/v2/workspace/sessions/${encodeURIComponent(sessionId)}`, { method: 'PATCH', body: JSON.stringify({ displayName }) });
  await renameRun(sessionId, displayName);
  return { sessionId, title: displayName, messageCount: 0, createdAt: new Date().toISOString(), updatedAt: new Date().toISOString(), pinned: false };
};

export const pinWorkspaceSession = async (sessionId: string, pinned: boolean): Promise<WebWorkspaceSummary> => {
  if (await apiVersion() === 'v2') return request(`/api/v2/workspace/sessions/${encodeURIComponent(sessionId)}`, { method: 'PATCH', body: JSON.stringify({ pinned }) });
  await pinRun(sessionId, pinned);
  const local = readV3Runs().find((run) => run.runId === sessionId);
  return { sessionId, title: local?.title ?? 'THETA 研究任务', messageCount: 0, createdAt: local?.updatedAt ?? new Date().toISOString(), updatedAt: new Date().toISOString(), pinned };
};

export const deleteWorkspaceSession = async (sessionId: string): Promise<{ sessionId: string }> => {
  if (await apiVersion() === 'v2') return request(`/api/v2/workspace/sessions/${encodeURIComponent(sessionId)}`, { method: 'DELETE' });
  await deleteRun(sessionId);
  return { sessionId };
};

export const getWorkspaceConversation = async (sessionId: string): Promise<WebWorkspaceTurn> => {
  if (await apiVersion() === 'v2') return request(`/api/v2/workspace/sessions/${encodeURIComponent(sessionId)}/conversation?limit=100`);
  const [conversation, snapshot, checkpoint] = await Promise.all([
    request<V3Conversation>(v3Path(sessionId, '/conversation')),
    request<V3RunSnapshot>(v3Path(sessionId)),
    v3Checkpoint(sessionId),
  ]);
  const status = v3Status(snapshot, checkpoint);
  return { sessionId, runId: sessionId, messages: mapV3Messages(conversation), tokenUsage: emptyTokenUsage(), interaction: status.interaction as WebAgentInteraction, status };
};

export const postWorkspaceMessage = async (
  sessionId: string,
  text: string,
  useLanguageProvider = true,
  attachments: WebAttachment[] = [],
  signal?: AbortSignal,
  modelPreference?: string,
): Promise<WebWorkspaceTurn> => {
  if (await apiVersion() === 'v2') return request(`/api/v2/workspace/sessions/${encodeURIComponent(sessionId)}/messages`, { method: 'POST', body: JSON.stringify({ text, useLanguageProvider, attachments, ...(modelPreference ? { modelPreference } : {}) }), signal });
  await submitBackgroundTurn(sessionId, '/messages', {content:text, attachments, ...(modelPreference ? { modelPreference } : {})}, signal);
  rememberV3Run(sessionId, text.slice(0, 80));
  return getWorkspaceConversation(sessionId);
};

export const listDatasets = async (projectId: string): Promise<{ datasets: WebDataset[] }> => {
  if (await apiVersion() === 'v2') return request('/api/v2/datasets');
  const data = await request<{ datasets: Array<Omit<WebDataset, 'name'> & { displayName: string }> }>(
    `/api/v3/datasets?projectId=${encodeURIComponent(projectId)}`,
  );
  return { datasets: data.datasets.map(({ displayName, ...dataset }) => ({ ...dataset, name: displayName })) };
};

export const uploadDataset = async (projectId: string, file: File): Promise<WebDataset> => {
  const body = new FormData();
  body.append('file', file);
  if (await apiVersion() === 'v2') return request('/api/v2/datasets/upload', { method: 'POST', body });
  const { displayName, ...dataset } = await request<Omit<WebDataset, 'name'> & { displayName: string }>(
    `/api/v3/datasets/upload?projectId=${encodeURIComponent(projectId)}`,
    { method: 'POST', body },
  );
  return { ...dataset, name: displayName };
};

export const listProjects = async (): Promise<{ projects: WebProject[] }> =>
  request('/api/v3/projects');

export const createProject = async (name: string): Promise<WebProject> =>
  request('/api/v3/projects', { method: 'POST', body: JSON.stringify({ name }) });

export const importManualResults = (source: { projectId?: number; datasetName?: string; selectedJobId?: string }): Promise<{
  project: WebProject; runId: string; copiedJobs: number; reused: boolean;
}> => request('/api/v3/projects/import-manual-results', {
  method: 'POST', body: JSON.stringify(source),
});

export const renameProject = async (projectId: string, name: string): Promise<WebProject> =>
  request(`/api/v3/projects/${encodeURIComponent(projectId)}`, {
    method: 'PATCH',
    body: JSON.stringify({ name }),
  });

export const pinProject = async (projectId: string, pinned: boolean): Promise<WebProject> =>
  request(`/api/v3/projects/${encodeURIComponent(projectId)}`, {
    method: 'PATCH',
    body: JSON.stringify({ pinned }),
  });

export const deleteProject = async (projectId: string): Promise<{ id: string }> =>
  request(`/api/v3/projects/${encodeURIComponent(projectId)}`, { method: 'DELETE' });

export const getRunWorkspaces = async (runId: string): Promise<WebRunWorkspaces> =>
  request(v3Path(runId, '/workspaces'));

export const invokeRunTool = async <T = Record<string, unknown>>(
  runId: string,
  toolId: string,
  input: Record<string, unknown> = {},
): Promise<WebToolInvocation<T>> => request(v3Path(runId, `/tools/${encodeURIComponent(toolId)}`), {
  method: 'POST',
  body: JSON.stringify({ input }),
});

export const runWorkflowAction = async (
  runId: string,
  action: 'discover' | 'research' | 'plan-design' | 'prepare-training' | 'advance-training',
): Promise<Record<string, unknown>> => request(v3Path(runId, `/${action}`), { method: 'POST', body: '{}' });

export const cancelRunTraining = async (runId: string, reason: string): Promise<Record<string, unknown>> =>
  request(v3Path(runId, '/cancel-training'), { method: 'POST', body: JSON.stringify({ content: reason }) });

export const getInferenceCatalog = async (): Promise<WebInferenceCatalog> => {
  if (typeof window !== 'undefined' && window.thetaDesktop) return window.thetaDesktop.catalog();
  return await apiVersion() === 'v2' ? request('/api/v2/inference') : request('/api/v3/inference');
};

export const getInferenceSettings = async (): Promise<WebInferenceSettings> =>
  await apiVersion() === 'v2' ? request('/api/v2/inference/settings') : request('/api/v3/inference/settings');

export const updateInferenceSettings = async (
  input: WebInferenceSettingsUpdate,
): Promise<WebInferenceSettings> => {
  if (typeof window !== 'undefined' && window.thetaDesktop && input.llm) {
    await window.thetaDesktop.saveInference(input.llm);
    return getInferenceSettings();
  }
  if (await apiVersion() === 'v2') return request('/api/v2/inference/settings', { method: 'PATCH', body: JSON.stringify(input) });
  return request('/api/v3/inference/settings', { method: 'PATCH', body: JSON.stringify(input) });
};

export const getRuntimeProfile = async (): Promise<WebRuntimeProfile> => {
  if (await apiVersion() === 'v2') return request('/api/v2/runtime');
  const health = await request<{ checks?: unknown[] }>('/api/v3/health');
  return {
    service: 'theta-agent-runtime',
    version: 'v3',
    compute: { backend: 'local', defaultDevice: 'cpu', scheduler: { supported: false, enabled: false } },
    capabilities: { domains: 1, tools: health.checks?.length ?? 0, skills: 0 },
    entryInteraction: v3Interaction({ runId: 'draft', status: 'ready', currentState: 'Intake', eventCount: 0 }),
  };
};

export const selectInferenceModel = async (providerId: string, model: string): Promise<unknown> =>
  await apiVersion() === 'v2'
    ? request('/api/v2/inference', { method: 'POST', body: JSON.stringify({ action: 'use', providerId, model }) })
    : { providerId, model, source: 'backend-environment' };

export const getRun = async (runId: string): Promise<WebRunDetail> => {
  if (await apiVersion() === 'v2') return request(`/api/v2/runs/${encodeURIComponent(runId)}`);
  const [snapshot, checkpoint] = await Promise.all([request<V3RunSnapshot>(v3Path(runId)), v3Checkpoint(runId)]);
  const status = v3Status(snapshot, checkpoint);
  const results = snapshot.artifactManifestHash || snapshot.trainingStatus === 'completed' ? {
    status: snapshot.trainingStatus ?? 'available',
    progress: snapshot.trainingPercent ?? 100,
    message: '训练产物已由 V3 后端登记。',
    visualizations: [],
    metrics: {},
    topics: [],
    warnings: [],
  } satisfies WebRunResults : undefined;
  return {
    runId,
    analysisMode: snapshot.analysisMode ?? 'topic',
    status,
    identity: { datasetRef: snapshot.datasetRef },
    plan: checkpoint?.kind === 'plan' ? checkpoint.content : undefined,
    results,
  };
};

export const getStatus = async (runId: string): Promise<WebRunStatus & Record<string, unknown>> => {
  if (await apiVersion() === 'v2') return request(`/api/v2/runs/${encodeURIComponent(runId)}/status`);
  const [snapshot, checkpoint] = await Promise.all([request<V3RunSnapshot>(v3Path(runId)), v3Checkpoint(runId)]);
  return v3Status(snapshot, checkpoint);
};

export const getResultCatalog = async (runId: string): Promise<WebResultCatalog> =>
  request<WebResultCatalog>(v3Path(runId, '/results'));

export const getConversation = async (runId: string): Promise<{ runId: string; messages: WebMessage[]; memory?: WebConversationMemory; tokenUsage: WebTokenUsage }> => {
  if (await apiVersion() === 'v2') return request(`/api/v2/runs/${encodeURIComponent(runId)}/conversation?limit=200`);
  const conversation = await request<V3Conversation>(v3Path(runId, '/conversation'));
  return { runId, messages: mapV3Messages(conversation), tokenUsage: emptyTokenUsage() };
};

export const postMessage = async (
  runId: string,
  text: string,
  useLanguageProvider = true,
  attachments: WebAttachment[] = [],
  signal?: AbortSignal,
  modelPreference?: string,
): Promise<WebPostMessageResult> => {
  if (await apiVersion() === 'v2') return request(`/api/v2/runs/${encodeURIComponent(runId)}/messages`, { method: 'POST', body: JSON.stringify({ text, useLanguageProvider, attachments, ...(modelPreference ? { modelPreference } : {}) }), signal });
  await submitBackgroundTurn(runId, '/messages', {content:text, attachments, ...(modelPreference ? {modelPreference} : {})}, signal);
  rememberV3Run(runId, text.slice(0, 80));
  const [conversation, status] = await Promise.all([getConversation(runId), getStatus(runId)]);
  return { runId, activeRunId: runId, messages: conversation.messages, status, tokenUsage: emptyTokenUsage() };
};

export const resultAssetUrl = (runId: string, relativePath: string): string =>
  `${BASE_URL}/api/v2/runs/${encodeURIComponent(runId)}/results/assets/${encodeURIComponent(relativePath)}`;

export const resultArchiveUrl = (runId: string): string =>
  `${BASE_URL}/api/v2/runs/${encodeURIComponent(runId)}/results/archive`;

export const getEvents = async (
  runId: string,
  options: { after?: string; limit?: number } = {},
): Promise<{ runId: string; events: WebRunEvent[]; count: number; total: number }> => {
  if (await apiVersion() === 'v3') {
    const snapshot = await request<V3ActivitySnapshot>(v3Path(runId, '/activities'));
    const events = snapshot.recent.map(mapV3Event);
    return { runId, events, count: events.length, total: events.length };
  }
  const params = new URLSearchParams();
  if (options.after) params.set('after', options.after);
  if (options.limit !== undefined) params.set('limit', String(options.limit));
  const query = params.size > 0 ? `?${params.toString()}` : '';
  return request(`/api/v2/runs/${encodeURIComponent(runId)}/events${query}`);
};

export const getReasoning = async (runId: string): Promise<WebReasoning> => {
  if (await apiVersion() === 'v2') return request(`/api/v2/runs/${encodeURIComponent(runId)}/reasoning`);
  const snapshot = await request<V3ActivitySnapshot>(v3Path(runId, '/activities'));
  const reasoningEvents = snapshot.recent.map(mapV3Event);
  return {
    runId,
    decisionGaps: [],
    toolCalls: snapshot.recent.filter((event) => event.toolId).map((event) => ({
      eventId: event.eventId,
      toolId: event.toolId as string,
      phase: event.status === 'failed' ? 'failed' : event.status === 'completed' ? 'completed' : 'started',
      label: event.displayName,
      timestamp: event.completedAt ?? event.startedAt,
      payload: event,
    })),
    reasoningEvents,
  };
};

export const postAction = async <T = unknown>(
  runId: string,
  action: Record<string, unknown>,
): Promise<{ result: T; status: WebRunStatus }> => {
  if (await apiVersion() === 'v2') return request(`/api/v2/runs/${encodeURIComponent(runId)}/actions`, { method: 'POST', body: JSON.stringify(action) });
  const actionName = String(action.action ?? '');
  if (actionName === 'message') {
    await submitBackgroundTurn(runId, '/messages', {content:String(action.text ?? '')});
  } else {
    if (!action.checkpointId || !action.expectedContentHash) throw new Error('请刷新并查看当前确认卡。');
    const revise = ['correctDataset', 'adjustPlan', 'revise'].includes(actionName);
    const reject = actionName === 'reject';
    await submitBackgroundTurn(runId, '/checkpoint-decision', {
      action: ['embedding_local', 'embedding_cloud'].includes(actionName) ? actionName : revise ? 'revise' : reject ? 'reject' : 'approve', checkpointId: action.checkpointId,
      expectedContentHash: action.expectedContentHash,
      ...((revise || reject) ? { feedback: String(action.text ?? action.reason ?? '拒绝') } : {}),
    });
  }
  const status = await getStatus(runId);
  return { result: status as T, status };
};

export interface StreamHandlers {
  onOpen?: () => void;
  onSnapshot?: (data: { runId: string; status: WebRunStatus; lastEventId?: string }) => void;
  onStatus?: (data: { status: WebRunStatus }) => void;
  onEvents?: (data: { events: WebRunEvent[] }) => void;
  onMessages?: (data: { messages: WebMessage[] }) => void;
  onTraining?: (data: Record<string, unknown>) => void;
  onError?: (error: Error) => void;
}

export const openRunStream = (runId: string, handlers: StreamHandlers): EventSource => {
  if (detectedApiVersion === 'v3') {
    const source = new EventSource(`${BASE_URL}${v3Path(runId, '/activities/stream')}`);
    source.addEventListener('sync', (event) => {
      try {
        const data = JSON.parse((event as MessageEvent).data) as { snapshot: V3RunSnapshot; checkpoint: V3Checkpoint | null; conversation: V3Conversation; activity: V3ActivitySnapshot };
        handlers.onStatus?.({ status: v3Status(data.snapshot, data.checkpoint) });
        handlers.onMessages?.({ messages: mapV3Messages(data.conversation) });
        handlers.onEvents?.({ events: data.activity.recent.map(mapV3Event) });
      } catch { /* The next snapshot resynchronizes the conversation. */ }
    });
    source.addEventListener('activity', (event) => {
      try {
        const snapshot = JSON.parse((event as MessageEvent).data) as V3ActivitySnapshot;
        handlers.onEvents?.({ events: snapshot.recent.map(mapV3Event) });
        handlers.onTraining?.({ ...snapshot.progress, phase: snapshot.phase });
      } catch {
        // The next activity snapshot resynchronizes the panel.
      }
    });
    source.onopen = () => handlers.onOpen?.();
    source.onerror = () => handlers.onError?.(new Error('实时流连接中断，将自动重连。'));
    return source;
  }
  const source = new EventSource(
    `${BASE_URL}/api/v2/runs/${encodeURIComponent(runId)}/stream`,
  );
  const wire = (kind: string, handler: ((data: never) => void) | undefined): void => {
    if (!handler) return;
    source.addEventListener(kind, (event) => {
      try {
        handler(JSON.parse((event as MessageEvent).data) as never);
      } catch {
        // Ignore malformed frames; the next tick resyncs.
      }
    });
  };
  wire('snapshot', handlers.onSnapshot as never);
  wire('status', handlers.onStatus as never);
  wire('events', handlers.onEvents as never);
  wire('messages', handlers.onMessages as never);
  wire('training', handlers.onTraining as never);
  source.onopen = () => handlers.onOpen?.();
  source.onerror = () => handlers.onError?.(new Error('实时流连接中断，将自动重连。'));
  return source;
};

export const setAnalysisMode = async (runId: string, analysisMode: 'topic' | 'free'): Promise<{analysisMode:'topic'|'free'}> => {
  if (await apiVersion() !== 'v3') throw new Error('自由分析需要新版 Agent 服务');
  return request(v3Path(runId), {method:'PATCH',body:JSON.stringify({analysisMode})});
};
