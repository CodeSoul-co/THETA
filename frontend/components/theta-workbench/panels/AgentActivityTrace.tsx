import { useMemo } from 'react'
import { systemText } from '@/lib/presentation'
import type { WebReasoning, WebReasoningToolCall } from '../api/client.ts'
import { IconChevronRightOutline14, JsonTree, StateDot } from '../ui/index.ts'
import { usePreferences } from '../preferences.tsx'
import css from '../styles/app.module.css'

interface AgentActivityTraceProps {
  reasoning?: WebReasoning
  working?: boolean
  currentProgress?: { phase: 'thinking' | 'tool'; label: string; step: number; toolId?: string }
}

interface ToolActivity {
  key: string
  toolId: string
  calls: WebReasoningToolCall[]
  latest: WebReasoningToolCall
  firstTimestamp: string
}

const record = (value: unknown): Record<string, unknown> =>
  value !== null && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {}

const toolKind = (toolId: string): 'search' | 'visual' | 'tool' => {
  if (/search|rag|retriev/i.test(toolId)) return 'search'
  if (/plot|chart|visual|image|result/i.test(toolId)) return 'visual'
  return 'tool'
}

const toolTitle = (toolId: string, locale: string): string => {
  const known: Record<string, [string, string]> = {
    'theta.rag.search': ['检索研究知识库', 'Search research knowledge'],
    'theta.model.catalog': ['读取模型目录', 'Read model catalog'],
    'theta.model.recommend': ['生成模型建议', 'Recommend a model'],
    'theta.dataset.inspect': ['理解数据集', 'Inspect the dataset'],
    'theta.dataset.explore': ['列出并读取上传数据', 'List and read uploaded data'],
    'theta.plan.propose': ['拟定研究计划', 'Propose a research plan'],
    'theta.training.start': ['启动受治理训练', 'Start governed training'],
  }
  const label = known[toolId]
  if (label) return locale === 'zh-CN' ? label[0] : label[1]
  if (toolKind(toolId) === 'search') return locale === 'zh-CN' ? '执行检索' : 'Search'
  if (toolKind(toolId) === 'visual') return locale === 'zh-CN' ? '生成研究产物' : 'Create research artifact'
  return locale === 'zh-CN' ? '调用工具' : 'Run tool'
}

const activityState = (phase: WebReasoningToolCall['phase']): 'error' | 'done' | 'ongoing' =>
  phase === 'failed' ? 'error' : phase === 'completed' || phase === 'validated' ? 'done' : 'ongoing'

const phaseTitle = (phase: WebReasoningToolCall['phase'], locale: string): string => {
  const labels: Record<WebReasoningToolCall['phase'], [string, string]> = {
    requested: ['请求执行工具', 'Tool requested'],
    started: ['开始执行工具', 'Tool started'],
    policy: ['权限策略校验通过', 'Policy check passed'],
    completed: ['工具执行完成', 'Tool completed'],
    failed: ['工具执行失败', 'Tool failed'],
    validated: ['输出契约校验通过', 'Output contract validated'],
  }
  return locale === 'zh-CN' ? labels[phase][0] : labels[phase][1]
}

const resultSummary = (activity: ToolActivity, locale: string): string => {
  const payloads = activity.calls.map((call) => record(call.payload))
  const query = payloads.map((payload) => payload.query).find((value) => typeof value === 'string')
  const evidenceCount = payloads
    .map((payload) => payload.evidenceCount ?? (Array.isArray(payload.evidence) ? payload.evidence.length : undefined))
    .find((value) => typeof value === 'number')
  if (toolKind(activity.toolId) === 'search') {
    if (typeof evidenceCount === 'number') {
      return locale === 'zh-CN' ? `找到 ${evidenceCount} 条证据` : `Found ${evidenceCount} evidence items`
    }
    if (typeof query === 'string') return query
  }
  const phases = activity.calls.length
  return locale === 'zh-CN' ? `${phases} 个可审计阶段` : `${phases} auditable stages`
}

export const AgentActivityTrace = ({
  reasoning,
  working = false,
  currentProgress,
}: AgentActivityTraceProps): React.ReactElement | null => {
  const { locale } = usePreferences()
  const toolActivities = useMemo(() => {
    const grouped = new Map<string, WebReasoningToolCall[]>()
    for (const call of reasoning?.toolCalls.slice(-80) ?? []) {
      const key = call.invocationId ?? call.toolId
      grouped.set(key, [...(grouped.get(key) ?? []), call])
    }
    return [...grouped.entries()].map(([key, calls]): ToolActivity => ({
      key,
      toolId: calls[0]?.toolId ?? 'unknown-tool',
      calls,
      latest: calls.at(-1) as WebReasoningToolCall,
      firstTimestamp: calls[0]?.timestamp ?? '',
    }))
  }, [reasoning?.toolCalls])
  if (!working && toolActivities.length === 0) return null

  return (
    <div className={css.activityTrace} aria-label={locale === 'zh-CN' ? 'Agent 活动' : 'Agent activity'}>
      {working && (
        <div className={css.activityWorking}>
          <span className={css.activitySpinner} />
          <span className={css.activityToolText}>
            <strong>{currentProgress?.phase === 'tool' ? '正在调用工具' : '正在思考'}</strong>
            <small>{systemText(currentProgress?.label, '正在分析请求并选择下一步能力') || '正在分析请求并选择下一步能力'}</small>
          </span>
        </div>
      )}
      {toolActivities.map((activity) => {
        const key = activity.key
        return (
          <details key={key} className={css.activityDisclosure} open>
            <summary>
              <StateDot size={7} state={activityState(activity.latest.phase)} />
              <span className={css.activityToolText}>
                <strong>{toolTitle(activity.toolId, locale)}</strong>
                <small>{resultSummary(activity, locale)}</small>
              </span>
              <span className={css.activityPhase}>{locale === 'zh-CN' ? phaseTitle(activity.latest.phase, locale).replace('工具', '') : activity.latest.phase}</span>
              <IconChevronRightOutline14 />
            </summary>
            <div className={css.activityStages}>
                {activity.calls.map((call) => (
                  <details key={call.eventId} className={css.activityStageDisclosure}>
                    <summary>
                      <StateDot size={6} state={activityState(call.phase)} />
                      <span>{phaseTitle(call.phase, locale)}</span>
                      <time>{new Date(call.timestamp).toLocaleTimeString(locale, { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</time>
                    </summary>
                    {call.payload != null && <div className={css.activityPayload}><JsonTree data={call.payload as object} copyable={false} /></div>}
                  </details>
                ))}
            </div>
          </details>
        )
      })}
    </div>
  )
}
