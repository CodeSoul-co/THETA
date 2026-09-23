"use client"

import { useInferenceSettings } from '@/components/theta-workbench/inference-settings'
import { SetupError } from '@/components/theta-workbench/panels/WorkbenchNotice'
import { openSetup } from '@/lib/workbench-guidance'
import { useEffect, useRef, useState } from 'react'
import { Sparkles } from 'lucide-react'
import { AiSidebar, type ChatMessage, type SendMessagePayload } from './ai-sidebar'
import { ResizableSidebar } from '@/components/layout/resizable-sidebar'
import { ETMAgentAPI } from '@/lib/api/etm-agent'
import { advisoryErrorMessage } from '@/lib/presentation'
import { readDraft, writeDraft } from '@/lib/workbench-draft'
import { restoreConsultations, type ConsultationHistory } from '@/lib/consultation-history'
import { ConsultationAPI } from '@/lib/api/consultations'

const messageId = () => crypto.randomUUID()

/** One project-scoped assistant for configuration and both result entry points. */
export function ProjectAssistant({ scopeKey, context, openRequest = 0 }: {
  scopeKey: string
  context: Record<string, unknown>
  openRequest?: number
}) {
  const { settings } = useInferenceSettings()
  const missingApi = !!settings && !settings.llm.apiKeyConfigured
  const historyKey = `${scopeKey}:consultation-threads`
  const [consultations, setConsultations] = useState<ConsultationHistory | null>(null)
  const [ready, setReady] = useState(false)
  const [reloadAttempt, setReloadAttempt] = useState(0)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [sending, setSending] = useState<string[]>([])
  const pendingSends = useRef(new Set<string>())
  const requestVersion = useRef(0)
  const mounted = useRef(true)
  const mutationPending = useRef(false)

  const load = async () => {
    const version = ++requestVersion.current
    try {
      const result = await ConsultationAPI.list(scopeKey)
      if (mounted.current && version === requestVersion.current) { setConsultations(result); setError('') }
    } catch (error) { if (mounted.current && version === requestVersion.current) setError(advisoryErrorMessage(error)) }
  }
  useEffect(() => {
    mounted.current = true
    setReady(false)
    const initialize = async () => {
      try {
        const saved = readDraft<ConsultationHistory | null>(historyKey, null)
        const legacy = readDraft<ChatMessage[]>(`${scopeKey}:consultation`, [])
        let result: ConsultationHistory
        if (saved || legacy.length) {
          const restored = restoreConsultations(saved, legacy)
          if (!saved) writeDraft(`${scopeKey}:consultation:${restored.activeId}:input`, readDraft(`${scopeKey}:consultation:input`, ''))
          result = await ConsultationAPI.import(scopeKey, restored)
        } else result = await ConsultationAPI.list(scopeKey)
        if (!result.activeId && !result.threads.length) result = await ConsultationAPI.create(scopeKey)
        if (mounted.current) { setConsultations(result); setError(''); setReady(true) }
      } catch (error) { if (mounted.current) { setError(advisoryErrorMessage(error)); setReady(true) } }
    }
    void initialize()
    return () => { mounted.current = false; requestVersion.current++ }
  }, [scopeKey, reloadAttempt])
  const active = consultations?.threads.find(thread => thread.id === consultations.activeId && !thread.deletedAt)
  const history = active?.messages ?? []
  const mutate = async (action: () => Promise<ConsultationHistory>) => {
    if (mutationPending.current) return
    mutationPending.current = true; setBusy(true)
    const version = ++requestVersion.current
    try {
      const result = await action()
      if (mounted.current && version === requestVersion.current) { setConsultations(result); setError('') }
    } catch (error) { if (mounted.current) setError(advisoryErrorMessage(error)) }
    finally { mutationPending.current = false; if (mounted.current) setBusy(false) }
  }
  const newConsultation = () => { void mutate(() => ConsultationAPI.create(scopeKey)) }
  const waiting = consultations?.threads.some(thread => thread.messages.some(message => message.isThinking)) || sending.length > 0
  useEffect(() => {
    const refresh = () => { if (!mutationPending.current) void load() }
    window.addEventListener('focus', refresh)
    const timer = waiting ? window.setInterval(refresh, 2000) : undefined
    return () => { window.removeEventListener('focus', refresh); if (timer) window.clearInterval(timer) }
  }, [scopeKey, waiting])
  const [open, setOpen] = useState(true)
  useEffect(() => { if (window.matchMedia('(max-width: 899px)').matches) setOpen(false) }, [])
  useEffect(() => { if (openRequest) setOpen(true) }, [openRequest])

  const send = async (payload: string | SendMessagePayload) => {
    if (missingApi) { openSetup('inference'); return false }
    const content = (typeof payload === 'string' ? payload : payload.content).trim()
    const charts = typeof payload === 'string' ? [] : payload.charts ?? []
    const chartAnalyses = typeof payload === 'string' ? [] : payload.chartAnalyses ?? []
    if ((!content && !charts.length) || !active || mutationPending.current || history.some(item => item.isThinking) || pendingSends.current.has(active.id)) return false
    const threadId = active.id
    pendingSends.current.add(threadId); setSending([...pendingSends.current])
    let failure = ''
    try {
      await ETMAgentAPI.advisory(content || '请解读已引用图表', {
        ...context,
        app_state: 'project_result_advisory',
        interaction_mode: 'consultation_and_result_interpretation_only',
        allowed_actions: ['answer_project_questions', 'interpret_selected_results'],
        forbidden_actions: ['create_training', 'start_training', 'modify_training', 'cancel_training'],
        selected_chart_analyses: chartAnalyses,
      }, charts, 'consultation', { scope: scopeKey, id: threadId, requestId: messageId() })
    } catch (error) {
      failure = advisoryErrorMessage(error)
    } finally {
      pendingSends.current.delete(threadId)
      if (mounted.current) { setSending([...pendingSends.current]); if (!mutationPending.current) await load(); if (failure) setError(failure) }
    }
    return !failure
  }

  return <>
    {/* Keep citations and unsent input when collapsed; do not create a second assistant. */}
      <ResizableSidebar hidden={!open} storageKey="theta.workspace.advisory-panel-ratio.v1" label="调整猫咪科学家宽度">
        {missingApi && <div className="shrink-0 border-b border-amber-200 bg-amber-50 p-3 text-amber-900"><SetupError error="API Key 未配置" /></div>}
        {!ready && <p className="p-4 text-sm text-slate-500">正在读取咨询记录…</p>}
        {error && <div role="alert" className="shrink-0 border-b border-amber-100 bg-amber-50 p-3 text-xs text-amber-800"><SetupError error={error} /><button type="button" className="ml-2 underline" onClick={() => consultations ? void load() : setReloadAttempt(attempt => attempt + 1)}>重新读取</button></div>}
        {ready && <AiSidebar key={`${scopeKey}:${active?.id ?? 'empty'}`} draftKey={`${scopeKey}:consultation:${active?.id ?? 'empty'}`} mode="advisory" chatHistory={history} onSendMessage={send} onCollapse={() => setOpen(false)}
          onNewConversation={newConsultation} activeConversationId={active?.id} consultationBusy={busy} sending={sending.includes(active?.id ?? '')} inputDisabled={!active || busy}
          historyStoredInDatabase={!!consultations}
          conversations={(consultations?.threads ?? []).map(thread => ({ id: thread.id, title: thread.title, updatedAt: thread.updatedAt, pinned: thread.pinned, deletedAt: thread.deletedAt, messageCount: thread.messages.filter(message => message.role === 'user').length }))}
          onSelectConversation={id => { void mutate(() => ConsultationAPI.patch(scopeKey, id, { select: true })) }}
          onPinConversation={(id, pinned) => { void mutate(() => ConsultationAPI.patch(scopeKey, id, { pinned })) }}
          onDeleteConversation={id => { void mutate(() => ConsultationAPI.delete(scopeKey, id)) }}
          onRestoreConversation={id => { void mutate(() => ConsultationAPI.patch(scopeKey, id, { deleted: false, select: true })) }} />}
      </ResizableSidebar>
    {!open && <button type="button" onClick={() => setOpen(true)} aria-label="展开猫咪科学家" className="flex w-10 shrink-0 flex-col items-center justify-center gap-2 border-l border-slate-200 bg-white text-slate-500 hover:bg-blue-50"><Sparkles className="h-5 w-5" /><span className="text-[10px]">助手</span></button>}
  </>
}
