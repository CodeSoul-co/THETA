import type { ChatMessage } from '../components/chat/ai-sidebar'

export interface ConsultationThread {
  id: string
  title: string
  updatedAt: string
  messages: ChatMessage[]
  pinned?: boolean
  deletedAt?: string
}
export interface ConsultationHistory {
  activeId: string
  threads: ConsultationThread[]
}

export function createConsultation(): ConsultationThread {
  return { id: crypto.randomUUID(), title: '新咨询', updatedAt: new Date().toISOString(), messages: [] }
}

/** Import the previous single conversation without removing its original snapshot. */
export function restoreConsultations(saved: ConsultationHistory | null, legacy: ChatMessage[]): ConsultationHistory {
  const threads = saved?.threads?.length ? saved.threads : [{ ...createConsultation(), title: legacy.find(m => m.role === 'user')?.content.slice(0, 32) || '新咨询', messages: legacy }]
  return {
    activeId: threads.some(t => t.id === saved?.activeId) ? saved!.activeId : threads[0].id,
    threads: threads.map(t => ({ ...t, messages: t.messages.map(m => m.isThinking ? {
      ...m, isThinking: false, content: '页面已重新打开，上次咨询回复未完成，请重新发送问题。',
    } : m) })),
  }
}

/** Target the originating thread, even if the user switched away while waiting. */
export function updateConsultation(state: ConsultationHistory, id: string, update: (messages: ChatMessage[]) => ChatMessage[]): ConsultationHistory {
  return { ...state, threads: state.threads.map(thread => {
    if (thread.id !== id) return thread
    const messages = update(thread.messages)
    return { ...thread, messages, title: messages.find(m => m.role === 'user')?.content.slice(0, 32) || '新咨询', updatedAt: new Date().toISOString() }
  }) }
}
