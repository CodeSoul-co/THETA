import test from 'node:test'
import assert from 'node:assert/strict'
import { createConsultation, restoreConsultations, updateConsultation } from './consultation-history.ts'

test('旧咨询迁移为历史，新建咨询不清空旧消息，刷新恢复选中记录', () => {
  const legacy = [{ id: 'q', role: 'user' as const, type: 'text' as const, content: '旧项目的问题', timestamp: '12:00' }]
  const old = restoreConsultations(null, legacy)
  const fresh = createConsultation()
  const state = { activeId: fresh.id, threads: [...old.threads, fresh] }
  const restored = restoreConsultations(JSON.parse(JSON.stringify(state)), [])
  assert.equal(restored.activeId, fresh.id)
  assert.equal(restored.threads[0].title, '旧项目的问题')
  assert.deepEqual(restored.threads[0].messages, legacy)
  assert.deepEqual(restored.threads[1].messages, [])
  assert.deepEqual(legacy[0].content, '旧项目的问题')
})

test('异步回复只进入发起咨询，刷新中断回复不会永久等待', () => {
  const a = createConsultation(), b = createConsultation()
  const state = { activeId: b.id, threads: [a, b] }
  const pending = updateConsultation(state, a.id, messages => [...messages, { id: 'reply', role: 'ai', type: 'text', content: '', timestamp: '12:01', isThinking: true }])
  assert.equal(pending.activeId, b.id)
  assert.equal(pending.threads[1].messages.length, 0)
  const restored = restoreConsultations(pending, [])
  assert.equal(restored.threads[0].messages[0].isThinking, false)
  assert.match(restored.threads[0].messages[0].content, /重新发送/)
  const finished = updateConsultation(pending, a.id, messages => messages.map(m => ({ ...m, content: '正确回复', isThinking: false })))
  assert.equal(finished.threads[0].messages[0].content, '正确回复')
  assert.deepEqual(finished.threads[1].messages, [])
})
