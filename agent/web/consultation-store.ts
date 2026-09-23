import { DatabaseSync } from 'node:sqlite';
import { randomUUID, createHash } from 'node:crypto';
import path from 'node:path';

export class ConsultationError extends Error {
  constructor(readonly status: number, message: string, readonly code = 'consultation_error') { super(message); }
}
export interface ConsultationMessage {
  id: string; role: 'user' | 'ai'; type: 'text'; content: string; timestamp: string;
  isThinking?: boolean; data?: Record<string, unknown>; requestId?: string; fingerprint?: string;
}
export interface ConsultationThread {
  id: string; title: string; createdAt: string; updatedAt: string; pinned: boolean;
  deletedAt?: string; messages: ConsultationMessage[];
}
export interface ConsultationSnapshot { activeId: string; threads: ConsultationThread[] }
const now = () => new Date().toISOString();
const threadId = (value: unknown) => {
  if (typeof value !== 'string' || !/^[a-zA-Z0-9_-]{1,120}$/.test(value)) throw new ConsultationError(400, '咨询编号无效。');
  return value;
};
export function consultationScope(value: unknown): string {
  if (typeof value !== 'string' || !value.trim() || value.length > 300) throw new ConsultationError(400, '请指定有效的项目咨询范围。');
  return value;
}

/** Separate from training sessions. Every query is bound to the authenticated owner. */
export class ConsultationStore {
  constructor(readonly home: string) {
    this.use(db => {
      db.exec(`CREATE TABLE IF NOT EXISTS consultations (owner TEXT NOT NULL, scope TEXT NOT NULL, id TEXT NOT NULL, value TEXT NOT NULL, PRIMARY KEY(owner,scope,id));
        CREATE TABLE IF NOT EXISTS consultation_selection (owner TEXT NOT NULL, scope TEXT NOT NULL, active_id TEXT NOT NULL, PRIMARY KEY(owner,scope));`);
      // A restarted service cannot finish its old inference; keep the question and expose interruption.
      for (const row of db.prepare("SELECT * FROM consultations WHERE value LIKE '%\"isThinking\":true%'").all()) {
        const thread = JSON.parse(String(row.value)) as ConsultationThread;
        thread.messages = thread.messages.map(message => message.isThinking ? { ...message, isThinking: false, content: '服务已重新启动，上次咨询回复未完成，请重新发送问题。' } : message);
        this.put(db, String(row.owner), String(row.scope), thread);
      }
    });
  }
  private use<T>(fn: (db: DatabaseSync) => T): T {
    const db = new DatabaseSync(path.join(this.home, 'research.sqlite'));
    try {
      db.exec('PRAGMA busy_timeout=5000; BEGIN IMMEDIATE');
      const result = fn(db); db.exec('COMMIT'); return result;
    } catch (error) { db.exec('ROLLBACK'); throw error; } finally { db.close(); }
  }
  private get(db: DatabaseSync, owner: string, scope: string, id: string): ConsultationThread {
    const row = db.prepare('SELECT value FROM consultations WHERE owner=? AND scope=? AND id=?').get(owner, scope, threadId(id));
    if (!row) throw new ConsultationError(404, '咨询记录不存在。');
    return JSON.parse(String(row.value));
  }
  private put(db: DatabaseSync, owner: string, scope: string, thread: ConsultationThread) {
    db.prepare('INSERT INTO consultations VALUES (?,?,?,?) ON CONFLICT(owner,scope,id) DO UPDATE SET value=excluded.value').run(owner, scope, thread.id, JSON.stringify(thread));
  }
  private select(db: DatabaseSync, owner: string, scope: string, id: string) {
    db.prepare('INSERT INTO consultation_selection VALUES (?,?,?) ON CONFLICT(owner,scope) DO UPDATE SET active_id=excluded.active_id').run(owner, scope, id);
  }
  private snapshot(db: DatabaseSync, owner: string, scope: string): ConsultationSnapshot {
    const threads = db.prepare('SELECT value FROM consultations WHERE owner=? AND scope=?').all(owner, scope).map(row => JSON.parse(String(row.value)) as ConsultationThread);
    threads.sort((a, b) => Number(b.pinned) - Number(a.pinned) || b.updatedAt.localeCompare(a.updatedAt));
    const selected = db.prepare('SELECT active_id FROM consultation_selection WHERE owner=? AND scope=?').get(owner, scope)?.active_id;
    const activeId = threads.find(t => t.id === selected && !t.deletedAt)?.id ?? threads.find(t => !t.deletedAt)?.id ?? '';
    return { activeId, threads };
  }
  list(owner: string, scope: string) { return this.use(db => this.snapshot(db, owner, scope)); }
  create(owner: string, scope: string, id: string = randomUUID()) {
    return this.use(db => {
      threadId(id);
      if (!db.prepare('SELECT 1 FROM consultations WHERE owner=? AND scope=? AND id=?').get(owner, scope, id)) {
        this.put(db, owner, scope, { id, title: '新咨询', createdAt: now(), updatedAt: now(), pinned: false, messages: [] });
      } else if (this.get(db, owner, scope, id).deletedAt) throw new ConsultationError(409, '此咨询已删除，请从已删除记录恢复。');
      this.select(db, owner, scope, id);
      return this.snapshot(db, owner, scope);
    });
  }
  import(owner: string, scope: string, input: unknown, activeId?: string) {
    if (!Array.isArray(input) || input.length > 1000) throw new ConsultationError(400, '迁移记录格式无效或超过 1000 条。');
    return this.use(db => {
      const hadSelection = !!db.prepare('SELECT 1 FROM consultation_selection WHERE owner=? AND scope=?').get(owner, scope);
      for (const item of input) {
        const id = threadId(item?.id);
        // Includes soft-deleted rows: an old browser must never overwrite or resurrect them.
        if (db.prepare('SELECT 1 FROM consultations WHERE owner=? AND scope=? AND id=?').get(owner, scope, id)) continue;
        if (!Array.isArray(item.messages) || item.messages.length > 2000) throw new ConsultationError(400, '迁移消息格式无效。');
        const messages = item.messages.map((m: Record<string, unknown>): ConsultationMessage => {
          if (!m || !['user', 'ai'].includes(String(m.role)) || typeof m.content !== 'string' || m.content.length > 100_000) throw new ConsultationError(400, '迁移消息格式无效。');
          return { id: threadId(m.id), role: m.role as 'user' | 'ai', type: 'text', content: m.isThinking ? '迁移前的咨询回复未完成，请重新发送问题。' : m.content,
            timestamp: String(m.timestamp ?? ''), ...(m.data && typeof m.data === 'object' ? { data: m.data as Record<string, unknown> } : {}) };
        });
        const date = typeof item.updatedAt === 'string' && Number.isFinite(Date.parse(item.updatedAt)) ? new Date(item.updatedAt).toISOString() : now();
        this.put(db, owner, scope, { id, title: String(item.title || messages.find((m: ConsultationMessage) => m.role === 'user')?.content || '新咨询').slice(0, 120), createdAt: date, updatedAt: date, pinned: item.pinned === true, messages });
      }
      if (!hadSelection && activeId && this.snapshot(db, owner, scope).threads.some(t => t.id === activeId && !t.deletedAt)) this.select(db, owner, scope, activeId);
      return this.snapshot(db, owner, scope);
    });
  }
  patch(owner: string, scope: string, id: string, input: { pinned?: boolean; deleted?: boolean; select?: boolean }) {
    return this.use(db => {
      const thread = this.get(db, owner, scope, id);
      if (input.pinned !== undefined) thread.pinned = input.pinned;
      if (input.deleted === true) thread.deletedAt = now();
      if (input.deleted === false) delete thread.deletedAt;
      if (input.select && thread.deletedAt) throw new ConsultationError(409, '请先恢复已删除的咨询。');
      this.put(db, owner, scope, thread);
      if (input.select) this.select(db, owner, scope, id);
      return this.snapshot(db, owner, scope);
    });
  }
  begin(owner: string, scope: string, id: string, requestId: string, content: string, payload: Record<string, unknown>) {
    return this.use(db => {
      const thread = this.get(db, owner, scope, id);
      if (thread.deletedAt) throw new ConsultationError(409, '此咨询已删除，请先恢复。');
      threadId(requestId);
      const fingerprint = createHash('sha256').update(JSON.stringify({ content, payload })).digest('hex');
      const previous = thread.messages.find(m => m.role === 'user' && m.requestId === requestId);
      if (previous) {
        if (previous.fingerprint !== fingerprint) throw new ConsultationError(409, '同一消息编号不能用于不同内容。');
        const reply = thread.messages.find(m => m.role === 'ai' && m.requestId === requestId);
        return { reused: true, reply, history: [] };
      }
      if (thread.messages.some(m => m.isThinking)) throw new ConsultationError(409, '此咨询正在回复，请稍后再发送。');
      if (thread.messages.length >= 2000) throw new ConsultationError(400, '此咨询记录较长，请新建咨询继续。');
      const history = thread.messages.slice(-12).map(({ role, content }) => ({ role, content: content.slice(0, 1500) }));
      const timestamp = new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
      thread.messages.push({ id: randomUUID(), role: 'user', type: 'text', content, timestamp, requestId, fingerprint, data: payload },
        { id: randomUUID(), role: 'ai', type: 'text', content: '', timestamp, requestId, isThinking: true });
      thread.title = thread.messages.find(m => m.role === 'user')!.content.slice(0, 32);
      thread.updatedAt = now(); this.put(db, owner, scope, thread);
      return { reused: false, history };
    });
  }
  finish(owner: string, scope: string, id: string, requestId: string, content: string) {
    this.use(db => {
      // Re-read to preserve pin/delete changes made while inference was in flight.
      const thread = this.get(db, owner, scope, id);
      thread.messages = thread.messages.map(m => m.role === 'ai' && m.requestId === requestId ? { ...m, content, isThinking: false } : m);
      thread.updatedAt = now(); this.put(db, owner, scope, thread);
    });
  }
}
