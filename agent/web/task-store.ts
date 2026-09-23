import { contentHash } from '../src/domain/research.js';
import type { ResearchStore } from '../src/memory/research-store.js';

export type TaskStatus = 'queued' | 'running' | 'waiting_human' | 'completed' | 'failed' | 'cancelled' | 'interrupted';
export interface WebTask {
  id: string; runId: string; requestId: string; fingerprint: string;
  status: TaskStatus; phase: string; objective: string;
  createdAt: string; updatedAt: string; completedAt?: string;
  ownerPid: number; lease?: string; error?: string; lastOperation?: string;
  request: Record<string, unknown>;
}
export const taskPending = (task: WebTask) => task.status === 'queued' || task.status === 'running';
export const publicTask = ({ id, status, phase, objective, createdAt, updatedAt, completedAt, error, lastOperation }: WebTask) =>
  ({ id, status, phase, objective, createdAt, updatedAt, completedAt, error, lastOperation });

/** A durable receipt, not permission to replay an interrupted effect. */
export class WebTaskStore {
  constructor(private records: ResearchStore) {}
  key(runId: string, requestId: string) { return `webtask-${contentHash({ runId, requestId })}`; }
  get(id: string) { return this.records.get<WebTask>('web-task', id); }
  find(runId: string, requestId: string) {
    try { return this.get(this.key(runId, requestId)); } catch (error) {
      if (error instanceof Error && error.message.startsWith('记录不存在')) return undefined;
      throw error;
    }
  }
  create(runId: string, requestId: string, request: Record<string, unknown>, objective: string) {
    const now = new Date().toISOString();
    const task: WebTask = { id: this.key(runId, requestId), runId, requestId, fingerprint: contentHash(request),
      request, objective, status: 'queued', phase: '已保存，等待处理', createdAt: now, updatedAt: now, ownerPid: process.pid };
    if (!this.records.putIfAbsent('web-task', task.id, task)) throw new Error('任务已存在');
    return task;
  }
  update(id: string, patch: Partial<WebTask>) {
    const task = { ...this.get(id), ...patch, updatedAt: new Date().toISOString() };
    this.records.put('web-task', id, task); return task;
  }
  recover(release: (runId: string, lease: string) => void) {
    for (let offset = 0; ; offset += 100) {
      const page = this.records.list<WebTask>('web-task', offset);
      if (!page.length) break;
      for (const task of page) {
        if (!taskPending(task)) continue;
        let alive = true;
        try { process.kill(task.ownerPid, 0); } catch (error) { alive = (error as NodeJS.ErrnoException).code !== 'ESRCH'; }
        if (alive) continue;
        this.update(task.id, { status: 'interrupted', phase: '服务中断，等待核查', error: '任务记录已保留。继续前先核查已保存结果与批准状态，不会自动重算。', completedAt: new Date().toISOString() });
        if (task.lease) release(task.runId, task.lease);
      }
    }
  }
}
