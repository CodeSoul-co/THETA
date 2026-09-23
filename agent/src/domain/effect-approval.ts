import { randomUUID } from 'node:crypto';
import { contentHash } from './research.js';
import { ResearchStore } from '../memory/research-store.js';

export interface EffectRequest {
  action: string; target: string; payload: unknown; summary: string;
}
export interface EffectApproval extends EffectRequest {
  id: string; hash: string; sessionId: string;
  status: 'pending' | 'approved' | 'denied'; expiresAt: number;
}
export interface ApprovalReceipt { id: string; hash: string }

/** Host-owned capability authorization. No research stages or LLM approval tool. */
export class EffectApprovals {
  constructor(private readonly store: ResearchStore) {}
  request(sessionId: string, request: EffectRequest, validForMs = 15 * 60_000): EffectApproval {
    const value: EffectApproval = { ...request, id: randomUUID(), sessionId,
      hash: contentHash({ action: request.action, target: request.target, payload: request.payload }),
      status: 'pending', expiresAt: Date.now() + validForMs };
    this.store.put('effect-approval', value.id, value); return value;
  }
  get(id: string): EffectApproval { return this.store.get('effect-approval', id); }
  decide(id: string, sessionId: string, expectedHash: string, approved: boolean): ApprovalReceipt {
    const value = this.get(id);
    if (value.sessionId !== sessionId || value.hash !== expectedHash || value.status !== 'pending' || (approved && Date.now() > value.expiresAt)) throw new Error('授权已过期、已处理或内容不匹配，请重新查看本次操作。');
    if (!this.store.compareAndSet('effect-approval', id, value, { ...value, status: approved ? 'approved' : 'denied' })) throw new Error('该授权已由另一个宿主处理。');
    return { id: value.id, hash: value.hash };
  }
  assert(receipt: ApprovalReceipt | undefined, action: string, target: string, payload: unknown): void {
    if (!receipt) throw new Error('需要用户确认本次外部调用或计算开销。');
    const value = this.get(receipt.id);
    if (value.status !== 'approved' || value.hash !== receipt.hash || value.hash !== contentHash({ action, target, payload }) || Date.now() > value.expiresAt) throw new Error('操作、配置或数据已变化，必须重新确认后执行。');
  }
}
