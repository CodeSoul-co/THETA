import { contentHash, type ComputeJob, type ResearchRun } from '../domain/research.js';
import { EffectApprovals, type ApprovalReceipt } from '../domain/effect-approval.js';
import type { ResearchStore } from '../memory/research-store.js';
import type { ProductSession } from '../memory/session-store.js';
import type { ComputeGateway, ComputeRequest } from '../adapters/compute-gateway.js';

interface TrainingBatch {
  id: string; sessionId: string; sourceHash: string; configurationHash: string;
  requests: ComputeRequest[]; receipt: ApprovalReceipt; states: ComputeJob[];
  error?: string; observationError?: string; cancelled?: boolean;
}
const terminal = (job: ComputeJob) => ['completed', 'failed', 'cancelled'].includes(job.status);

/** Durable, serial execution of exactly the models confirmed in the shared editor. */
export class TrainingBatches {
  private readonly approvals: EffectApprovals;
  constructor(private records: ResearchStore, private compute: ComputeGateway, private backend: string) { this.approvals = new EffectApprovals(records); }
  get(id: string): TrainingBatch | undefined { return this.records.list<TrainingBatch>('training-batch').find(batch => batch.id === id); }
  restore(session: ProductSession, id: string): boolean {
    const batch = this.get(id);
    if (!batch || batch.sessionId !== session.id) throw new Error('训练批次不属于当前会话');
    session.runIds ??= []; session.trainingBatchIds ??= [];
    const missing = !session.trainingBatchIds.includes(id);
    if (missing) {
      session.trainingBatchIds.push(id);
      for (const request of batch.requests) if (!session.runIds.includes(request.runId)) session.runIds.push(request.runId);
      session.runId = batch.requests[0].runId;
    }
    if (`batch-${session.pendingConfirmation?.checkpointId}` === id) session.pendingConfirmation = undefined;
    return missing;
  }
  active(session: ProductSession): boolean { return (session.trainingBatchIds ?? []).some(id => { const batch = this.get(id); return batch && !batch.error && !batch.cancelled && batch.states.some(job => !terminal(job)); }); }
  view(session: ProductSession): { job: ComputeJob; jobs: ComputeJob[]; summary: string } | undefined {
    const batch = this.get(session.trainingBatchIds?.at(-1) ?? ''); if (!batch) return;
    if (!this.active(session) && !batch.requests.some(request => request.runId === session.runId)) return;
    const done = batch.states.filter(job => job.status === 'completed').length;
    const current = batch.states.find(job => job.status === 'running');
    const status: ComputeJob['status'] = batch.cancelled ? 'cancelled' : batch.error ? 'failed' : done === batch.states.length ? 'completed' : batch.states.every(terminal) ? 'failed' : 'running';
    const phase = current?.phase ?? (status === 'running' ? '等待下一模型训练' : status === 'completed' ? '全部模型训练完成' : '训练已停止');
    return { job: { id: batch.id, status, phase: `已完成 ${done}/${batch.states.length} · ${phase}`, percent: Math.round(batch.states.reduce((sum, job) => sum + job.percent, 0) / batch.states.length), error: batch.error }, jobs: batch.states,
      summary: `${phase}；已完成 ${done}/${batch.states.length} 个模型。${batch.error ?? batch.observationError ?? ''}` };
  }
  create(session: ProductSession, input: { id: string; sourceHash: string; configurationHash: string; requests: ComputeRequest[]; goal: string }): void {
    const { id, sourceHash, configurationHash, requests } = input;
    if (this.get(id)) throw new Error('这张训练卡已经提交，请查看已保存任务。');
    // A confirmed batch may wait for earlier models; its scope is immutable and its
    // lifetime covers only these explicitly bounded jobs, not future experiments.
    const approval = this.approvals.request(session.id, { action: 'compute.batch', target: this.backend, payload: requests, summary: `用户在配置弹窗确认 ${requests.length} 个模型：${requests.map(r => r.plan.modelId).join('、')}` }, requests.reduce((sum, request) => sum + request.plan.timeoutSeconds * 1000, 0) + 30 * 60_000);
    const receipt = this.approvals.decide(approval.id, session.id, approval.hash, true);
    const states: ComputeJob[] = requests.map(request => ({ id: request.jobId, status: 'queued', phase: '等待训练', percent: 0 }));
    for (const [index, request] of requests.entries()) {
      const run: ResearchRun = { id: request.runId, datasetRef: request.dataset.datasetRef, goal: input.goal, plan: request.plan, planHash: contentHash(request), computeBackend: this.backend,
        jobs: [request.jobId], notes: [], lastObservedJob: states[index] };
      this.records.put('run', run.id, run);
      session.runIds ??= []; if (!session.runIds.includes(run.id)) session.runIds.push(run.id);
    }
    this.records.put('training-batch', id, { id, sessionId: session.id, sourceHash, configurationHash, requests, receipt, states } satisfies TrainingBatch);
    session.trainingBatchIds ??= []; session.trainingBatchIds.push(id);
    session.runId = requests[0].runId; session.monitorTraining = true; session.pendingConfirmation = undefined;
  }
  async advance(session: ProductSession, save: () => void): Promise<void> {
    for (const id of session.trainingBatchIds ?? []) {
      let batch = this.get(id)!;
      if (batch.error || batch.cancelled || batch.states.every(terminal)) continue;
      try {
        for (const [index, state] of batch.states.entries()) {
          if (state.status !== 'running') continue;
          let job: ComputeJob;
          try { job = await this.compute.status(state.id); batch.observationError = undefined; }
          catch { batch.observationError = '暂时无法读取训练进度，将自动重试；不会重复提交训练。'; continue; }
          // Compute's queued state is already submitted, unlike our not-yet-submitted queue.
          batch.states[index] = { ...job, status: job.status === 'queued' ? 'running' : job.status };
          const run = this.records.get<ResearchRun>('run', batch.requests[index].runId); run.lastObservedJob = job; this.records.put('run', run.id, run);
        }
        if (batch.states.some(job => job.status === 'running')) { this.records.put('training-batch', id, batch); continue; }
        const index = batch.states.findIndex(job => job.status === 'queued');
        if (index >= 0) {
          this.approvals.assert(batch.receipt, 'compute.batch', this.backend, batch.requests);
          const request = batch.requests[index];
          const approval = this.approvals.request(session.id, { action: 'compute.submit', target: this.backend, payload: request, summary: `已确认批次 ${id} · ${request.plan.modelId}` });
          const receipt = this.approvals.decide(approval.id, session.id, approval.hash, true);
          const job = await this.compute.submit(request, receipt);
          batch.states[index] = { ...job, status: job.status === 'queued' ? 'running' : job.status };
          const run = this.records.get<ResearchRun>('run', request.runId); run.activeJob = job.id; run.lastObservedJob = job; this.records.put('run', run.id, run);
        }
      } catch (error) {
        batch.error = error instanceof Error ? error.message : String(error);
        batch.states = batch.states.map(job => job.status === 'queued' ? { ...job, status: 'failed', phase: '队列提交失败', error: batch.error } : job);
        this.saveRunStates(batch);
      }
      this.records.put('training-batch', id, batch);
    }
    session.monitorTraining = this.active(session); save();
  }
  async cancel(session: ProductSession): Promise<void> {
    for (const id of session.trainingBatchIds ?? []) {
      const batch = this.get(id)!;
      if (batch.sessionId !== session.id) throw new Error('训练批次不属于当前会话');
      for (const job of batch.states) if (job.status === 'running') await this.compute.cancel(job.id);
      batch.cancelled = true;
      batch.states = batch.states.map(job => terminal(job) ? job : { ...job, status: 'cancelled', phase: '用户取消' });
      this.saveRunStates(batch);
      this.records.put('training-batch', id, batch);
    }
    session.monitorTraining = false;
  }
  private saveRunStates(batch: TrainingBatch): void {
    batch.requests.forEach((request, index) => {
      const run = this.records.get<ResearchRun>('run', request.runId);
      run.lastObservedJob = batch.states[index]; this.records.put('run', run.id, run);
    });
  }
}
