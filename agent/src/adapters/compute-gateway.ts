import path from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import type { ComputeJob, Dataset, TrainingPlan } from '../domain/research.js';
import type { CapabilityWorker } from './python-worker.js';
import { EffectApprovals, type ApprovalReceipt } from '../domain/effect-approval.js';
import { ResearchStore } from '../memory/research-store.js';

export type ResultView = 'summary' | 'artifacts' | 'tables' | 'figures' | 'distribution' | 'metrics' | 'report';

export interface ComputeRequest { jobId: string; runId: string; dataset: Dataset; plan: TrainingPlan; execution?: Record<string, unknown> }
export interface ComputeGateway {
  submit(request: ComputeRequest, authorization?: ApprovalReceipt): Promise<ComputeJob>;
  status(jobId: string): Promise<ComputeJob>;
  cancel(jobId: string): Promise<ComputeJob>;
  results(jobId: string, view: ResultView, offset?: number): Promise<unknown>;
  /** Cheap local snapshot for list views; absent on remote gateways. */
  cachedStatus?(jobId: string): ComputeJob | undefined;
}
/** Single-machine transport of the existing worker pipeline, not a copy of THETA. */
export class LocalComputeGateway implements ComputeGateway {
  constructor(private readonly home: string, private readonly worker: CapabilityWorker) {}
  submit(request: ComputeRequest, authorization?: ApprovalReceipt): Promise<ComputeJob> {
    new EffectApprovals(new ResearchStore(this.home)).assert(authorization, 'compute.submit', 'local', request);
    return this.worker.call('compute.submit', { home: this.home, ...request, authorization });
  }
  status(jobId: string): Promise<ComputeJob> { return this.worker.call('compute.status', { home: this.home, jobId }); }
  /**
   * A local job already lives in compute.sqlite: reading the row avoids spawning a
   * Python worker per job for every result-list request. Returns undefined when the
   * store cannot be read, so callers can fall back to the capability call.
   */
  cachedStatus(jobId: string): ComputeJob | undefined {
    let db: DatabaseSync | undefined;
    try {
      db = new DatabaseSync(path.join(this.home, 'compute.sqlite'), { readOnly: true });
      const row = db.prepare('SELECT state, value FROM jobs WHERE id = ?').get(jobId) as { state?: string; value?: string } | undefined;
      if (!row?.value) return undefined;
      const job = JSON.parse(row.value) as ComputeJob;
      return { ...job, id: job.id ?? jobId, status: (row.state as ComputeJob['status']) ?? job.status };
    } catch { return undefined; }
    finally { try { db?.close(); } catch { /* ignore */ } }
  }
  cancel(jobId: string): Promise<ComputeJob> { return this.worker.call('compute.cancel', { home: this.home, jobId }); }
  results(jobId: string, view: ResultView, offset?: number): Promise<unknown> { return this.worker.call('compute.results', { home: this.home, jobId, view, offset }); }
}
