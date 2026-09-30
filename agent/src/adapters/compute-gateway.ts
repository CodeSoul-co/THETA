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
   * The worker owns its database. Hosts use the API and their bounded observer
   * cache rather than reading local files (which may belong to a remote worker).
   */
  cachedStatus(jobId: string): ComputeJob | undefined {
    return undefined;
  }
  cancel(jobId: string): Promise<ComputeJob> { return this.worker.call('compute.cancel', { home: this.home, jobId }); }
  results(jobId: string, view: ResultView, offset?: number): Promise<unknown> { return this.worker.call('compute.results', { home: this.home, jobId, view, offset }); }
}
