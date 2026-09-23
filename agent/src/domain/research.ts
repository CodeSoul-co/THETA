import { createHash } from 'node:crypto';

export interface Dataset {
  datasetRef: string; sha256: string; fileName: string; managedPath: string; sizeBytes: number;
}
export interface TrainingPlan {
  modelId: string; textColumn: string; timeColumn?: string; labelColumn?: string; device?: string; covariates?: string[];
  params: Record<string, string | number | boolean | null | number[]>;
  rationale: string; timeoutSeconds: number; externalRequestLimit?: number;
}
export interface ResearchRun {
  id: string; goal: string; datasetRef?: string; profile?: Record<string, unknown>; computeBackend?: string;
  plan?: TrainingPlan; planHash?: string; revision?: number;
  jobs: string[]; activeJob?: string; lastObservedJob?: { id: string; status: ComputeJob['status']; percent: number; phase?: string; telemetry?: JobTelemetry; resultDir?: string; resultWarning?: string }; notes: string[];
}
export interface JobTelemetry {
  schemaVersion: 'theta.job-observation.v1'; observedAt: number; percentKind: 'stage_marker';
  health: 'waiting' | 'responding' | 'unresponsive' | 'finished';
  heartbeatAgeSeconds: number | null; elapsedSeconds: number | null; phaseElapsedSeconds: number | null;
  lastLogAgeSeconds: number | null;
  iteration: { current: number; total: number; source: string; meaning: string } | null;
  activity: 'fitting' | 'visualizing' | null; limitation: string;
}
export interface ComputeJob {
  id: string; status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
  phase: string; percent: number; error?: string;
  message?: string; phaseHistory?: { phase: string; at: number }[];
  diagnostics?: { available: boolean; stage?: string; category?: string; message?: string; reason?: string };
  resultDir?: string; resultWarning?: string; resultHash?: string; pid?: number;
  telemetry?: JobTelemetry;
  runtime?: { profile: string; revision: string; python: string; mode: string; dependencyFingerprint: string };
}
export const contentHash = (value: unknown): string => {
  const canonical = (item: unknown): unknown => Array.isArray(item) ? item.map(canonical)
    : item && typeof item === 'object' ? Object.fromEntries(Object.entries(item).sort(([a], [b]) => a.localeCompare(b)).map(([k, v]) => [k, canonical(v)])) : item;
  return createHash('sha256').update(JSON.stringify(canonical(value))).digest('hex');
};
