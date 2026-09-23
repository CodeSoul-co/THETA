import { mkdirSync } from 'node:fs';
import path from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import { randomUUID } from 'node:crypto';
import type { PromptMessage } from '../providers/types.js';
import type { ExecutionEvent } from '../conversation/execution-events.js';
import type { TrainingPlan } from '../domain/research.js';
import type { MiningPlan } from '../tools/mining-tools.js';
import type { MethodReview } from '../tools/method-review.js';
import type { EmpiricalPlan, StatisticalReport } from '../tools/statistics-tools.js';

export interface AnalysisNotebook {
  objective: string;
  candidates: Array<{ hypothesis: string; evidence: string[]; counterevidence: string[]; status: 'untested' | 'supported' | 'rejected' | 'uncertain' }>;
  completed: string[];
  openQuestions: string[];
  artifacts: string[];
  nextAction: string;
}

export const notebookKey = (session: ProductSession): string => session.runId ?? JSON.stringify([...session.datasetRefs].sort());

export interface ProductSession {
  id: string;
  title: string;
  analysisMode?: 'topic' | 'free';
  webTaskId?: string;
  skillArtifacts?: Array<{name:string;path:string}>;
  statisticalPlans?: Record<string, {datasetRef?:string;plan:EmpiricalPlan}>;
  statisticalReports?: StatisticalReport[];
  pendingStatisticalInterpretation?: string;
  statisticalExecution?: {analysisId:string;approvalId:string;studyKey:string;status:'running'|'complete'|'failed';startedAt:string;error?:string};
  runId?: string;
  runIds?: string[];
  monitorTraining?: boolean;
  contextId?: string;
  lastUserIntent?: string;
  reports?: Array<{ jobId: string; reportPath: string; manifestPath?: string; reportStatus?: 'complete' | 'incomplete'; files: Array<{ name: string; path: string; kind: string }> }>;
  pendingArtifacts?: Array<{ name: string; path: string }>;
  resultGrants?: Record<string, { id: string; hash: string }>;
  resultSelection?: { mode: 'manual' | 'automatic'; strategy?: 'latest_completed' | 'all_completed'; resolvedJobIds: string[]; resolvedAt: string };
  pendingSynthesis?: { id: string; question: string; jobIds: string[]; contextHash: string; evidenceMessageStart?: number };
  interpretations?: Array<{ id: string; question: string; jobIds: string[]; contextHash: string; documentPath: string }>;
  pendingUnderstanding?: { datasetRefs: string[]; goal: string };
  executionEvents?: ExecutionEvent[];
  datasetRefs: string[];
  messages: PromptMessage[];
  analysisNotebooks?: Record<string, AnalysisNotebook>;
  miningPlans?: Record<string, MiningPlan>;
  methodReviews?: Record<string, MethodReview>;
  embeddingSelections?: Record<string, { scope: string }>;
  trainingBatchIds?: string[];
  pendingConfirmation?: { kind: string; checkpointId: string; contentHash: string; summary: string; runId: string;
    trainingPlans?: TrainingPlan[];
    embeddingChoice?: { plan: TrainingPlan; plans?: TrainingPlan[]; datasetHash: string; backend: string; cloud: { configured: boolean; provider: string; endpoint: string; model: string; configurationIssues?: string[] } } };
  updatedAt: string;
}

export class ProductSessionStore {
  private readonly db: DatabaseSync;
  constructor(directory: string) {
    mkdirSync(directory, { recursive: true });
    this.db = new DatabaseSync(path.join(directory, 'sessions.sqlite'));
    this.db.exec('PRAGMA busy_timeout=5000; PRAGMA journal_mode=WAL; CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL, lease TEXT, expires INTEGER)');
  }
  create(): ProductSession {
    const session: ProductSession = { id: `chat-${randomUUID()}`, title: '新的研究对话', datasetRefs: [], messages: [], updatedAt: new Date().toISOString() };
    this.db.prepare('INSERT INTO sessions (id,value,updated_at) VALUES (?,?,?)').run(session.id, JSON.stringify(session), session.updatedAt);
    return session;
  }
  get(id: string): ProductSession {
    const row = this.db.prepare('SELECT value FROM sessions WHERE id=?').get(id);
    if (!row) throw new Error(`会话不存在：${id}`);
    return JSON.parse(String(row.value));
  }
  list(): Array<{ id: string; title: string; updatedAt: string }> {
    return this.db.prepare('SELECT value FROM sessions ORDER BY updated_at DESC LIMIT 30').all().map((row) => {
      const session = JSON.parse(String(row.value)) as ProductSession;
      return { id: session.id, title: session.title, updatedAt: session.updatedAt };
    });
  }
  save(session: ProductSession, lease?: string): void {
    session.updatedAt = new Date().toISOString();
    const result = this.db.prepare('UPDATE sessions SET value=?,updated_at=? WHERE id=? AND (lease IS NULL OR lease=?)')
      .run(JSON.stringify(session), session.updatedAt, session.id, lease ?? '');
    if (result.changes !== 1) throw new Error('会话正在另一个终端中使用，当前写入已停止。');
  }
  acquire(id: string): string {
    const lease = randomUUID();
    const result = this.db.prepare('UPDATE sessions SET lease=?,expires=? WHERE id=? AND (lease IS NULL OR expires<?)').run(lease, Date.now() + 120_000, id, Date.now());
    if (result.changes !== 1) throw new Error('该会话正在另一个终端执行，请稍后重试。');
    return lease;
  }
  renew(id: string, lease: string): void { this.db.prepare('UPDATE sessions SET expires=? WHERE id=? AND lease=?').run(Date.now() + 120_000, id, lease); }
  release(id: string, lease: string): void { this.db.prepare('UPDATE sessions SET lease=NULL,expires=NULL WHERE id=? AND lease=?').run(id, lease); }
  close(): void { this.db.close(); }
}
