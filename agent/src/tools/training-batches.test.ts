import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { LocalProductTools } from './local-tools.js';
import { ProductSessionStore } from '../memory/session-store.js';
import { ResearchStore } from '../memory/research-store.js';
import type { CapabilityWorker } from '../adapters/python-worker.js';
import type { ComputeGateway, ComputeRequest } from '../adapters/compute-gateway.js';
import type { ComputeJob } from '../domain/research.js';

const plan = (modelId: string, params = {}) => ({ modelId, textColumn: '正文', params, rationale: 'Agent 建议', timeoutSeconds: 43200 });
test('edited multi-model configuration is validated atomically, queued serially and restored without duplicate jobs', async t => {
  const home = mkdtempSync(path.join(tmpdir(), 'theta-training-editor-'));
  const sessions = new ProductSessionStore(home); let session = sessions.create();
  const submissions: ComputeRequest[] = []; const jobs = new Map<string, ComputeJob>(); let statusUnavailable = false;
  const worker: CapabilityWorker = { async call<T>(name: string, args: any): Promise<T> {
    if (name === 'dataset.import') return { datasetRef: 'fixture', sha256: 'sha', managedPath: '/fixture.csv', fileName: 'fixture.csv', sizeBytes: 12 } as T;
    if (name === 'dataset.profile') return { columns: ['正文', '时间', '来源'] } as T;
    if (name === 'runtime.config') return { embedding: { configured: true, provider: 'fixture', endpoint: 'https://example.invalid/v1', model: 'embed' } } as T;
    if (name === 'plan.validate') return { valid: true } as T;
    if (name === 'compute.preview') {
      if (args.plan.modelId === 'dtm' && !args.plan.timeColumn) throw new Error('DTM 必须选择时间列');
      if (args.plan.params.invalid !== undefined) throw new Error('模型参数不支持');
      const mode = args.plan.params.embedding_provider ?? 'local';
      return { rowCount: 30, readiness: { ready: true }, execution: { embedding: { mode, provider: 'fixture', model: 'embed', endpoint: 'https://example.invalid/v1/embeddings' }, maxExternalRequests: mode === 'cloud' ? args.plan.externalRequestLimit : 0 } } as T;
    }
    throw new Error(name);
  } };
  const compute: ComputeGateway = {
    async submit(request) { assert.ok(!submissions.some(item => item.jobId === request.jobId)); submissions.push(request); const job: ComputeJob = { id: request.jobId, status: 'running', phase: '训练中', percent: 30 }; jobs.set(job.id, job); return job; },
    async status(id) { if (statusUnavailable) throw new Error('temporary status outage'); return jobs.get(id)!; }, async results() { throw new Error('not needed'); },
    async cancel(id) { const job: ComputeJob = { id, status: 'cancelled', phase: '用户取消', percent: 0 }; jobs.set(id, job); return job; },
  };
  const options = { runtimeDb: path.join(home, 'runtime.sqlite'), uploadDir: home, worker, compute };
  let tools = new LocalProductTools(options);
  const save = () => sessions.save(session);
  const call = (name: string, args = {}) => tools.execute(name, args, { session, userMessage: '推荐并配置多个模型', save });
  t.after(() => { sessions.close(); rmSync(home, { recursive: true, force: true }); });
  await tools.attach('/fixture.csv', session); await call('run_create');
  await call('training_configure', { plans: [plan('lda', { num_topics: 7, max_iter: 5 }), plan('prodlda', { num_topics: 9, epochs: 20 })] });
  const original = session.pendingConfirmation!;
  const beforeSubmission = structuredClone(session);
  assert.equal(original.kind, 'training_config'); assert.equal(submissions.length, 0);
  assert.deepEqual((await tools.trainingEditor(session) as any).plans.map((p: any) => p.params.num_topics), [7, 9]);
  const edited = { checkpointId: original.checkpointId, expectedContentHash: original.contentHash,
    plans: [plan('lda', { num_topics: 10, alpha: 0.15, max_iter: 2 }), plan('prodlda', { num_topics: 12, epochs: 3 }), plan('dtm', { num_topics: 5 })] };
  await assert.rejects(tools.submitTrainingConfiguration(session, edited, save), /时间列/u);
  assert.equal(submissions.length, 0, 'A later invalid model must not start the earlier ones');
  assert.equal(session.pendingConfirmation?.checkpointId, original.checkpointId);
  edited.plans[2] = { ...edited.plans[2], timeColumn: '时间' } as any;
  await tools.submitTrainingConfiguration(session, edited, save);
  assert.equal(submissions.length, 1); assert.equal(session.pendingConfirmation, undefined);
  assert.ok(submissions[0].runId.length <= 36, 'Worker dataset.project_id contract');
  assert.equal(submissions[0].plan.params.alpha, 0.15); assert.equal(submissions[0].plan.params.epochs, undefined);
  await tools.submitTrainingConfiguration(session, edited, save);
  assert.equal(submissions.length, 1);
  session = beforeSubmission;
  await tools.submitTrainingConfiguration(session, edited, save);
  assert.equal(submissions.length, 1, 'Recover a saved batch if the process died before the session save');
  assert.equal(session.pendingConfirmation, undefined);
  assert.equal(session.trainingBatchIds?.length, 1);
  await assert.rejects(tools.submitTrainingConfiguration(session, { ...edited, plans: [plan('lda')] }, save), /旧卡/u);

  session = sessions.get(session.id); tools = new LocalProductTools(options);
  statusUnavailable = true; await call('run_status');
  assert.equal(session.monitorTraining, true, 'Transient status failures retain the running queue');
  assert.equal(submissions.length, 1, 'An unreadable running job cannot dispatch the next model');
  statusUnavailable = false;
  await call('run_status'); assert.equal(submissions.length, 1, 'Refreshing while one model runs cannot dispatch another');
  jobs.set(submissions[0].jobId, { id: submissions[0].jobId, status: 'completed', phase: '完成', percent: 100 });
  await call('run_status'); assert.equal(submissions.length, 2);
  assert.equal(submissions[1].plan.params.epochs, 3); assert.equal(submissions[1].plan.params.alpha, undefined);
  jobs.set(submissions[1].jobId, { id: submissions[1].jobId, status: 'completed', phase: '完成', percent: 100 });
  await call('run_status'); assert.equal(submissions.length, 3); assert.equal(submissions[2].plan.timeColumn, '时间');
  jobs.set(submissions[2].jobId, { id: submissions[2].jobId, status: 'completed', phase: '完成', percent: 100 });
  await call('run_status'); assert.equal(session.monitorTraining, false);
  assert.equal((tools.readState(session) as any).lastObservedJob.percent, 100);
  assert.equal((await tools.resultCatalog(session)).results.length, 3);

  await call('run_create'); await call('training_configure', { plans: [plan('lda'), plan('btm')] });
  const pending = session.pendingConfirmation!;
  await tools.submitTrainingConfiguration(session, { checkpointId: pending.checkpointId, expectedContentHash: pending.contentHash, plans: pending.trainingPlans! }, save);
  assert.equal(submissions.length, 4);
  await call('training_cancel'); await tools.approve(session, '确认', undefined, save);
  await call('run_status'); assert.equal(submissions.length, 4, 'Cancelled queue never starts the second model');
  assert.equal(new ResearchStore(home).get<any>('run', session.runIds!.at(-1)!).lastObservedJob.status, 'cancelled');

  await call('run_create'); await call('training_configure', { plans: [plan('lda'), plan('theta', { mode: 'zero_shot' })] });
  assert.equal(session.pendingConfirmation?.kind, 'embedding_choice');
  await tools.selectEmbedding(session, 'cloud', undefined, save);
  assert.equal(session.pendingConfirmation?.kind, 'training_config');
  assert.deepEqual(session.pendingConfirmation?.trainingPlans?.map(p => p.modelId), ['lda', 'theta']);
  const cloudCard = session.pendingConfirmation!;
  const cloudInput = { checkpointId: cloudCard.checkpointId, expectedContentHash: cloudCard.contentHash, plans: cloudCard.trainingPlans! };
  await assert.rejects(tools.submitTrainingConfiguration(session, cloudInput, save), /明确勾选/u);
  await assert.rejects(tools.submitTrainingConfiguration(session, { ...cloudInput, cloudConfirmed: true, cloudSelection: { provider: 'fixture', model: 'changed', endpoint: 'https://example.invalid/v1' } }, save), /服务已变化/u);
  assert.equal(submissions.length, 4);
  await tools.submitTrainingConfiguration(session, { ...cloudInput, cloudConfirmed: true, cloudSelection: { provider: 'fixture', model: 'embed', endpoint: 'https://example.invalid/v1' } }, save);
  assert.equal(submissions.length, 5);
  jobs.set(submissions[4].jobId, { id: submissions[4].jobId, status: 'completed', phase: '完成', percent: 100 });
  await call('run_status'); assert.equal(submissions.length, 6); assert.equal(submissions[5].plan.params.embedding_provider, 'cloud');
  assert.equal(new ResearchStore(home).get<any>('run', submissions[5].runId).plan.modelId, 'theta');
});
