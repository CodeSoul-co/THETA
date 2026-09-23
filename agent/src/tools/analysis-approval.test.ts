import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { LocalProductTools } from './local-tools.js';
import { ProductSessionStore } from '../memory/session-store.js';
import type { CapabilityWorker } from '../adapters/python-worker.js';
import type { ComputeGateway } from '../adapters/compute-gateway.js';
import type { ComputeJob } from '../domain/research.js';
import { observationText } from '../conversation/conversation-agent.js';

test('publication evidence keeps raw rows and late temporal tables within the approval receipt', async () => {
  const { ResearchStore } = await import('../memory/research-store.js');
  const home = mkdtempSync(path.join(os.tmpdir(), 'theta-publication-'));
  const store = new ProductSessionStore(home); const session = store.create(); session.runId = 'run-fixture';
  new ResearchStore(home).put('run', session.runId, { id: session.runId, goal: '合成语料流程验收', jobs: ['job-fixture'], activeJob: 'job-fixture', notes: [] });
  const files = Array.from({ length: 120 }, (_, i) => ({ name: `native/chart-${i}.svg`, kind: 'figure', path: `/very/long/report/location/${'x'.repeat(180)}/native/chart-${i}.svg` }));
  const compute: ComputeGateway = {
    async submit() { throw new Error('Must not retrain'); }, async cancel() { throw new Error('Must not cancel'); },
    async status() { return { id: 'job-fixture', status: 'completed', phase: 'completed', percent: 100, resultHash: 'fixture-hash' }; },
    async results() { return { schemaVersion: 'theta.result-report.v2', reportStatus: 'complete', jobId: 'job-fixture', reportPath: '/reports/index.html', resultDir: '/training', files,
      sourceData: { matrixRowsAligned: true, rows: [{ sourceRow: 180, text: 'Verified source text' }] },
      evidence: { figures: files.map(file => ({ relativePath: file.name, format: 'svg', limitation: 'repeated manifest advice '.repeat(30), sha256: 'a'.repeat(64) })),
        tables: [{ relativePath: 'topic_weights_by_year.csv', rows: [{ year: 2025, topic_1: 0.4 }] }] } }; },
  };
  const tools = new LocalProductTools({ runtimeDb: path.join(home, 'research.sqlite'), uploadDir: path.join(home, 'uploads'), compute });
  try {
    await tools.execute('results_synthesize', { question: '逐图表解释' }, { session, userMessage: '逐图表解释', save: () => store.save(session) });
    const serialized = observationText(await tools.approve(session, '确认'), 64000);
    assert.equal(JSON.parse(serialized).truncated, undefined);
    assert.match(serialized, /Verified source text/);
    assert.match(serialized, /topic_weights_by_year/);
    assert.equal(session.reports![0].files.length, 120, 'Full clickable files remain in delivery');
  } finally { store.close(); rmSync(home, { recursive: true, force: true }); }
});

test('full approval lifecycle: train, refuse result access, revise synthesis, restore and compare studies', async () => {
  const home = mkdtempSync(path.join(os.tmpdir(), 'theta-analysis-'));
  const store = new ProductSessionStore(home); let session = store.create();
  let reads = 0, submissions = 0, hash = 'verified-v1';
  const jobs = new Map<string, ComputeJob>();
  const worker: CapabilityWorker = { async call<T>(name: string, args: unknown): Promise<T> {
    return ({
      'dataset.import': { datasetRef: 'dataset-one', sha256: 'sha-one', fileName: 'fixture.csv', managedPath: '/fixture.csv', sizeBytes: 20 },
      'plan.validate': { valid: true },
      'compute.preview': { execution: { embedding: { mode: 'local' } }, rowCount: 36, readiness: { ready: true } },
      'dataset.understand': { excerpts: [{ text: 'Late delivery and slow refunds' }] },
    }[name]) as T;
  } };
  const compute: ComputeGateway = {
    async submit(request, receipt) { assert.ok(receipt); submissions++; const job: ComputeJob = { id: request.jobId, status: 'completed', phase: 'completed', percent: 100, resultHash: hash }; jobs.set(job.id, job); return job; },
    async status(id) { return { ...jobs.get(id)!, resultHash: hash }; },
    async cancel() { throw new Error('Not running'); },
    async results(id, view) { reads++; return { jobId: id, view, schemaVersion: 'theta.result-report.v1', files: [], missingEvidence: ['fixture only'] }; },
  };
  const options = { runtimeDb: path.join(home, 'research.sqlite'), uploadDir: path.join(home, 'uploads'), worker, compute };
  let tools = new LocalProductTools(options);
  const call = (name: string, args = {}) => tools.execute(name, args, { session, userMessage: '客服改善方向', save: () => store.save(session) });
  const pending = async (name: string, args = {}) => assert.equal((await call(name, args) as { needsUser: boolean }).needsUser, true);
  try {
    await tools.attach('/fixture.csv', session);
    const run = await call('run_create') as { datasetRef: string; id: string };
    assert.equal(run.datasetRef, 'dataset-one', 'A single attached dataset survives run creation without an explicit reference');
    await call('dataset_understand'); tools.recordUnderstanding(session, '样本涉及配送与退款，需要验证用户来源。');
    const originalUnderstanding = (tools.readContext(session) as { markdown: string }).markdown;
    await call('run_update', { goal: '客服主题流程验证' });
    assert.match((tools.readContext(session) as { markdown: string }).markdown, /客服主题流程验证/);
    assert.ok(originalUnderstanding.includes('样本涉及配送'));
    assert.match((tools.readContext(session) as { markdown: string }).markdown, /样本涉及配送/);
    const plan = { modelId: 'lda', textColumn: 'text', params: { num_topics: 3 }, rationale: 'Interpretable baseline' };
    await call('plan_propose', plan); await pending('training_advance');
    assert.equal(submissions, 0); await tools.approve(session, '确认'); assert.equal(submissions, 1);
    await assert.rejects(call('run_update', { goal: 'STM实验' }), /已有计算任务/);
    await assert.rejects(call('plan_propose', { ...plan, modelId: 'stm' }), /切换模型/);
    await pending('results_read', { view: 'report' }); assert.equal(reads, 0);
    tools.deny(session); assert.equal(reads, 0);
    await call('research_read'); assert.equal(session.pendingConfirmation, undefined);
    await pending('results_read', { view: 'tables' });
    hash = 'changed'; await assert.rejects(tools.approve(session, '确认'), /结果或业务上下文已变化/); assert.equal(reads, 0);
    await pending('results_read', { view: 'report' });
    store.save(session); session = store.get(session.id); tools = new LocalProductTools(options);
    await tools.approve(session, '确认'); assert.equal(reads, 1);
    await call('results_read', { view: 'figures' }); assert.equal(reads, 2, 'Focused views are included in this report authorization');
    await pending('results_synthesize', { question: '渠道是否造成退款差异？' }); assert.equal(reads, 2);
    tools.deny(session);
    await pending('results_synthesize', { question: '只给待验证的改善假设，不作因果推断' });
    await call('context_save', { title: '新目标', goal: '聚焦退款', understanding: '无法推断因果', questions: [] });
    await assert.rejects(tools.approve(session, '确认'), /结果或业务上下文已变化/); assert.equal(reads, 2);
    await pending('results_synthesize', { question: '聚焦退款有哪些证据局限？' });
    await tools.approve(session, '确认'); tools.recordUnderstanding(session, '分布只能提示文本差异，无法证明渠道导致退款问题。');
    assert.match(readFileSync(session.interpretations![0].documentPath, 'utf8'), /无法证明/);
    assert.equal(session.pendingSynthesis, undefined);
    await assert.rejects(tools.approve(session, '确认'), /需要先展示/);
    await call('run_create', { goal: '第二个模型' }); await call('plan_propose', { ...plan, modelId: 'stm', covariates: ['channel'] });
    await pending('training_advance'); await tools.approve(session, '确认');
    const studies = await call('runs_list') as Array<{ id: string; jobs: string[] }>;
    assert.equal(studies.length, 2); assert.equal(submissions, 2);
    await pending('results_synthesize', { question: '比较两个模型，主题编号不直接对应', jobIds: studies.flatMap(item => item.jobs) });
    const result = await tools.approve(session, '确认') as { reports: unknown[] };
    assert.equal(result.reports.length, 2);
    await call('run_select', { runId: run.id }); assert.equal(session.runId, run.id);
    await assert.rejects(call('results_synthesize', { question: '读取别人结果', jobIds: ['job-other-session'] }), /本会话/);
  } finally { store.close(); rmSync(home, { recursive: true, force: true }); }
});

test('THETA asks for an embedding choice before a separate training approval, surviving reload', async () => {
  const home = mkdtempSync(path.join(os.tmpdir(), 'theta-cloud-default-'));
  const store = new ProductSessionStore(home); let session = store.create();
  let submissions = 0; let cloudConfigured = true;
  let submitted: any;
  const worker: CapabilityWorker = { async call<T>(name: string, args: unknown): Promise<T> {
    if (name === 'runtime.config') return { embedding: { preferredMode: 'cloud', configured: cloudConfigured, provider: 'zhipu', model: 'embedding-3', endpoint: 'https://example.invalid/v4' } } as T;
    if (name === 'plan.validate') return { valid: true } as T;
    if (name === 'compute.preview') { const plan = (args as any).plan; return { execution: { embedding: { mode: plan.params.embedding_provider, model: plan.params.embedding_model, endpoint: plan.params.embedding_api_base }, maxExternalRequests: plan.externalRequestLimit }, rowCount: 2400, readiness: { ready: true } } as T; }
    if (name === 'dataset.import') return { datasetRef: 'one', sha256: 'one', fileName: 'fixture.csv', managedPath: '/fixture.csv' } as T;
    throw new Error(`Unexpected effect: ${name}`);
  } };
  const compute: ComputeGateway = { async submit(request) { submissions++; submitted = request; return { id: request.jobId, status: 'completed', percent: 100, phase: '完成' }; }, async status(id) { return { id, status: 'completed', percent: 100, phase: '完成' }; }, async cancel() { throw new Error('Unexpected cancel'); }, async results() { throw new Error('Unexpected results'); } };
  const options = { runtimeDb: path.join(home, 'research.sqlite'), uploadDir: home, worker, compute };
  let tools = new LocalProductTools(options);
  const call = (name: string, args = {}) => tools.execute(name, args, { session, userMessage: '准备云embedding方案', save: () => store.save(session) });
  try {
    await tools.attach('/fixture.csv', session); await call('run_create');
    const plan = { modelId: 'theta', textColumn: 'text', params: { mode: 'zero_shot' }, rationale: '检验默认配置', externalRequestLimit: 200 };
    for (const provider of [undefined, 'cloud', 'local']) {
      const proposed = await call('plan_propose', { ...plan, params: { ...plan.params, ...(provider ? { embedding_provider: provider } : {}) } }) as { needsUser: boolean };
      assert.equal(proposed.needsUser, true, 'LLM arguments are never a user choice');
      assert.equal(session.pendingConfirmation?.kind, 'embedding_choice');
      await call('training_request_approval');
      assert.equal(session.pendingConfirmation?.kind, 'embedding_choice');
      await assert.rejects(tools.approve(session, '确认'));
    }
    session = store.get(session.id); tools = new LocalProductTools(options);
    await tools.selectEmbedding(session, 'local', undefined, () => store.save(session));
    assert.equal(session.pendingConfirmation?.kind, 'action');
    assert.equal(submissions, 0, 'Choice must never submit compute');
    assert.match(session.pendingConfirmation!.summary, /本地资源/u);
    session = store.get(session.id); tools = new LocalProductTools(options);
    await tools.approve(session, '确认');
    assert.equal(submissions, 1);
    assert.equal(submitted.plan.params.embedding_provider, 'local');

    await call('run_create');
    await call('plan_propose', { ...plan, params: { ...plan.params, embedding_api_base: 'https://llm-invented.invalid', embedding_model: 'invented' } });
    cloudConfigured = false;
    await assert.rejects(tools.selectEmbedding(session, 'cloud'), /配置未就绪/u);
    assert.equal(session.pendingConfirmation?.kind, 'embedding_choice');
    cloudConfigured = true;
    await tools.selectEmbedding(session, 'cloud', undefined, () => store.save(session));
    assert.equal(submissions, 1);
    assert.match(session.pendingConfirmation!.summary, /embedding-3/u);
    assert.match(session.pendingConfirmation!.summary, /200 次/u);
    await tools.approve(session, '确认');
    assert.equal(submissions, 2);
    assert.equal(submitted.plan.params.embedding_provider, 'cloud');
    assert.equal(submitted.plan.params.embedding_api_base, 'https://example.invalid/v4');
    assert.equal(submitted.execution.embedding.mode, 'cloud');
    assert.equal(submitted.execution.maxExternalRequests, 200);

    await call('plan_propose', { ...plan, params: { ...plan.params, num_topics: 10 } });
    const pending = structuredClone(session.pendingConfirmation);
    await call('run_create'); session.pendingConfirmation = pending;
    await assert.rejects(tools.selectEmbedding(session, 'local'), /研究或数据已变化/u);
    await call('plan_propose', { ...plan, params: { mode: 'supervised' }, labelColumn: 'label' });
    assert.equal(session.pendingConfirmation, undefined, 'Fine-tuning has no invalid cloud choice');
  } finally { store.close(); rmSync(home, { recursive: true, force: true }); }
});

test('failed report retains result location; incomplete delivery cannot trigger substantive synthesis', async () => {
  const { ResearchStore } = await import('../memory/research-store.js');
  const home = mkdtempSync(path.join(os.tmpdir(), 'theta-report-failure-'));
  const store = new ProductSessionStore(home); const session = store.create(); session.runId = 'run-fixture';
  new ResearchStore(home).put('run', session.runId, { id: session.runId, goal: '文本研究', jobs: ['job-fixture'], activeJob: 'job-fixture', notes: [] });
  let fail = true;
  const compute: ComputeGateway = {
    async submit() { throw new Error('Must not retrain'); }, async cancel() { throw new Error('Must not cancel'); },
    async status() { return { id: 'job-fixture', status: 'completed', phase: 'completed', percent: 100, resultDir: '/fixture/original-results', resultHash: 'fixture-hash' }; },
    async results() {
      if (fail) throw new Error('Missing BERTopic document assignments');
      return { schemaVersion: 'theta.result-report.v2', reportStatus: 'incomplete', jobId: 'job-fixture', resultDir: '/fixture/original-results', reportPath: '/fixture/diagnostic/index.html', logPath: '/fixture/diagnostic/visualization.log',
        files: [{ name: 'index.html', kind: 'report', path: '/fixture/diagnostic/index.html' }], evidence: {}, missingEvidence: ['Missing BERTopic document assignments'] };
    },
  };
  const observedJobs: ComputeJob[] = [];
  const tools = new LocalProductTools({ runtimeDb: path.join(home, 'research.sqlite'), uploadDir: path.join(home, 'uploads'), compute, onJobObserved: job => observedJobs.push(job) });
  const call = (name: string, args: unknown) => tools.execute(name, args, { session, userMessage: '读取结果', save: () => store.save(session) });
  try {
    const status = await call('run_status', {}) as { summary: string };
    assert.match(status.summary, /原始结果目录：\/fixture\/original-results/);
    assert.equal(observedJobs[0].resultDir, '/fixture/original-results', 'The host receives the path before any report or inference');
    await call('results_read', { view: 'report' });
    await assert.rejects(tools.approve(session, '确认'), /Missing BERTopic.*\n原始结果目录：\/fixture\/original-results/);
    assert.equal(session.pendingConfirmation, undefined);
    fail = false;
    await call('results_read', { view: 'report' });
    const report = await tools.approve(session, '确认') as { instruction: string; reports: Array<{ reportStatus: string; resultDir: string }> };
    assert.match(report.instruction, /结果整理未完成/);
    assert.equal(report.reports[0].reportStatus, 'incomplete');
    assert.equal(session.reports![0].reportStatus, 'incomplete');
    await call('results_synthesize', { question: '创业主题有什么含义？' });
    await tools.approve(session, '确认');
    assert.equal(session.pendingSynthesis, undefined, 'No synthesis agent is started for missing evidence');
  } finally { store.close(); rmSync(home, { recursive: true, force: true }); }
});
