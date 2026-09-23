import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { once } from 'node:events';
import { createAgentServer } from './server.js';
import { LocalProductTools, explicitEmbeddingChoice } from '../src/tools/local-tools.js';
import { ProductSessionStore } from '../src/memory/session-store.js';
import { ResearchStore } from '../src/memory/research-store.js';
import type { CapabilityWorker } from '../src/adapters/python-worker.js';
import type { ComputeGateway, ComputeRequest } from '../src/adapters/compute-gateway.js';
import { MiniMaxInferenceProvider } from '../src/providers/minimax.js';

test('web embedding selection persists, rejects stale cards, and cannot authorize compute', async t => {
  const home = mkdtempSync(path.join(tmpdir(), 'theta-embedding-web-'));
  let inferenceCalls = 0; const submitted: ComputeRequest[] = [];
  const worker: CapabilityWorker = { async call<T>(name: string, args: any): Promise<T> {
    if (name === 'runtime.config') return { embedding: { configured: true, provider: 'fixture', endpoint: 'https://example.invalid/v1', model: 'fixture-embedding', preferredMode: 'cloud' } } as T;
    if (name === 'dataset.import') return { datasetRef: 'fixture-dataset', sha256: 'fixture-sha', fileName: 'fixture.csv', managedPath: '/fixture.csv', sizeBytes: 12 } as T;
    if (name === 'plan.validate') return { valid: true } as T;
    if (name === 'dataset.profile') return { columns: ['text', 'year'] } as T;
    if (name === 'compute.preview') return { execution: { embedding: { mode: args.plan.params.embedding_provider ?? 'local', endpoint: args.plan.params.embedding_api_base, model: args.plan.params.embedding_model }, maxExternalRequests: args.plan.externalRequestLimit }, rowCount: 2400, readiness: { ready: true } } as T;
    throw new Error(`Unexpected operation ${name}`);
  } };
  const compute: ComputeGateway = {
    async submit(request) { submitted.push(request); return { id: request.jobId, status: 'completed', percent: 100, phase: '完成' }; },
    async status(id) { return { id, status: 'completed', percent: 100, phase: '完成' }; },
    async cancel() { throw new Error('Unexpected cancel'); }, async results() { throw new Error('Unexpected results'); },
  };
  const provider = new MiniMaxInferenceProvider({ apiKey: 'fixture', baseUrl: 'https://example.invalid/v1', model: 'fixture', fetchImpl: async () => {
    inferenceCalls++; return new Response(JSON.stringify({ choices: [{ message: { content: '已收到主机回执。' } }] }), { status: 200 });
  } });
  const server = createAgentServer(home, () => provider, {}, { worker, compute }); server.listen(0, '127.0.0.1'); await once(server, 'listening');
  const base = `http://127.0.0.1:${(server.address() as { port: number }).port}/api/v3`;
  const sessions = new ProductSessionStore(home);
  t.after(async () => { server.closeAllConnections(); await new Promise<void>(resolve => server.close(() => resolve())); sessions.close(); rmSync(home, { recursive: true, force: true }); });
  const request = async (route: string, body?: unknown) => {
    const response = await fetch(base + route, { method: body === undefined ? 'GET' : 'POST', ...(body === undefined ? {} : { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }) });
    return { status: response.status, ...await response.json() };
  };
  const project = (await request('/projects', { name: '嵌入方式回归' })).data;
  for (const mode of ['local', 'cloud'] as const) {
    const webRun = (await request('/runs', { projectId: project.id, researchGoal: '测试嵌入选择' })).data;
    const session = sessions.get(webRun.runId);
    const tools = new LocalProductTools({ runtimeDb: path.join(home, 'runtime.sqlite'), uploadDir: home, worker, compute });
    const context = { session, userMessage: '准备分析', save: () => sessions.save(session) };
    await tools.attach('/fixture.csv', session);
    await tools.execute('run_create', {}, context);
    await tools.execute('plan_propose', { modelId: 'theta', textColumn: 'text', params: { mode: 'zero_shot' }, externalRequestLimit: 200, rationale: '测试' }, context);
    const route = `/runs/${session.id}`;
    const choice = (await request(`${route}/checkpoint`)).data;
    assert.equal(choice.kind, 'embedding_choice');
    assert.equal(choice.content.cloudAvailable, true);
    const snapshot = (await request(route)).data; assert.equal(snapshot.status, 'waiting_human');
    const history = (await request(`${route}/conversation`)).data;
    assert.ok(history.messages.some((m: any) => m.content.includes('embedding_choice')));
    const decision = { checkpointId: choice.checkpointId, expectedContentHash: choice.contentHash };
    assert.equal((await request(`${route}/checkpoint-decision`, { ...decision, action: 'approve' })).status, 400);
    assert.equal((await request(`${route}/checkpoint-decision`, { ...decision, expectedContentHash: 'stale', action: `embedding_${mode}` })).status, 409);
    const count = submitted.length; const callsBefore = inferenceCalls;
    assert.equal((await request(`${route}/checkpoint-decision`, { ...decision, action: `embedding_${mode}` })).status, 200);
    assert.equal(inferenceCalls, callsBefore, 'Preference preparation needs no additional model call');
    assert.equal(submitted.length, count, 'Selecting a source is not compute authorization');
    const training = (await request(`${route}/checkpoint`)).data;
    assert.equal(training.kind, 'action'); assert.notEqual(training.checkpointId, choice.checkpointId);
    assert.match(training.view.title, /训练/u);
    const restored = sessions.get(session.id);
    const run = new ResearchStore(home).get<any>('run', restored.runId!);
    assert.equal(run.plan.params.embedding_provider, mode);
    assert.equal((await request(`${route}/checkpoint-decision`, { ...decision, action: `embedding_${mode}` })).status, 409);
    assert.equal((await request(`${route}/checkpoint-decision`, { checkpointId: training.checkpointId, expectedContentHash: training.contentHash, action: 'embedding_cloud' })).status, 409);
    assert.equal((await request(`${route}/checkpoint-decision`, { checkpointId: training.checkpointId, expectedContentHash: training.contentHash, action: 'approve' })).status, 200);
    assert.equal(submitted.length, count + 1);
    assert.equal(submitted.at(-1)!.plan.params.embedding_provider, mode);
    assert.equal((submitted.at(-1)!.execution?.embedding as any)?.mode, mode);
  }
  const webRun = (await request('/runs', { projectId: project.id, researchGoal: '可编辑多模型训练配置' })).data;
  const session = sessions.get(webRun.runId); const route = `/runs/${session.id}`;
  const tools = new LocalProductTools({ runtimeDb: path.join(home, 'runtime.sqlite'), uploadDir: home, worker, compute });
  const context = { session, userMessage: '推荐两种模型', save: () => sessions.save(session) };
  await tools.attach('/fixture.csv', session); await tools.execute('run_create', {}, context);
  const plans = [{ modelId: 'lda', textColumn: 'text', params: { num_topics: 7, max_iter: 4, 'text.stopwords': '预填\nfixture' }, rationale: '基线' }, { modelId: 'prodlda', textColumn: 'text', params: { num_topics: 9, epochs: 10 }, rationale: '神经对照' }];
  await tools.execute('training_configure', { plans }, context);
  const checkpoint = (await request(`${route}/checkpoint`)).data;
  assert.equal(checkpoint.content.trainingEditable, true);
  const editor = (await request(`${route}/training-configuration`)).data;
  assert.deepEqual(editor.columns, ['text', 'year']); assert.deepEqual(editor.plans.map((p: any) => p.params.num_topics), [7, 9]);
  assert.equal(editor.stopwords.count, 2, 'The shared stopword control retains the Agent-provided list');
  assert.equal(editor.stopwords.id, (await request(`${route}/training-configuration`)).data.stopwords.id);
  const wordResponse = await fetch(base + `${route}/training-stopwords?filename=custom.txt`, { method: 'POST', headers: { 'Content-Type': 'text/plain' }, body: '测试\nexample' });
  assert.equal(wordResponse.status, 200); const words = (await wordResponse.json()).data;
  const edited = { checkpointId: checkpoint.checkpointId, expectedContentHash: checkpoint.contentHash, stopwordsId: words.id, plans: [{ ...editor.plans[0], params: { num_topics: 10, alpha: 0.1, max_iter: 2 } }, editor.plans[1]] };
  const callsBefore = inferenceCalls; const countBefore = submitted.length;
  assert.equal((await request(`${route}/training-configuration`, { ...edited, expectedContentHash: 'stale' })).status, 400);
  assert.equal(submitted.length, countBefore);
  assert.equal((await request(`${route}/training-configuration`, edited)).status, 200);
  assert.equal(inferenceCalls, callsBefore, 'Final dialog submission bypasses inference and uses exactly the user edits');
  assert.equal(submitted.length, countBefore + 1);
  assert.equal(submitted.at(-1)!.plan.params.num_topics, 10);
  assert.equal(submitted.at(-1)!.plan.params['text.stopwords'], '测试\nexample');
  assert.equal((await request(`${route}/checkpoint`)).data, null);
  assert.equal((await request(`${route}/training-configuration`, edited)).status, 200);
  assert.equal(submitted.length, countBefore + 1, 'Repeated final clicks never retrain');
  const history = (await request(`${route}/conversation`)).data;
  assert.ok(history.messages.some((m: any) => m.messageKind === 'activity.interaction.snapshot' && JSON.parse(m.content).resolution === 'approved'));
});

test('host text choices must be explicit and cannot be inferred from quoted material or broad consent', () => {
  for (const text of ['本地嵌入', '使用本地', 'local']) assert.equal(explicitEmbeddingChoice(text), 'local');
  for (const text of ['云端嵌入', '选择云端', 'cloud']) assert.equal(explicitEmbeddingChoice(text), 'cloud');
  for (const text of ['确认', '都行', '文件里写着：云端嵌入', '不要使用云端', '本地或云端有什么区别']) assert.equal(explicitEmbeddingChoice(text), undefined);
});
