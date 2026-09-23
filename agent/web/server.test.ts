import test, { type TestContext } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, writeFileSync, mkdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { once } from 'node:events';
import { createAgentServer } from './server.js';
import { MiniMaxInferenceProvider } from '../src/providers/minimax.js';
import { ProductSessionStore } from '../src/memory/session-store.js';
import { ResearchStore } from '../src/memory/research-store.js';
import { DatabaseSync } from 'node:sqlite';
async function fixture(t: TestContext) {
  const home = mkdtempSync(path.join(tmpdir(), 'theta-web-test-'));
  let calls = 0; let lastRequest = '';
  const provider = new MiniMaxInferenceProvider({ apiKey: 'test', baseUrl: 'https://example.invalid/v1', model: 'fixture', fetchImpl: async (_url, init) => {
    calls++; lastRequest = String(init?.body);
    if (JSON.parse(lastRequest).messages.at(-2)?.content === 'fail') throw new Error('fixture response failure');
    if (JSON.parse(lastRequest).messages.at(-2)?.content === 'slow') await new Promise<void>((resolve, reject) => {
      const timer = setTimeout(resolve, 5000); init?.signal?.addEventListener('abort', () => { clearTimeout(timer); reject(new Error('aborted')); }, { once: true });
    });
    return new Response(JSON.stringify({ choices: [{ message: { content: '已收到，继续讨论。' } }] }), { status: 200 });
  } });
  const server = createAgentServer(home, () => provider); server.listen(0, '127.0.0.1'); await once(server, 'listening');
  const base = `http://127.0.0.1:${(server.address() as { port: number }).port}/api/v3`;
  t.after(async () => { server.closeAllConnections(); await new Promise<void>(resolve => server.close(() => resolve())); rmSync(home, { recursive: true, force: true }); });
  const request = async (route: string, body?: unknown, method = body === undefined ? 'GET' : 'POST') => {
    const res = await fetch(base + route, { method, ...(body === undefined ? {} : { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }) });
    return { status: res.status, ...(await res.json()) };
  };
  const project = (await request('/projects', { name: '测试项目' })).data;
  const run = (await request('/runs', { projectId: project.id, researchGoal: '首次消息' })).data;
  return { home, base, request, project, run, calls: () => calls, lastRequest: () => lastRequest };
}
test('uploaded datasets remain scoped to their project', async t => {
  const f = await fixture(t);
  const other = (await f.request('/projects', { name: '另一个项目' })).data;
  const body = new FormData(); body.append('file', new File(['text,year\nOPC主题研究,2026'], 'OPCdata.csv', { type: 'text/csv' }));
  const uploaded = await fetch(`${f.base}/datasets/upload?projectId=${encodeURIComponent(f.project.id)}`, { method: 'POST', body });
  assert.equal(uploaded.status, 200);
  const first = (await f.request(`/datasets?projectId=${encodeURIComponent(f.project.id)}`)).data.datasets;
  const second = (await f.request(`/datasets?projectId=${encodeURIComponent(other.id)}`)).data.datasets;
  assert.deepEqual(first.map((dataset: {displayName:string}) => dataset.displayName), ['OPCdata.csv']);
  assert.deepEqual(second, []);
});
test('failed inference stays visible after history refresh', async t => {
  const f = await fixture(t);
  assert.equal((await f.request(`/runs/${f.run.runId}/messages`, { content: 'fail' })).status, 400);
  const messages = (await f.request(`/runs/${f.run.runId}/conversation`)).data.messages;
  assert.ok(messages.some((m: { messageKind?: string; content: string }) => m.messageKind === 'conversation.error' && m.content.includes('本轮回复未能完成')));
});
test('first message is delivered once; history and rename survive reload', async t => {
  const f = await fixture(t); assert.equal(f.calls(), 0);
  assert.equal((await f.request(`/runs/${f.run.runId}/messages`, { content: '首次消息' })).status, 200); assert.equal(f.calls(), 1);
  const history = (await f.request(`/runs/${f.run.runId}/conversation`)).data.messages;
  assert.deepEqual(history.map((m: { role: string }) => m.role), ['user', 'assistant']); assert.equal(history[0].content, '首次消息');
  const counts = (await f.request(`/runs/${f.run.runId}`)).data;
  assert.equal(counts.userMessageCount, 1); assert.equal(counts.messageCount, 1);
  await f.request(`/runs/${f.run.runId}`, { displayName: '已改名' }, 'PATCH'); assert.equal((await f.request('/runs')).data.runs[0].conversationTitle, '已改名');
});
test('topic model preference is validated and reaches plan context without approving execution', async t => {
  const f = await fixture(t); const route = `/runs/${f.run.runId}/messages`;
  assert.equal((await f.request(route, { content: '准备主题研究方案', modelPreference: 'unknown-model' })).status, 400);
  assert.equal(f.calls(), 0);
  assert.equal((await f.request(route, { content: '准备主题研究方案', modelPreference: 'lda' })).status, 200);
  const request = JSON.parse(f.lastRequest());
  assert.match(JSON.stringify(request.messages), /主题模型偏好为 \\"lda\\"/u);
  assert.match(JSON.stringify(request.messages), /不代表计算授权/u);
});
test('manual advisory is stateless, tool-free and interprets source chart data without vision input', async t => {
  const f = await fixture(t);
  const rejectedImage = await f.request('/advisory', {
    message: '解读这张图',
    images: [{ name: 'topics.png', dataUrl: 'data:image/png;base64,aGVsbG8=' }],
  });
  assert.equal(rejectedImage.status, 400);
  assert.match(rejectedImage.error.message, /CSV\/JSON/u);
  const result = await f.request('/advisory', {
    message: '解读这张主题分布图',
    context: { project: 'OPC', interactionMode: 'consultation_only' },
    charts: [{
      chartName: '主题占比分布.png',
      chartPath: 'global/主题占比分布.png',
      dataset: 'OPC',
      model: 'theta',
      sources: [{ name: 'topic_proportions.csv', path: 'global/topic_proportions.csv', format: 'csv', content: 'topic_id,mean_weight\n1,0.6\n2,0.4', truncated: false, totalCharacters: 37 }],
    }],
  });
  assert.equal(result.status, 200);
  assert.equal(result.data.message, '已收到，继续讨论。');
  const providerRequest = JSON.parse(f.lastRequest());
  assert.equal(providerRequest.tools, undefined);
  assert.equal(providerRequest.tool_choice, undefined);
  assert.equal(typeof providerRequest.messages.at(-1).content, 'string');
  assert.match(providerRequest.messages.at(-1).content, /topic_id,mean_weight/u);
  assert.doesNotMatch(providerRequest.messages.at(-1).content, /image_url|data:image/u);
  assert.match(providerRequest.messages[0].content, /只读咨询与结果解读助手/u);
  assert.equal((await f.request('/runs')).data.runs.length, 1);
  assert.deepEqual((await f.request(`/runs/${f.run.runId}/conversation`)).data.messages, []);
});
test('sidebar advisory stores actual messages and charts; pin/delete and migration survive HTTP reads', async t => {
  const f = await fixture(t);
  const route = '/consultations?scope=manual%3Aproject-1';
  const created = await f.request(route, { id: 'consult-test' });
  assert.equal(created.status, 200);
  assert.equal(created.data.activeId, 'consult-test');
  const body = {
    message: '请解释主题占比', context: { project: 'OPC' },
    consultation: { scope: 'manual:project-1', id: 'consult-test', requestId: 'request-one' },
    charts: [{ chartName: '主题占比', sources: [{ name: '占比.csv', format: 'csv', content: 'topic,weight\n1,1' }] }],
  };
  assert.equal((await f.request('/advisory', body)).status, 200);
  assert.equal((await f.request('/advisory', body)).status, 200);
  assert.equal(f.calls(), 1);
  const thread = (await f.request(route)).data.threads[0];
  assert.deepEqual(thread.messages.map((m: {role:string}) => m.role), ['user', 'ai']);
  assert.equal(thread.messages[1].content, '已收到，继续讨论。');
  assert.equal(thread.messages[0].data.charts[0].sources[0].content, 'topic,weight\n1,1');
  await f.request('/consultations/consult-test?scope=manual%3Aproject-1', { pinned: true }, 'PATCH');
  await f.request('/consultations/consult-test?scope=manual%3Aproject-1', undefined, 'DELETE');
  await f.request('/consultations/import?scope=manual%3Aproject-1', { threads: [{ ...thread, messages: [] }] });
  const deleted = (await f.request(route)).data;
  assert.equal(deleted.activeId, '');
  assert.ok(deleted.threads[0].deletedAt);
  assert.equal(deleted.threads[0].pinned, true);
  assert.equal(deleted.threads[0].messages.length, 2);
  assert.equal((await f.request('/advisory', { ...body, consultation: { ...body.consultation, requestId: 'new-request' } })).status, 409);
  await f.request('/consultations/consult-test?scope=manual%3Aproject-1', { deleted: false, select: true }, 'PATCH');
  assert.equal((await f.request(route)).data.activeId, 'consult-test');
  assert.equal((await f.request('/consultations?scope=manual%3Aother')).data.threads.length, 0);
  assert.equal((await f.request('/runs')).data.runs.length, 1);
  assert.equal((await f.request('/consultations/consult-test?scope=manual%3Aproject-1', { pinned: 'yes' }, 'PATCH')).status, 400);
});
test('result API separates exact manual or automatic selection from interpretation', async t => {
  const f = await fixture(t); const sessions = new ProductSessionStore(f.home); const session = sessions.get(f.run.runId);
  session.runId = 'research-selection'; session.runIds = ['research-selection'];
  const reportDirectory = path.join(f.home, 'selection-report'); mkdirSync(reportDirectory);
  const reportPath = path.join(reportDirectory, 'index.html'); const figurePath = path.join(reportDirectory, 'figure.svg'); const tablePath = path.join(reportDirectory, 'table.csv');
  writeFileSync(reportPath, '<h1>Result</h1>'); writeFileSync(figurePath, '<svg/>'); writeFileSync(tablePath, 'topic,weight\n1,1');
  session.reports = [{ jobId: 'job-latest', reportPath, reportStatus: 'complete', files: [{ name: 'figure.svg', path: figurePath, kind: 'figure' }, { name: 'table.csv', path: tablePath, kind: 'table' }] }];
  new ResearchStore(f.home).put('run', session.runId, {id:session.runId,goal:'compare',plan:{modelId:'lda',textColumn:'text',params:{num_topics:2},rationale:'test',timeoutSeconds:30},jobs:['job-first','job-latest','job-running'],activeJob:'job-running',notes:[]});
  sessions.save(session); sessions.close();
  const db = new DatabaseSync(path.join(f.home,'compute.sqlite'));
  db.exec('CREATE TABLE jobs (id TEXT PRIMARY KEY, request TEXT NOT NULL, state TEXT NOT NULL, value TEXT NOT NULL, cancel INTEGER NOT NULL DEFAULT 0, updated REAL NOT NULL)');
  const insert = db.prepare('INSERT INTO jobs (id,request,state,value,updated) VALUES (?,?,?,?,?)');
  for (const [id,status] of [['job-first','completed'],['job-latest','completed'],['job-running','running']]) insert.run(id,'{}',status,JSON.stringify({id,status,phase:status,percent:status==='completed'?100:50,plan:{modelId:'lda'}}),Date.now()/1000);
  db.close();
  const route = `/runs/${f.run.runId}`;
  const empty = await f.request(`${route}/results`); assert.equal(empty.status,200); assert.equal(empty.data.results.length,3); assert.equal(f.calls(),0);
  const hydrated = empty.data.results.find((result: {jobId:string}) => result.jobId === 'job-latest');
  assert.equal(hydrated.artifacts.fileCount,2); assert.equal(hydrated.artifacts.figureCount,1); assert.equal(hydrated.artifacts.tableCount,1);
  assert.deepEqual(hydrated.artifacts.files.map((file: {name:string;kind:string}) => ({name:file.name,kind:file.kind})), [
    {name:'figure.svg',kind:'figure'}, {name:'table.csv',kind:'table'},
  ]);
  assert.match(hydrated.artifacts.files[0].url,/\/artifacts\/[a-f0-9]{64}\/download$/u);
  assert.match(hydrated.artifacts.reportUrl,/\/artifacts\/[a-f0-9]{64}\/download$/u);
  assert.equal((await fetch(new URL(hydrated.artifacts.reportUrl,f.base))).status,200);
  const manual = await f.request(`${route}/result-selection`,{mode:'manual',jobIds:['job-first']});
  assert.deepEqual(manual.data.selection.resolvedJobIds,['job-first']); assert.equal(f.calls(),0,'selection does not invoke interpretation');
  assert.equal((await f.request(`${route}/result-selection`,{mode:'manual',jobIds:['job-running']})).status,400);
  const automatic = await f.request(`${route}/result-selection`,{mode:'automatic',strategy:'latest_completed'});
  assert.deepEqual(automatic.data.selection.resolvedJobIds,['job-latest']);
  await f.request(`${route}/messages`,{content:'解释选中的结果',resultSelection:{mode:'manual',jobIds:['job-first']}});
  assert.match(f.lastRequest(),/job-first/u); assert.match(f.lastRequest(),/选择本身未读取结果/u);
});
test('follow-up attachment is an attached dataset, not a catalog selection', async t => {
  const f = await fixture(t); new ResearchStore(f.home).put('dataset', 'dataset-fixture', { datasetRef: 'dataset-fixture', fileName: '测试.csv', sha256: 'abc', managedPath: '/unused', sizeBytes: 10 });
  const result = await f.request(`/runs/${f.run.runId}/messages`, { content: '看一下附件', attachments: [{ kind: 'dataset', id: 'dataset-fixture' }] });
  assert.equal(result.status, 200); assert.match(f.lastRequest(), /datasetRef.*dataset-fixture/u); assert.equal((await f.request('/runs')).data.runs.length, 1);
  assert.equal((await f.request(`/runs/${f.run.runId}/conversation`)).data.messages[0].content, '看一下附件');
});
test('stale approval fails before inference or checkpoint consumption', async t => {
  const f = await fixture(t); const store = new ProductSessionStore(f.home); const s = store.get(f.run.runId);
  s.pendingConfirmation = { kind: 'action', checkpointId: 'new-card', contentHash: 'new-hash', summary: '当前操作', runId: 'research' }; store.save(s); store.close();
  const result = await f.request(`/runs/${s.id}/checkpoint-decision`, { action: 'approve', checkpointId: 'old-card', expectedContentHash: 'old-hash' });
  assert.equal(result.status, 409); assert.equal(f.calls(), 0); assert.equal((await f.request(`/runs/${s.id}/checkpoint`)).data.checkpointId, 'new-card');
});
test('stop aborts active generation and releases session for the next turn', async t => {
  const f = await fixture(t); const pending = f.request(`/runs/${f.run.runId}/messages`, { content: 'slow' });
  while (!f.calls()) await new Promise(r => setTimeout(r, 10));
  assert.equal((await f.request(`/runs/${f.run.runId}/messages`, { content: 'duplicate' })).status, 409);
  await f.request(`/runs/${f.run.runId}/stop`, {}); await pending;
  assert.equal((await f.request(`/runs/${f.run.runId}/messages`, { content: '继续' })).status, 200);
  assert.ok((await f.request(`/runs/${f.run.runId}/conversation`)).data.messages.some((m: { content: string }) => m.content.includes('暂停')));
});
test('report links work in a browser while arbitrary local files are denied', async t => {
  const f = await fixture(t); const store = new ProductSessionStore(f.home); const s = store.get(f.run.runId);
  const directory = path.join(f.home, 'report'); mkdirSync(directory); const reportPath = path.join(directory, 'index.html'); writeFileSync(reportPath, `<img src="plot.svg"><a href="file://${directory}/plot.svg">图表</a>`); writeFileSync(path.join(directory, 'plot.svg'), '<svg/>');
  s.reports = [{ jobId: 'job', reportPath, files: [] }]; s.messages.push({ role: 'assistant', content: `[报告](<${reportPath}>)` }); store.save(s); store.close();
  const delivered = (await f.request(`/runs/${s.id}/conversation`)).data.messages[0].content;
  assert.match(delivered, /\/artifacts\/[a-f0-9]{64}\/download/u);
  const deliveredPath = /\]\((\/api\/v3\/[^)]+)\)/u.exec(delivered)?.[1]; assert.ok(deliveredPath);
  const report = await fetch(new URL(deliveredPath, f.base).toString()); assert.equal(report.status, 200); const html = await report.text(); assert.match(html, /\/artifacts\/[a-f0-9]{64}\/download/u); assert.doesNotMatch(html, /file:\/\//u);
  assert.equal((await fetch(`${f.base}/runs/${s.id}/files?path=${encodeURIComponent(path.join(f.home, 'sessions.sqlite'))}`)).status, 403);
});
test('archival persists without destroying the core history', async t => {
  const f = await fixture(t); await f.request(`/runs/${f.run.runId}/messages`, { content: '保留记录' }); await f.request(`/runs/${f.run.runId}`, undefined, 'DELETE');
  assert.equal((await f.request('/runs')).data.runs.length, 0); assert.equal((await f.request(`/runs/${f.run.runId}`)).status, 404);
  const store = new ProductSessionStore(f.home); assert.equal(store.get(f.run.runId).messages.filter(m => m.role === 'user').length, 1); store.close();
});

test('skill downloads allow only registered files of the current session', async t => {
  const f = await fixture(t); const store = new ProductSessionStore(f.home); const session = store.get(f.run.runId);
  const archive = path.join(f.home, 'template.tar.gz'); const sibling = path.join(f.home, 'private.txt');
  writeFileSync(archive, 'template'); writeFileSync(sibling, 'private');
  session.skillArtifacts = [{ name: 'template', path: archive }]; store.save(session); store.close();
  const fileUrl = (id: string, file: string) => `${f.base}/runs/${id}/files?path=${encodeURIComponent(file)}`;
  assert.equal((await fetch(fileUrl(session.id, archive))).status, 200);
  assert.equal((await fetch(fileUrl(session.id, sibling))).status, 403);
  const other = (await f.request('/runs', { researchGoal: 'Other session' })).data;
  assert.equal((await fetch(fileUrl(other.runId, archive))).status, 403);
});

async function pendingCard(f: Awaited<ReturnType<typeof fixture>>, summary = '允许执行这次本地 cpu计算？\n参数：10 次迭代') {
  const { EffectApprovals } = await import('../src/domain/effect-approval.js');
  const records = new ResearchStore(f.home);
  const store = new ProductSessionStore(f.home); const session = store.get(f.run.runId);
  session.runId = 'card-research';
  records.put('run', session.runId, { id: session.runId, goal: '卡片验证', jobs: [], notes: [] });
  const effect = new EffectApprovals(records).request(session.id, { action: 'compute.submit', target: 'local', payload: {}, summary });
  session.pendingConfirmation = { kind: 'action', runId: session.runId, checkpointId: effect.id, contentHash: effect.hash, summary };
  store.save(session); store.close();
  return { checkpointId: effect.id, expectedContentHash: effect.hash };
}
const historyCards = (response: { data: { messages: Array<{ messageKind?: string; content: string }> } }) => response.data.messages.filter(m => m.messageKind === 'activity.interaction.snapshot').map(m => JSON.parse(m.content));

test('reject is persisted once with no execution and cannot be replayed', async t => {
  const f = await fixture(t); const card = await pendingCard(f);
  const before = historyCards(await f.request(`/runs/${f.run.runId}/conversation`));
  assert.equal(before[0].resolution, 'pending');
  const route = `/runs/${f.run.runId}/checkpoint-decision`;
  assert.equal((await f.request(route, { ...card, action: 'reject' })).status, 200);
  const history = historyCards(await f.request(`/runs/${f.run.runId}/conversation`));
  assert.equal(history.length, 1); assert.equal(history[0].resolution, 'rejected'); assert.match(history[0].feedback, /拒绝/u);
  assert.equal((await f.request(route, { ...card, action: 'approve' })).status, 409);
  assert.equal(new ResearchStore(f.home).list('job').length, 0);
  assert.equal((await f.request(`/runs/${f.run.runId}/checkpoint`)).data, null);
});
test('revision requires feedback, preserves the exact feedback, and invalidates the old card', async t => {
  const f = await fixture(t); const card = await pendingCard(f); const route = `/runs/${f.run.runId}/checkpoint-decision`;
  assert.equal((await f.request(route, { ...card, action: 'revise', feedback: '  ' })).status, 400);
  assert.equal(f.calls(), 0);
  const feedback = '改为 5 次迭代\n其他保持不变';
  assert.equal((await f.request(route, { ...card, action: 'revise', feedback })).status, 200);
  const history = historyCards(await f.request(`/runs/${f.run.runId}/conversation`));
  assert.equal(history[0].resolution, 'revised'); assert.equal(history[0].feedback, feedback);
  assert.match(f.lastRequest(), /新的确认卡/u);
  assert.equal((await f.request(route, { ...card, action: 'approve' })).status, 409);
});
test('unknown actions and mismatched hashes never consume a pending card', async t => {
  const f = await fixture(t); const card = await pendingCard(f); const route = `/runs/${f.run.runId}/checkpoint-decision`;
  assert.equal((await f.request(route, { ...card, action: 'delete', feedback: '拒绝' })).status, 400);
  assert.equal((await f.request(route, { ...card, action: 'reject', expectedContentHash: 'wrong' })).status, 409);
  assert.equal((await f.request(`/runs/${f.run.runId}/checkpoint`)).data.checkpointId, card.checkpointId);
  assert.equal(f.calls(), 0);
});
test('a replaced card is historical and repeated snapshots do not duplicate or move it', async t => {
  const f = await fixture(t); await pendingCard(f, '原方案');
  const first = (await f.request(`/runs/${f.run.runId}/conversation`)).data.messages[0];
  await pendingCard(f, '新方案');
  const history = await f.request(`/runs/${f.run.runId}/conversation`);
  assert.equal(historyCards(history).length, 2); assert.equal(historyCards(history)[0].resolution, 'superseded');
  assert.equal(history.data.messages[0].createdAt, first.createdAt);
  assert.deepEqual((await f.request(`/runs/${f.run.runId}/conversation`)).data.messages, history.data.messages);
});
test('approval execution errors retain the pending card without a false approved receipt', async t => {
  const f = await fixture(t); const card = await pendingCard(f);
  // No valid plan exists: the real tool must reject before submitting computation.
  assert.equal((await f.request(`/runs/${f.run.runId}/checkpoint-decision`, { ...card, action: 'approve' })).status, 400);
  const history = historyCards(await f.request(`/runs/${f.run.runId}/conversation`));
  assert.equal(history[0].resolution, 'pending'); assert.match(history[0].error, /具体模型/u); assert.equal(f.calls(), 0);
});
test('expired confirmation is displayed as expired and is never approved', async t => {
  const f = await fixture(t); const card = await pendingCard(f);
  const records = new ResearchStore(f.home);
  const effect = records.get<Record<string, unknown>>('effect-approval', card.checkpointId);
  records.put('effect-approval', card.checkpointId, { ...effect, expiresAt: Date.now() - 1000 });
  const history = historyCards(await f.request(`/runs/${f.run.runId}/conversation`));
  assert.equal(history[0].resolution, 'expired');
  assert.equal((await f.request(`/runs/${f.run.runId}/checkpoint-decision`, { ...card, action: 'approve' })).status, 409);
  assert.equal(historyCards(await f.request(`/runs/${f.run.runId}/conversation`))[0].resolution, 'expired');
});

test('analysis mode persists per conversation and cannot change under pending approval or interpretation',async t=>{
  const f=await fixture(t);const route=`/runs/${f.run.runId}`;
  assert.equal((await f.request(route)).data.analysisMode,'topic');
  assert.equal((await f.request(route,{analysisMode:'free'},'PATCH')).data.analysisMode,'free');
  assert.equal((await f.request(route)).data.analysisMode,'free');
  assert.equal((await f.request(route,{analysisMode:'invalid'},'PATCH')).status,400);
  const sessions=new ProductSessionStore(f.home);const s=sessions.get(f.run.runId);
  s.pendingConfirmation={kind:'action',checkpointId:'pending',contentHash:'hash',summary:'Test scope',runId:s.id};sessions.save(s);sessions.close();
  assert.equal((await f.request(route,{analysisMode:'topic'},'PATCH')).status,409);
  assert.equal((await f.request(route)).data.analysisMode,'free');
  const resumed=new ProductSessionStore(f.home);const interrupted=resumed.get(f.run.runId);
  interrupted.pendingConfirmation=undefined;interrupted.pendingStatisticalInterpretation='stats-interrupted';resumed.save(interrupted);resumed.close();
  assert.equal((await f.request(route,{analysisMode:'topic'},'PATCH')).status,409);
});

test('background message acknowledges before completion, deduplicates and survives reload until stopped',async t=>{
  const f=await fixture(t);const route=`/runs/${f.run.runId}`;
  const body={content:'slow',async:true,requestId:'test-background-one'};
  const accepted=await f.request(`${route}/messages`,body);
  assert.equal(accepted.status,202);assert.equal(accepted.data.task.status,'running');
  const taskId=accepted.data.task.id;
  assert.equal((await f.request(route)).data.task.id,taskId);
  assert.equal((await f.request(`${route}/messages`,body)).data.task.id,taskId);
  assert.equal((await f.request(`${route}/messages`,{...body,content:'different'})).status,409);
  assert.equal((await f.request(`${route}/messages`,{...body,requestId:'test-background-two'})).status,409);
  await f.request(`${route}/stop`,{});
  let task;
  for(let n=0;n<500;n++){
    task=(await f.request(`${route}/tasks/${taskId}`)).data;
    if(!['queued','running'].includes(task.status))break;
    await new Promise(r=>setTimeout(r,20));
  }
  assert.equal(task.status,'cancelled');assert.equal(f.calls(),1);
  assert.equal((await f.request(`${route}/messages`,body)).data.task.id,taskId);
  assert.equal(f.calls(),1);
});

test('background inference failure remains queryable and does not claim completion',async t=>{
  const f=await fixture(t);const route=`/runs/${f.run.runId}`;
  const accepted=await f.request(`${route}/messages`,{content:'fail',async:true,requestId:'test-background-failure'});
  assert.equal(accepted.status,202);
  let task;
  for(let n=0;n<500;n++){
    task=(await f.request(`${route}/tasks/${accepted.data.task.id}`)).data;
    if(task.status==='failed')break;
    await new Promise(r=>setTimeout(r,20));
  }
  assert.equal(task.status,'failed');assert.match(task.error,/fixture response failure/);
  const other=(await f.request('/runs',{projectId:'other',researchGoal:'Other'})).data;
  assert.equal((await f.request(`/runs/${other.runId}/tasks/${task.id}`)).status,404);
});
