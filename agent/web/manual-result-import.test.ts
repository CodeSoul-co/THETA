import test, { type TestContext } from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, rmSync, symlinkSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { once } from 'node:events';
import { DatabaseSync } from 'node:sqlite';
import { createAgentServer } from './server.js';
import { ResearchStore } from '../src/memory/research-store.js';
import { ProductSessionStore } from '../src/memory/session-store.js';
import { manualResultImporter } from './manual-result-import.js';
import type { ResearchRun } from '../src/domain/research.js';
import { LocalProductTools } from '../src/tools/local-tools.js';

function fixture(t: TestContext) {
  const root = mkdtempSync(path.join(tmpdir(), 'theta-manual-copy-'));
  const home = path.join(root, 'agent'); const manualHome = path.join(root, 'manual');
  mkdirSync(manualHome); mkdirSync(home);
  const sha = (value: string) => createHash('sha256').update(value).digest('hex');
  const dataset = { datasetRef: 'dataset-original', fileName: '数据.csv', sha256: sha('text\n测试\n'), sizeBytes: 12, managedPath: path.join(manualHome, '数据.csv') };
  writeFileSync(dataset.managedPath, 'text\n测试\n');
  const db = new DatabaseSync(path.join(manualHome, 'manual.sqlite'));
  db.exec('CREATE TABLE records(id INTEGER PRIMARY KEY,kind TEXT,value TEXT)');
  const add = (id: number, kind: string, value: unknown) => db.prepare('INSERT INTO records VALUES (?,?,?)').run(id, kind, JSON.stringify(value));
  const ids = ['a', 'b', 'c'].map(char => `job-${char.repeat(64)}`);
  add(1, 'project', { name: '手动结果', dataset_name: 'test' });
  add(2, 'file', { dataset_name: 'test', dataset });
  add(3, 'job', { dataset_name: 'test', workers: ids.map((id, i) => ({ id, model: ['lda', 'prodlda', 'dtm'][i] })) });
  const compute = new DatabaseSync(path.join(manualHome, 'compute.sqlite'));
  compute.exec('CREATE TABLE jobs(id TEXT PRIMARY KEY,request TEXT,state TEXT,value TEXT)');
  ids.forEach((id, i) => {
    const resultDir = path.join(manualHome, 'compute', id, 'results');
    mkdirSync(resultDir, { recursive: true });
    const artifacts = { 'topic_words.json': '{"topics":[["测试"]]}', 'topic_proportions.csv': 'topic_id,mean_weight\n1,0.6\n2,0.4\n', 'topic_proportions.png': 'original-figure' };
    for (const [name, content] of Object.entries(artifacts)) writeFileSync(path.join(resultDir, name), content);
    const plan = { modelId: ['lda', 'prodlda', 'dtm'][i], textColumn: 'text', params: { num_topics: 10, epochs: i + 2 }, rationale: '手动配置', timeoutSeconds: 600 };
    const status = i === 2 ? 'running' : 'completed';
    compute.prepare('INSERT INTO jobs VALUES (?,?,?,?)').run(id, JSON.stringify({ jobId: id, dataset, plan }), status,
      JSON.stringify({ id, status, phase: status, percent: i === 2 ? 20 : 100, plan, resultDir, resultHash: sha(Object.entries(artifacts).sort(([a], [b]) => a.localeCompare(b)).map(([name, content]) => `${name}\0${sha(content)}\n`).join('')) }));
  });
  compute.close(); db.close();
  const records = new ResearchStore(home); const sessions = new ProductSessionStore(home);
  const importer = manualResultImporter(home, manualHome, records, sessions);
  t.after(() => { sessions.close(); rmSync(root, { recursive: true, force: true }); });
  return { home, manualHome, ids, dataset, records, sessions, importer };
}

test('copies every completed model, data and parameters independently; repeat/restart reuses the project', async t => {
  const f = fixture(t);
  const input = { projectId: 1, selectedJobId: f.ids[1] };
  const [first, concurrent] = await Promise.all([f.importer(input, 'local'), f.importer(input, 'local')]);
  assert.equal(first.runId, concurrent.runId);
  assert.equal(first.copiedJobs, 2);
  const session = f.sessions.get(first.runId);
  assert.equal(session.pendingConfirmation, undefined);
  assert.equal(session.trainingBatchIds, undefined);
  assert.equal(session.messages.length, 1);
  assert.equal(session.messages[0].role, 'assistant');
  const runs = session.runIds!.map(id => f.records.get<ResearchRun>('run', id));
  assert.deepEqual(runs.map(run => run.plan?.modelId), ['lda', 'prodlda']);
  assert.equal(runs[1].plan?.params.epochs, 3);
  assert.equal(session.runId, runs[1].id);
  assert.notEqual(runs[0].activeJob, f.ids[0]);
  const copiedFile = session.reports![0].files.find(file => file.name === 'topic_words.json')!.path;
  assert.ok(copiedFile.startsWith(path.dirname(session.reports![0].reportPath) + path.sep));
  assert.ok(!session.reports![0].reportPath.startsWith(path.join(f.home, 'compute')));
  writeFileSync(copiedFile, '修改副本');
  assert.match(readFileSync(path.join(f.manualHome, 'compute', f.ids[0], 'results', 'topic_words.json'), 'utf8'), /topics/);
  const copiedDataset = f.records.get<typeof f.dataset>('dataset', session.datasetRefs[0]);
  assert.notEqual(copiedDataset.managedPath, f.dataset.managedPath);
  assert.equal(readFileSync(copiedDataset.managedPath, 'utf8'), readFileSync(f.dataset.managedPath, 'utf8'));
  const again = await manualResultImporter(f.home, f.manualHome, f.records, f.sessions)(input, 'local');
  assert.equal(again.reused, true); assert.equal(again.runId, first.runId);
  assert.equal(f.records.list('web-project').length, 1);
  const legacyAlias = await f.importer({ datasetName: 'test' }, 'local');
  assert.equal(legacyAlias.runId, first.runId);
});

test('imports an older dataset-only project using its registered result group', async t => {
  const f = fixture(t);
  const db = new DatabaseSync(path.join(f.manualHome, 'manual.sqlite'));
  db.prepare("DELETE FROM records WHERE kind='project'").run(); db.close();
  const result = await f.importer({ datasetName: 'test' }, 'local');
  assert.equal(result.copiedJobs, 2);
  assert.equal(result.project.name, 'test · 对话副本');
});

test('rejects running, foreign, missing and tampered results without creating a project', async t => {
  const f = fixture(t);
  await assert.rejects(f.importer({ projectId: 1, selectedJobId: f.ids[2] }, 'local'), /尚未完成/);
  await assert.rejects(f.importer({ projectId: 1, selectedJobId: 'other' }, 'local'), /不属于/);
  await assert.rejects(f.importer({ projectId: 999 }, 'local'), /不存在/);
  writeFileSync(path.join(f.manualHome, 'compute', f.ids[0], 'results', 'topic_words.json'), '篡改');
  await assert.rejects(f.importer({ projectId: 1 }, 'local'), /校验/);
  assert.equal(f.records.list('web-project').length, 0);
});

test('does not follow symlinks outside the source workspace', async t => {
  const f = fixture(t);
  symlinkSync(f.dataset.managedPath, path.join(f.manualHome, 'compute', f.ids[0], 'outside.csv'));
  await assert.rejects(f.importer({ projectId: 1 }, 'local'), /符号链接/);
  assert.equal(f.records.list('web-project').length, 0);
});

test('HTTP copy exposes durable conversation, dataset and result downloads without inference or retraining', async t => {
  let server: ReturnType<typeof createAgentServer> | undefined;
  // after hooks run in registration order; close SQLite before removing the fixture on Windows.
  t.after(async () => {
    if (!server) return;
    server.closeAllConnections();
    await new Promise<void>(resolve => server!.close(() => resolve()));
  });
  const f = fixture(t);
  let inference = 0;
  server = createAgentServer(f.home, () => { inference++; return undefined; }, { manualHome: f.manualHome });
  server.listen(0, '127.0.0.1'); await once(server, 'listening');
  const base = `http://127.0.0.1:${(server.address() as { port: number }).port}`;
  const response = await fetch(base + '/api/v3/projects/import-manual-results', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ projectId: 1 }) });
  assert.equal(response.status, 200);
  const { data: imported } = await response.json();
  const get = async (route: string) => (await (await fetch(base + route)).json()).data;
  const catalog = await get(`/api/v3/runs/${imported.runId}/results`);
  assert.equal(catalog.results.length, 2);
  assert.ok(catalog.results.every((result: {status:string}) => result.status === 'completed'));
  const artifact = await fetch(base + catalog.results[0].artifacts.files.find((file: {name:string}) => file.name === 'topic_words.json').url);
  assert.equal(artifact.status, 200); assert.match(await artifact.text(), /topics/);
  assert.equal((await get(`/api/v3/runs/${imported.runId}/conversation`)).messages.length, 1);
  assert.equal((await get(`/api/v3/datasets?projectId=${imported.project.id}`)).datasets.length, 1);
  assert.equal(inference, 0);
});

test('copied figures without native plotting provenance cannot be replaced by guessed CSV charts', async t => {
  const f = fixture(t);
  const imported = await f.importer({ projectId: 1 }, 'local');
  const session = f.sessions.get(imported.runId);
  const report = session.reports![0];
  const tools = new LocalProductTools({ runtimeDb: path.join(f.home, 'runtime.sqlite'), uploadDir: path.join(f.home, 'uploads') });
  await assert.rejects(tools.execute('figure_adjust', { jobId: report.jobId, figure: 'topic_proportions.png',
    spec: { title: '主题占比（副本验收）' } },
    { session, userMessage: '在副本中修改图表', save: () => f.sessions.save(session) }), /缺少原生绘图记录/);
  assert.equal(readFileSync(path.join(f.manualHome, 'compute', f.ids[0], 'results', 'topic_proportions.png'), 'utf8'), 'original-figure');
  assert.equal(readFileSync(path.join(f.home, 'compute', report.jobId, 'results', 'topic_proportions.png'), 'utf8'), 'original-figure');
  assert.ok(!f.sessions.get(imported.runId).reports![0].files.some(file => file.name.startsWith('adjustments/')));
});


test('imports verified Windows trees with mixed-case and nested names, but rejects changed bytes', async t => {
  const f = fixture(t);
  const sha = (value: string | Buffer) => createHash('sha256').update(value).digest('hex');
  const db = new DatabaseSync(path.join(f.manualHome, 'compute.sqlite'));
  for (const id of f.ids.slice(0, 2)) {
    const row = db.prepare('SELECT value FROM jobs WHERE id=?').get(id)!;
    const job = JSON.parse(String(row.value));
    const entries = ['topic_words.json', 'topic_proportions.csv', 'topic_proportions.png', 'README.md', 'zh/overview.csv', 'zh.index.html'];
    mkdirSync(path.join(job.resultDir, 'zh'));
    for (const name of entries.slice(3)) writeFileSync(path.join(job.resultDir, name), 'verified original');
    // Equivalent to Python sorted(PureWindowsPath(...)): directory components
    // are compared case-insensitively, but the hashed names retain their case.
    const ordered = ['README.md', 'topic_proportions.csv', 'topic_proportions.png', 'topic_words.json', 'zh/overview.csv', 'zh.index.html'];
    job.resultHash = sha(ordered.map(name => `${name}\0${sha(readFileSync(path.join(job.resultDir, name)))}\n`).join(''));
    db.prepare('UPDATE jobs SET value=? WHERE id=?').run(JSON.stringify(job), id);
  }
  db.close();
  const imported = await f.importer({ projectId: 1 }, 'local');
  assert.equal(imported.copiedJobs, 2);
  assert.ok(f.sessions.get(imported.runId).reports![0].files.some(file => file.name === 'README.md'));
  writeFileSync(path.join(f.manualHome, 'compute', f.ids[0], 'results', 'README.md'), 'changed bytes');
  await assert.rejects(f.importer({ projectId: 1 }, 'another-owner'), /校验/);
});
