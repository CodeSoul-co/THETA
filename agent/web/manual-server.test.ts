import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, mkdtempSync, rmSync, mkdirSync, writeFileSync } from 'node:fs';
import { DatabaseSync } from 'node:sqlite';
import os from 'node:os';
import path from 'node:path';
import type { AddressInfo } from 'node:net';
import { request } from 'node:http';
import { createManualServer } from './manual-server.js';
import type { CapabilityWorker } from '../src/adapters/python-worker.js';

test('统一结果目录保留真实任务身份、文件分类及下载隔离，不触发训练或重新绘图', async () => {
  const home = mkdtempSync(path.join(os.tmpdir(), 'theta-manual-results-'));
  const states = new Map<string, any>();
  const db = new DatabaseSync(path.join(home, 'manual.sqlite'));
  db.exec('CREATE TABLE records (id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, value TEXT NOT NULL)');
  for (const version of ['old', 'new']) {
    const root = path.join(home, version);
    mkdirSync(path.join(root, 'zh', 'global'), { recursive: true });
    writeFileSync(path.join(root, 'zh', 'global', 'topic_proportions.png'), version);
    writeFileSync(path.join(root, 'zh', 'global', 'topic_proportions.csv'), `topic,value\n1,${version}`);
    writeFileSync(path.join(root, 'theta.npy'), version);
    writeFileSync(path.join(root, 'model.pt'), version);
    const id = `compute-${version}`;
    states.set(id, { id, status: 'completed', phase: 'completed', percent: 100, resultDir: root });
    db.prepare('INSERT INTO records(kind,value) VALUES (?,?)').run('job', JSON.stringify({ dataset_name: 'data', models: ['prodlda'], status: 'succeeded', workers: [{ id, model: 'prodlda' }] }));
  }
  db.close();
  const worker: CapabilityWorker = { async call<T>(operation: string, input: any): Promise<T> {
    assert.equal(operation, 'compute.status', '读取结果不能启动训练或重新绘图');
    return states.get(input.jobId) as T;
  } };
  const server = createManualServer(home, worker);
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve));
  const base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
  try {
    const catalog = await (await fetch(base + '/api/results/data/catalog')).json() as any;
    assert.equal(catalog.results.length, 2);
    for (const result of catalog.results) {
      assert.equal(result.status, 'completed');
      assert.equal(result.artifacts.figureCount, 1);
      assert.equal(result.artifacts.tableCount, 1);
      assert.equal(result.artifacts.matrixCount, 1);
      assert.equal(result.artifacts.fileCount, 4);
      assert.ok(result.artifacts.archiveUrl.includes(`job_id=${result.jobId}`));
      for (const file of result.artifacts.files) {
        const response = await fetch(base + file.url.replace('/api/backend', ''));
        assert.equal(response.status, 200);
        assert.ok((await response.text()).includes(result.jobId.replace('compute-', '')));
      }
    }
    assert.equal((await fetch(base + '/api/results/foreign/visualizations/file?job_id=compute-old&path=theta.npy')).status, 404);
    assert.equal((await fetch(base + '/api/results/data/visualizations/file?job_id=compute-old&path=../manual.sqlite')).status, 404);
    assert.deepEqual((await (await fetch(base + '/api/results/empty/catalog')).json() as any).results, []);
  } finally {
    await new Promise<void>(resolve => server.close(() => resolve()));
    assert.equal(existsSync(path.join(home, 'manual.sqlite-wal')), false, '关闭回调必须等待数据库释放，Windows 才能移动或清理数据目录');
    rmSync(home, { recursive: true, force: true });
  }
});

test('项目创建拒绝同名和并发重复，重命名受保护，数据集碰撞和归档重建保持隔离', async () => {
  const home = mkdtempSync(path.join(os.tmpdir(), 'theta-manual-projects-'));
  const worker: CapabilityWorker = { async call<T>(operation: string, input: any): Promise<T> {
    if (operation === 'dataset.import') return { datasetRef: 'project-test', sha256: 'test', managedPath: input.filePath, fileName: 'data.csv' } as T;
    throw new Error(operation);
  } };
  let server = createManualServer(home, worker);
  const listen = async () => { await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve)); return `http://127.0.0.1:${(server.address() as AddressInfo).port}`; };
  let base = await listen();
  const create = (name: string, extra = {}) => fetch(base + '/api/projects', { method: 'POST', body: JSON.stringify({ name, ...extra }) });
  const update = (id: number, input: any) => fetch(base + `/api/projects/${id}`, { method: 'PATCH', body: JSON.stringify(input) });
  try {
    const responses = await Promise.all([create('测试'), create(' 测试 '), create('测试')]);
    assert.deepEqual(responses.map(r => r.status).sort(), [201, 409, 409]);
    const original = await responses.find(r => r.status === 201)!.json() as any;
    assert.match((await responses.find(r => r.status === 409)!.json() as any).detail, /已存在同名项目/u);
    assert.equal((await (await fetch(base + '/api/projects')).json() as any[]).length, 1);
    await update(original.id, { task_id: 'existing-training', status: 'running', pipeline_status: 'running' });
    assert.equal((await create('测试')).status, 409);
    const preserved = (await (await fetch(base + '/api/projects')).json() as any[])[0];
    assert.equal(preserved.task_id, 'existing-training');
    assert.equal(preserved.status, 'running');

    const first = await (await create('A B')).json() as any;
    const second = await (await create('a_b')).json() as any;
    assert.notEqual(first.dataset_name, second.dataset_name, '不同名称归一化后不能共用数据集');
    assert.equal((await create('a b')).status, 409);
    assert.equal((await update(second.id, { name: ' A B ' })).status, 409);
    assert.equal((await update(second.id, { dataset_name: original.dataset_name })).status, 409);
    assert.equal((await update(first.id, { name: 'A B', dataset_name: first.dataset_name })).status, 200);

    await fetch(base + '/api/upload?dataset_name=orphan&filename=data.csv', { method: 'POST', body: 'text\nold file' });
    const orphan = await (await create('orphan')).json() as any;
    assert.notEqual(orphan.dataset_name, 'orphan', '不得接管已有的无项目上传');
    await fetch(base + `/api/projects/${original.id}`, { method: 'DELETE' });
    const rebuilt = await (await create('测试', { task_id: 'existing-training', archived: true })).json() as any;
    assert.notEqual(rebuilt.dataset_name, original.dataset_name);
    assert.equal(rebuilt.task_id, undefined);
    assert.equal(rebuilt.archived, undefined);
    assert.equal(rebuilt.pipeline_status, 'draft');

    await new Promise<void>(resolve => server.close(() => resolve()));
    server = createManualServer(home, worker);
    base = await listen();
    assert.equal((await create('测试')).status, 409, '服务重启后仍防重复');
    const projects = await (await fetch(base + '/api/projects')).json() as any[];
    assert.equal(projects.filter(p => p.name === '测试').length, 1);
    assert.equal(new Set(projects.map(p => p.dataset_name)).size, projects.length);
  } finally {
    await new Promise<void>(resolve => server.close(() => resolve()));
    rmSync(home, { recursive: true, force: true });
  }
});

test('手动参数读取复用模型契约，原生命名空间直接送到模型预检', async () => {
  const home = mkdtempSync(path.join(os.tmpdir(), 'theta-manual-parameters-'));
  const previews: any[] = [];
  const worker: CapabilityWorker = { async call<T>(operation: string, input: any): Promise<T> {
    if (operation === 'models.inspect') return { modelId: input.modelId, parameters: input.modelId === 'lda' ? { 'model.alpha': { type: 'float' }, max_iter: { type: 'int' } } : { n_neighbors: { type: 'int' } } } as T;
    if (operation === 'dataset.import') return { datasetRef: 'params', sha256: 'test', managedPath: input.filePath, fileName: 'data.csv' } as T;
    if (operation === 'dataset.profile') return { columns: ['正文', '时间', '渠道'] } as T;
    if (operation === 'compute.preview') { previews.push(input.plan); return { execution: { embedding: { mode: 'local' } }, readiness: { ready: false } } as T; }
    throw new Error(operation);
  } };
  const server = createManualServer(home, worker);
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve));
  const base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
  try {
    const lda = await (await fetch(base + '/api/models/lda')).json() as any;
    const bert = await (await fetch(base + '/api/models/bertopic')).json() as any;
    assert.ok(lda.parameters['model.alpha']); assert.equal(lda.parameters.epochs, undefined);
    assert.ok(bert.parameters.n_neighbors); assert.equal(bert.parameters.max_iter, undefined);
    assert.equal((await fetch(base + '/api/models/unknown')).status, 404);
    const file = await (await fetch(base + '/api/upload?dataset_name=parameters&filename=data.csv', { method: 'POST', body: '正文,时间,渠道\n文本,2026,平台' })).json() as any;
    await fetch(base + '/api/train/start', { method: 'POST', body: JSON.stringify({ file_id: file.id, dataset_name: 'parameters', model_type: 'lda', num_topics: 10, text_column: '正文', time_column: '时间', model_params: { lda: { max_iter: 7, 'model.alpha': .2 } } }) });
    for (let i = 0; i < 20 && !previews.length; i++) await new Promise(resolve => setTimeout(resolve, 10));
    assert.equal(previews[0].textColumn, '正文'); assert.equal(previews[0].timeColumn, '时间');
    assert.equal(previews[0].params['model.alpha'], .2); assert.equal(previews[0].params.max_iter, 7);
    assert.equal(previews[0].params.epochs, undefined);
  } finally {
    await new Promise<void>(resolve => server.close(() => resolve()));
    rmSync(home, { recursive: true, force: true });
  }
});

test('云端执行必须绑定界面确认的接收服务和模型，配置变化不得提交', async () => {
  const home = mkdtempSync(path.join(os.tmpdir(), 'theta-manual-cloud-'));
  let submitted = 0;
  const worker: CapabilityWorker = { async call<T>(operation: string, input: any): Promise<T> {
    if (operation === 'dataset.import') return { datasetRef: 'cloud-test', sha256: 'test', fileName: 'data.csv', managedPath: input.filePath, sizeBytes: 10 } as T;
    if (operation === 'dataset.profile') return { columns: ['text'] } as T;
    if (operation === 'compute.preview') return { execution: { embedding: { mode: 'cloud', provider: 'zhipu', model: 'embedding-3', endpoint: 'https://open.bigmodel.cn/api/paas/v4/embeddings' } }, readiness: { ready: true } } as T;
    if (operation === 'compute.submit') { submitted++; return { id: input.jobId, status: 'completed' } as T; }
    if (operation === 'compute.status') return { id: input.jobId, status: 'completed' } as T;
    throw new Error(operation);
  } };
  const server = createManualServer(home, worker);
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve));
  const base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
  try {
    const file = await (await fetch(base + '/api/upload?dataset_name=cloud-test&filename=data.csv', { method: 'POST', body: 'text\ntest' })).json() as any;
    const shown = { provider: 'zhipu', model: 'embedding-3', endpoint: 'https://open.bigmodel.cn/api/paas/v4' };
    for (const selection of [undefined, { ...shown, model: 'changed' }, { ...shown, endpoint: 'https://other.example' }, shown]) {
      const before = submitted;
      const job = await (await fetch(base + '/api/train/start', { method: 'POST', body: JSON.stringify({ file_id: file.id, dataset_name: 'cloud-test', model_type: 'theta', embedding_provider: 'cloud', cloud_confirmed: true, external_request_limit: 200, cloud_selection: selection }) })).json() as any;
      let status: any;
      for (let i = 0; i < 40; i++) {
        status = await (await fetch(base + `/api/train/${job.id}/status`)).json();
        if (!['pending', 'running'].includes(status.status)) break;
        await new Promise(resolve => setTimeout(resolve, 10));
      }
      if (selection !== shown) {
        assert.equal(status.status, 'failed');
        assert.match(status.error_message, /界面确认内容不一致/u);
        assert.equal(submitted, before);
      } else assert.equal(submitted, before + 1);
    }
  } finally {
    await new Promise<void>(resolve => server.close(() => resolve()));
    rmSync(home, { recursive: true, force: true });
  }
});

test('手动多模型按持久队列串行执行，服务重启后继续，取消不再启动后续模型', async () => {
  const home = mkdtempSync(path.join(os.tmpdir(), 'theta-manual-queue-'));
  const states = new Map<string, any>();
  const submitted: string[] = [];
  const worker: CapabilityWorker = { async call<T>(operation: string, input: any): Promise<T> {
    if (operation === 'dataset.import') return { datasetRef: 'queue', sha256: 'test', fileName: 'data.csv', managedPath: input.filePath, sizeBytes: 20 } as T;
    if (operation === 'dataset.profile') return { columns: ['text'] } as T;
    if (operation === 'compute.preview') return { execution: { embedding: { mode: 'local' } }, readiness: { ready: true } } as T;
    if (operation === 'compute.submit') {
      assert.ok([...states.values()].every(state => state.status !== 'running'), '不得并行启动多个模型');
      submitted.push(input.plan.modelId);
      states.set(input.jobId, { id: input.jobId, status: 'running', phase: 'training', percent: 35 });
      return states.get(input.jobId);
    }
    if (operation === 'compute.status') return states.get(input.jobId);
    if (operation === 'compute.cancel') { states.get(input.jobId).status = 'cancelled'; return states.get(input.jobId); }
    throw new Error(operation);
  } };
  let server = createManualServer(home, worker);
  const listen = async () => { await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve)); return `http://127.0.0.1:${(server.address() as AddressInfo).port}`; };
  let base = await listen();
  try {
    const file = await (await fetch(base + '/api/upload?dataset_name=test&filename=data.csv', { method: 'POST', body: 'text\ntest' })).json() as any;
    const job = await (await fetch(base + '/api/train/start', { method: 'POST', body: JSON.stringify({ file_id: file.id, dataset_name: 'test', model_type: 'lda,nvdm,gsm', num_topics: 10 }) })).json() as any;
    for (let i = 0; i < 20 && !submitted.length; i++) await new Promise(resolve => setTimeout(resolve, 20));
    assert.deepEqual(submitted, ['lda']);
    const first = await (await fetch(base + `/api/train/${job.id}/status`)).json() as any;
    assert.deepEqual(first.queued_models, ['nvdm', 'gsm']);
    assert.equal(first.queue, undefined, '队列执行策略不返回浏览器');
    await new Promise<void>(resolve => server.close(() => resolve()));
    states.values().next().value.status = 'completed';
    server = createManualServer(home, worker); base = await listen();
    for (let i = 0; i < 45 && submitted.length < 2; i++) await new Promise(resolve => setTimeout(resolve, 100));
    assert.deepEqual(submitted, ['lda', 'nvdm']);
    const cancel = await (await fetch(base + `/api/train/${job.id}/cancel`, { method: 'POST' })).json() as any;
    assert.equal(cancel.status, 'cancelled');
    assert.deepEqual(submitted, ['lda', 'nvdm']);
  } finally {
    await new Promise<void>(resolve => server.close(() => resolve()));
    rmSync(home, { recursive: true, force: true });
  }
});

test('本机免登录、上传回执、列预览及训练失败边界；不创建对话对象', async () => {
  const home = mkdtempSync(path.join(os.tmpdir(), 'theta-manual-test-'));
  const operations: string[] = [];
  const plans: any[] = [];
  const worker: CapabilityWorker = { async call<T>(operation: string, input: any): Promise<T> {
    operations.push(operation);
    if (operation === 'dataset.import') return { datasetRef: 'test', sha256: 'test', fileName: 'data.csv', managedPath: input.filePath, sizeBytes: 20 } as T;
    if (operation === 'dataset.profile') return { columns: ['text'] } as T;
    if (operation === 'dataset.preview') return { columns: ['text'], rows: [['测试文本']] } as T;
    if (operation === 'compute.preview') { plans.push(input.plan); return { execution: { embedding: { mode: 'local' } }, readiness: { ready: false } } as T; }
    throw new Error(`Unexpected operation: ${operation}`);
  } };
  const server = createManualServer(home, worker);
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve));
  const base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
  try {
    assert.equal((await (await fetch(base + '/api/auth/me')).json() as any).username, '本地开发');
    assert.equal((await fetch(base + '/api/files', { headers: { Origin: 'https://evil.example' } })).status, 403);
    const rejectedHost = await new Promise<number | undefined>((resolve, reject) => {
      const req = request(base + '/api/files', { headers: { Host: 'evil.example:4321' } }, res => { res.resume(); resolve(res.statusCode); });
      req.on('error', reject); req.end();
    });
    assert.equal(rejectedHost, 403);
    assert.equal((await fetch(base + '/api/upload?dataset_name=test&filename=..%2Fsecret.csv', { method: 'POST', body: 'text\ntest' })).status, 400);
    assert.equal((await fetch(base + '/api/upload?dataset_name=test&filename=data.csv', { method: 'POST', body: '' })).status, 400);
    const response = await fetch(base + '/api/upload?dataset_name=test&filename=data.csv', { method: 'POST', body: 'text\ntest\nhello' });
    assert.equal(response.status, 201); const file = await response.json() as any;
    assert.ok(file.id > 0);
    const preview = await (await fetch(base + `/api/datasets/test/preview?file_id=${file.id}`)).json();
    assert.deepEqual(preview, { columns: ['text'], rows: [['测试文本']] });
    const start = async (data: unknown) => {
      const response = await fetch(base + '/api/train/start', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) });
      // Preflight errors belong to the durable job, not a long HTTP request.
      if (response.status === 201) {
        const job = await response.clone().json() as any;
        const status = await (await fetch(base + `/api/train/${job.id}/status`)).json() as any;
        assert.equal(status.status, 'failed');
        assert.match(status.error_message, /未就绪/u);
      }
      return response;
    };
    assert.equal((await start({ file_id: 999, dataset_name: 'test' })).status, 404);
    assert.equal((await start({ file_id: file.id, dataset_name: 'other' })).status, 400);
    assert.equal((await start({ file_id: file.id, dataset_name: 'test', model_type: 'lda' })).status, 201);
    assert.equal(plans.at(-1).params['text.stopwords'], undefined);
    await start({ file_id: file.id, dataset_name: 'test', model_type: 'hdp', num_topics: 10 });
    assert.equal(plans.at(-1).params.num_topics, undefined);
    assert.equal(plans.at(-1).params.max_topics, 10);
    const beforeCloud = plans.length;
    assert.equal((await start({ file_id: file.id, dataset_name: 'test', model_type: 'theta', embedding_provider: 'cloud' })).status, 400);
    assert.equal(plans.length, beforeCloud, 'no cloud preview before explicit consent');
    assert.equal((await start({ file_id: file.id, dataset_name: 'test', model_type: 'theta', embedding_provider: 'cloud', cloud_confirmed: true, external_request_limit: 0 })).status, 400);
    assert.equal(plans.length, beforeCloud);
    await start({ file_id: file.id, dataset_name: 'test', model_type: 'theta', embedding_provider: 'cloud', cloud_confirmed: true, external_request_limit: 200 });
    assert.equal(plans.at(-1).params.embedding_provider, 'cloud');
    assert.equal(plans.at(-1).externalRequestLimit, 200);
    await start({ file_id: file.id, dataset_name: 'test', model_type: 'theta' });
    assert.equal(plans.at(-1).params.embedding_provider, 'local');
    assert.equal(plans.at(-1).externalRequestLimit, undefined);
    const exportResponse = await fetch(base + '/api/stopwords/default');
    assert.match(exportResponse.headers.get('content-disposition')!, /attachment/u);
    assert.match(await exportResponse.text(), /# zh\.txt/u);
    assert.equal((await fetch(base + '/api/stopwords?filename=bad.txt', { method: 'POST', body: '' })).status, 400);
    assert.equal((await fetch(base + '/api/stopwords?filename=bad.csv', { method: 'POST', body: 'word' })).status, 400);
    const uploaded = await fetch(base + '/api/stopwords?filename=custom.txt', { method: 'POST', body: '分析\nRESEARCH\nresearch' });
    assert.equal(uploaded.status, 201);
    const stopwords = await uploaded.json() as any;
    assert.equal(stopwords.count, 2);
    assert.equal((await start({ file_id: file.id, dataset_name: 'test', model_type: 'lda', plot_language: 'en', stopwords_id: stopwords.id })).status, 201); // admitted, but preflight failed
    assert.equal(plans.at(-1).params.language, 'en');
    assert.equal(plans.at(-1).params['text.stopwords'], '分析\nresearch');
    assert.equal((await start({ file_id: file.id, dataset_name: 'test', model_type: 'lda', stopwords_id: 999999 })).status, 404);
    assert.equal((await start({ file_id: file.id, dataset_name: 'test', model_type: 'lda', plot_language: 'unknown' })).status, 400);
    await start({ file_id: file.id, dataset_name: 'test', model_type: 'lda', plot_language: 'zh' });
    assert.equal(plans.at(-1).params['text.stopwords'], undefined);
    assert.ok((await (await fetch(base + '/api/train/jobs')).json() as any[]).every(job => job.status === 'failed'));
    assert.equal(operations.includes('compute.submit'), false);
    assert.equal((await fetch(base + '/api/v3/runs')).status, 404);
  } finally {
    await new Promise<void>(resolve => server.close(() => resolve()));
    rmSync(home, { recursive: true, force: true });
  }
});

test('训练状态在轮次变化时刷新细节，并在完成与服务重启后保留日志', async t => {
  const home = mkdtempSync(path.join(os.tmpdir(), 'theta-manual-telemetry-'));
  const manual = new DatabaseSync(path.join(home, 'manual.sqlite'));
  manual.exec('CREATE TABLE records (id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, value TEXT NOT NULL)');
  manual.prepare('INSERT INTO records(kind,value) VALUES (?,?)').run('job', JSON.stringify({ dataset_name: 'fixture', models: ['prodlda'], status: 'running', workers: [{ id: 'progress-job', model: 'prodlda' }] }));
  manual.close();
  const compute = new DatabaseSync(path.join(home, 'compute.sqlite'));
  compute.exec('CREATE TABLE jobs (id TEXT PRIMARY KEY, state TEXT, value TEXT)');
  let epoch = 1;
  let calls = 0;
  let state = 'running';
  const snapshot = () => ({ id: 'progress-job', status: state, phase: state === 'running' ? 'training' : 'completed', percent: state === 'running' ? 60 : 100 });
  const persist = () => compute.prepare('INSERT OR REPLACE INTO jobs VALUES (?,?,?)').run('progress-job', state, JSON.stringify(snapshot()));
  persist();
  let now = Date.now();
  t.mock.method(Date, 'now', () => now);
  const worker: CapabilityWorker = { async call<T>(operation: string): Promise<T> {
    assert.equal(operation, 'compute.status'); calls++;
    return { ...snapshot(), telemetry: { events: [{ id: epoch, at: 1, kind: 'epoch', current: epoch, total: 2, metrics: { loss: 1 / epoch } }] } } as T;
  } };
  let server = createManualServer(home, worker);
  const listen = async () => { await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve)); return `http://127.0.0.1:${(server.address() as AddressInfo).port}`; };
  let base = await listen();
  const read = async () => (await (await fetch(base + '/api/train/1/status')).json()) as any;
  try {
    assert.equal((await read()).states[0].telemetry.events[0].current, 1);
    await fetch(base + '/api/train/jobs');
    assert.equal(calls, 1, '列表读取不重复采集详细日志');
    epoch = 2; now += 2500;
    assert.equal((await read()).states[0].telemetry.events[0].current, 2);
    assert.equal(calls, 2, '阶段百分比不变时仍刷新 epoch');
    state = 'completed'; persist(); now += 2500;
    assert.equal((await read()).status, 'succeeded');
    await new Promise<void>(resolve => server.close(() => resolve()));
    server = createManualServer(home, worker); base = await listen();
    assert.equal((await read()).states[0].telemetry.events[0].current, 2, '已完成任务不能用缺少日志的 SQLite 快照替代详细状态');
    const catalog = await (await fetch(base + '/api/results/fixture/catalog')).json() as any;
    assert.equal(catalog.results[0].execution.telemetry.events[0].current, 2, '结果页可恢复当前模型的执行日志');
  } finally {
    await new Promise<void>(resolve => server.close(() => resolve()));
    compute.close(); rmSync(home, { recursive: true, force: true });
  }
});

test('truncated uploads are rejected and document collections cannot include another project', async t => {
  const home = mkdtempSync(path.join(os.tmpdir(), 'theta-upload-'));
  const calls: string[] = [];
  const worker: CapabilityWorker = { async call<T>(operation: string, input: any): Promise<T> {
    calls.push(operation);
    if (operation === 'dataset.import') return { fileName: 'source.txt', sizeBytes: 4, datasetRef: 'd', sha256: 'hash', managedPath: input.filePath } as T;
    if (operation === 'dataset.combine') return { fileName: '文档合集.jsonl', sizeBytes: 20, datasetRef: 'combined', inputKind: 'text' } as T;
    throw new Error(operation);
  } };
  const server = createManualServer(home, worker);
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(async () => { server.closeAllConnections(); await new Promise<void>(resolve => server.close(() => resolve())); rmSync(home, {recursive:true, force:true}); });
  const base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
  const upload = (dataset: string, size: string) => fetch(`${base}/api/upload?filename=source.txt&dataset_name=${dataset}`, { method: 'POST', headers: {'x-theta-file-size': size}, body:'text' });
  assert.equal((await upload('one', '100')).status, 400);
  assert.equal(calls.length, 0);
  const file = await (await upload('one', '4')).json() as any;
  const combine = (dataset: string) => fetch(`${base}/api/datasets/${dataset}/combine`, {method:'POST',body:JSON.stringify({fileIds:[file.id]})});
  assert.equal((await combine('two')).status, 400);
  assert.equal(calls.filter(call => call === 'dataset.combine').length, 0);
  const merged = await combine('one'); assert.equal(merged.status, 201);
  assert.equal((await merged.json() as any).inputKind, 'text');
  const files = await (await fetch(base + '/api/files')).json() as any[];
  assert.equal(files.length, 2, 'Original upload stays available after combining');
});
