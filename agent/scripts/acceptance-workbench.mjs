// Explicit, bounded CPU acceptance on disposable synthetic data; no external API.
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { createAgentServer } from '../dist/web/server.js';
import { createManualServer } from '../dist/web/manual-server.js';
import { PythonCapabilityWorker } from '../dist/src/adapters/python-worker.js';

if (!process.argv.includes('--confirm-local-training')) throw new Error('Use --confirm-local-training: synthetic LDA, K=2, 2 iterations, 4-minute acceptance deadline; no cloud services.');
const home = mkdtempSync(path.join(tmpdir(), 'theta-workbench-acceptance-'));
const worker = new PythonCapabilityWorker();
const manualHome = path.join(home, 'manual'), agentHome = path.join(home, 'agent');
const listen = async server => { await new Promise(resolve => server.listen(0, '127.0.0.1', resolve)); return `http://127.0.0.1:${server.address().port}`; };
let servers = [], activeJob, request;
async function host(token) {
  servers = [createManualServer(manualHome, worker, { localToken: token }), createAgentServer(agentHome, () => undefined, { mode: 'local', manualHome, localToken: token }, { worker })];
  const [manual, agent] = await Promise.all(servers.map(listen));
  return async (kind, route, input, method = input ? 'POST' : 'GET') => {
    const response = await fetch((kind === 'manual' ? manual : agent) + route, { method,
      headers: { 'content-type': 'application/json', ...(token ? { 'x-theta-desktop-token': token } : {}) },
      body: typeof input === 'string' ? input : input ? JSON.stringify(input) : undefined });
    const body = await response.json();
    assert.ok(response.ok, `${method} ${route}: ${JSON.stringify(body)}`);
    return kind === 'agent' ? body.data : body;
  };
}
async function close() { await Promise.all(servers.map(server => new Promise(resolve => { server.close(resolve); server.closeAllConnections(); }))); }
try {
  const web = request = await host();
  const project = await web('manual', '/api/projects', { name: 'Cross-end synthetic research' });
  const upload = (filename, body) => web('manual', `/api/upload?dataset_name=${encodeURIComponent(project.dataset_name)}&filename=${encodeURIComponent(filename)}`, body);
  const rows = Array.from({ length: 36 }, (_, i) => i % 18 < 9 ? `computer code robot machine software technology ${i}` : `garden apple fruit orchard harvest trees ${i}`);
  const file = await upload('research.csv', 'text\n' + rows.join('\n'));
  const first = await upload('folder-a.txt', rows.slice(0, 18).join('\n'));
  const second = await upload('folder-b.md', rows.slice(18).join('\n'));
  const combined = await web('manual', `/api/datasets/${project.dataset_name}/combine`, { fileIds: [first.id, second.id] });
  const preview = await web('manual', `/api/datasets/${project.dataset_name}/preview?file_id=${combined.file_id}`);
  assert.equal(preview.totalRecords, 36, JSON.stringify(preview));
  const removed = await web('manual', `/api/files/${first.id}`, undefined, 'DELETE');
  assert.ok(removed.removed_file_ids.includes(combined.file_id));
  const job = await web('manual', '/api/train/start', { dataset_name: project.dataset_name, file_id: file.id,
    model_type: 'lda,dtm', num_topics: 2, text_column: 'text', plot_language: 'en',
    model_params: { lda: { max_iter: 2, vocab_size: 100, language: 'english' } },
    data_split: { enabled: true, mode: 'ratio', method: 'sequential', ratios: [.5, .25, .25], seed: 42 }, timeout_seconds: 180 });
  activeJob = job.id;
  let status;
  const deadline = Date.now() + 240000;
  do {
    await new Promise(resolve => setTimeout(resolve, 1000));
    status = await web('manual', `/api/train/${job.id}/status`);
    if (status.status !== 'running' || Math.round((deadline - Date.now()) / 1000) % 10 === 0) console.log(JSON.stringify({ status: status.status, models: status.states?.map(item => ({ status: item.status, phase: item.phase, activity: item.telemetry?.activity })) }));
  } while (['pending', 'running'].includes(status.status) && Date.now() < deadline);
  assert.ok(!['pending', 'running'].includes(status.status), 'Training exceeded acceptance deadline');
  activeJob = undefined;
  const catalog = await web('manual', `/api/results/${project.dataset_name}/catalog`);
  const success = catalog.results.find(item => item.modelId === 'lda' && item.status === 'completed');
  assert.ok(success, JSON.stringify({ status, catalog }));
  assert.ok(catalog.results.some(item => item.modelId === 'dtm' && item.status === 'failed' && /时间/.test(item.error)), JSON.stringify(catalog));
  const splitFile = success.artifacts.files.find(item => item.name.endsWith('split_results.json'));
  assert.ok(splitFile, 'Persist all/train/validation/test evaluation');
  const split = await web('manual', splitFile.url.replace('/api/backend', ''));
  assert.deepEqual(Object.keys(split.groups).sort(), ['all', 'test', 'train', 'validation']);
  assert.ok(split.groups.train.topicProportions.some((value, i) => Math.abs(value - split.groups.test.topicProportions[i]) > .01), 'Independent data must have independent topic weights');
  assert.ok(success.artifacts.files.some(item => /splits\/train\/.*\.(svg|png)$/.test(item.name)), 'Train split figures must be available');
  const imported = await web('agent', '/api/v3/projects/import-manual-results', { projectId: project.id, selectedJobId: success.jobId });
  const result = await web('agent', `/api/v3/runs/${imported.runId}/results`);
  assert.ok(result.results.some(item => item.status === 'completed'));
  const skill = { files: [{ path: 'SKILL.md', content: Buffer.from('---\nname: cross-end-demo\ndescription: Synthetic testing\n---\nUse actual results.').toString('base64') }] };
  await web('agent', '/api/v3/skills/import', skill);
  await close();
  // Same saved fixture, restarted behind the desktop's private-token transport.
  const desktop = await host('d'.repeat(64));
  assert.equal((await desktop('manual', '/api/projects')).find(item => item.id === project.id).name, project.name);
  assert.ok((await desktop('agent', `/api/v3/runs/${imported.runId}/results`)).results.some(item => item.status === 'completed'));
  assert.ok((await desktop('agent', '/api/v3/skills')).skills.some(item => item.id === 'cross-end-demo'));
  await desktop('agent', '/api/v3/skills/cross-end-demo', { enabled: false }, 'PATCH');
  assert.equal((await desktop('agent', '/api/v3/skills')).skills.find(item => item.id === 'cross-end-demo').enabled, false);
  console.log(JSON.stringify({ ok: true, checks: ['folder aggregation/deletion', 'LDA training', 'isolated DTM failure', 'four split metrics/figures', 'distinct topic weights', 'manual-to-Agent result copy', 'persisted Web/desktop catalog and skills'], artifactCount: success.artifacts.fileCount }));
} finally {
  let disposable = true;
  try {
    if (activeJob) {
      await request('manual', `/api/train/${activeJob}/cancel`, {});
      const deadline = Date.now() + 10000;
      while (Date.now() < deadline && ['pending', 'running', 'cancelling'].includes((await request('manual', `/api/train/${activeJob}/status`)).status)) await new Promise(resolve => setTimeout(resolve, 250));
      assert.ok(!['pending', 'running', 'cancelling'].includes((await request('manual', `/api/train/${activeJob}/status`)).status), 'Cancellation pending');
    }
  } catch (error) { disposable = false; console.error(`Retained fixture at ${home}: ${error.message}`); }
  await close();
  if (disposable) rmSync(home, { recursive: true, force: true });
}
