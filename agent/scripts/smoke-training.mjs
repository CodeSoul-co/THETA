// Explicit integration smoke: a tiny repository fixture, real local CPU worker, no LLM.
import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { LocalProductTools } from '../dist/src/tools/local-tools.js';
import { ProductSessionStore } from '../dist/src/memory/session-store.js';

const home = mkdtempSync(path.join(os.tmpdir(), 'theta-agent-smoke-'));
const root = fileURLToPath(new URL('../../', import.meta.url));
const options = { runtimeDb: path.join(home, 'research.sqlite'), uploadDir: path.join(home, 'uploads') };
const store = new ProductSessionStore(home);
const session = store.create();
const tools = new LocalProductTools(options);
const context = { session, userMessage: 'Integration fixture: compare technology and food topics', save: () => store.save(session) };
const call = (name, input = {}) => tools.execute(name, input, context);
try {
  const dataset = await tools.attach(path.join(root, 'trainning/testdata/lda_smoke.csv'), session);
  await call('run_create', { datasetRef: dataset.datasetRef, goal: context.userMessage });
  await call('research_continue');
  await call('plan_propose', { modelId: 'lda', textColumn: 'text', rationale: 'Small CPU regression fixture', params: { num_topics: 2, max_iter: 2, vocab_size: 100, skip_eval: true, skip_viz: true, language: 'english' }, timeoutSeconds: 180 });
  const readiness = await call('training_prepare');
  assert.notEqual(readiness.ready, false, JSON.stringify(readiness));
  const pending = await call('training_advance');
  assert.equal(pending.needsUser, true);
  await tools.approve(session, '确认');
  const submitted = await call('training_advance');
  const jobId = submitted.job.id;
  assert.equal((await call('training_advance')).job.id, jobId);
  // Simulate another host process restoring the session and local gateway.
  let result;
  for (let count = 0; count < 120; count++) {
    await new Promise((resolve) => setTimeout(resolve, 2000));
    result = await new LocalProductTools(options).execute('run_status', {}, context);
    if (count % 5 === 0) console.log(JSON.stringify({ home, status: result.job.status, phase: result.job.phase, percent: result.job.percent }));
    if (!['queued', 'running'].includes(result.job.status)) break;
  }
  assert.equal(result.job.status, 'completed', JSON.stringify(result));
  assert.equal((await call('results_read', { view: 'summary' })).needsUser, true);
  await tools.approve(session, '确认'); // Explicit fixture report consent, distinct from training.
  const evidence = await call('results_read', { view: 'summary' });
  assert.ok(evidence.evidence.length > 0, 'Expected actual result evidence');
  const artifacts = await call('results_read', { view: 'artifacts' });
  assert.ok(artifacts.files.length > 0);
  const report = { home, jobId, status: result.job.status, evidenceFiles: evidence.evidence.map((item) => item.relativePath), artifactCount: artifacts.files.length };
  writeFileSync(path.join(home, 'smoke-report.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report));
} finally { store.close(); }
