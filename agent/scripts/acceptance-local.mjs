// Explicit opt-in only: two bounded real CPU training jobs on synthetic data.
import assert from 'node:assert/strict';
import { writeSupportFixture } from './support-fixture.mjs';
import { mkdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { LocalProductTools } from '../dist/src/tools/local-tools.js';
import { ProductSessionStore } from '../dist/src/memory/session-store.js';
if (!process.argv.includes('--confirm-local-training')) throw new Error('Explicit --confirm-local-training required (LDA + STM, K=3, max_iter=5, 180s/job, no cloud/downloads).');
const home = path.resolve(process.argv.slice(2).find(arg => !arg.startsWith('--')) ?? '.theta_agent/acceptance/local');
mkdirSync(home, { recursive: true });
const source = path.join(home, 'outside-data', 'support feedback.csv');
writeSupportFixture(source);
const store = new ProductSessionStore(home), session = store.create();
const tools = new LocalProductTools({ runtimeDb: path.join(home, 'research.sqlite'), uploadDir: path.join(home, 'uploads') });
const context = { session, userMessage: '找出客服改善优先级，比较线上和电话渠道；合成验收数据', save: () => store.save(session) };
const call = (name, args = {}) => tools.execute(name, args, context);
const results = [];
try {
  await tools.attach(source, session);
  await call('dataset_understand');
  tools.recordUnderstanding(session, '这是36条合成客服记录，涉及配送、退款与应用登录，渠道为 web/phone。目标是识别待验证的改善方向。合成数据不代表真实客户比例或因果关系。');
  for (const modelId of ['lda', 'stm']) {
    await call('run_create', { goal: context.userMessage });
    await call('plan_propose', { modelId, textColumn: 'text', ...(modelId === 'stm' ? { covariates: ['channel'] } : {}), params: { num_topics: 3, max_iter: 5, vocab_size: 100, skip_eval: true, skip_viz: true, language: 'english' }, rationale: 'Authorized bounded acceptance on synthetic support records', timeoutSeconds: 180 });
    const pending = await call('training_advance');
    assert.equal(pending.needsUser, true, JSON.stringify(pending));
    const submitted = await tools.approve(session, '确认'); context.save();
    console.log(JSON.stringify({ modelId, jobId: submitted.job.id, status: submitted.job.status }));
    let outcome;
    const deadline = Date.now() + 210000;
    do {
      await new Promise(resolve => setTimeout(resolve, 2000));
      outcome = await call('run_status');
    } while (['queued', 'running'].includes(outcome.job.status) && Date.now() < deadline);
    results.push({ modelId, runId: session.runId, ...outcome.job });
    writeFileSync(path.join(home, 'training.json'), JSON.stringify({ home, sessionId: session.id, results }, null, 2));
    console.log(JSON.stringify({ modelId, status: outcome.job.status, phase: outcome.job.phase, error: outcome.job.error, diagnostics: outcome.job.diagnostics }));
  }
} finally { store.save(session); store.close(); }
assert.ok(results.every(item => item.status === 'completed'), 'See training.json for actual failures; no automatic re-training.');
