// Explicit opt-in: real local CPU training from non-CSV sources; no LLM.
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { LocalProductTools } from '../dist/src/tools/local-tools.js';
import { ProductSessionStore } from '../dist/src/memory/session-store.js';
import { pythonExecutable } from '../dist/src/adapters/python-worker.js';

const SAMPLES = [
  'parcel delivery arrived late tracking shipment courier delayed package shipping service',
  'refund payment billing charged twice invoice money return customer support response',
  'application login password account crashes software update screen authentication error',
];
const ROWS = 36;
const records = Array.from({ length: ROWS }, (_, index) => ({
  text: SAMPLES[index % 3] + ' ' + ['help request', 'problem unresolved', 'issue resolved', 'followup contact'][index % 4],
  date: '2026-08-' + String(index % 28 + 1).padStart(2, '0'),
  channel: index % 2 ? 'phone' : 'web',
}));

const home = mkdtempSync(path.join(os.tmpdir(), 'theta-conversion-acceptance-'));
const options = { runtimeDb: path.join(home, 'research.sqlite'), uploadDir: path.join(home, 'uploads') };

function writeExcel(target) {
  const rows = [['内容', '发布时间', '渠道'], ...records.map((row) => [row.text, row.date, row.channel])];
  const program = "import json, sys, openpyxl\nrows = json.loads(sys.argv[1])\nbook = openpyxl.Workbook()\nsheet = book.active\nsheet.title = '反馈'\nfor row in rows:\n    sheet.append(row)\nbook.save(sys.argv[2])";
  const result = spawnSync(pythonExecutable(), ['-c', program, JSON.stringify(rows), target], { encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  return target;
}

function writeJsonl(target) {
  writeFileSync(target, records.map((row) => JSON.stringify({ body: row.text, created_at: row.date, channel: row.channel })).join('\n') + '\n');
  return target;
}

async function train(file, { modelId, textColumn, covariates = [], params }) {
  const store = new ProductSessionStore(home);
  const session = store.create();
  const tools = new LocalProductTools(options);
  const context = { session, userMessage: '格式转换验收：非 CSV 数据源', save: () => store.save(session) };
  const call = (name, input = {}) => tools.execute(name, input, context);
  try {
    const dataset = await tools.attach(file, session);
    await call('run_create', { datasetRef: dataset.datasetRef, goal: context.userMessage });
    await call('research_continue');
    await call('plan_propose', { modelId, textColumn, ...(covariates.length ? { covariates } : {}), rationale: 'Acceptance: source format conversion', params, timeoutSeconds: 300 });
    const readiness = await call('training_prepare');
    assert.notEqual(readiness.ready, false, JSON.stringify(readiness));
    assert.equal((await call('training_advance')).needsUser, true);
    await tools.approve(session, '确认');
    const submitted = await call('training_advance');
    const jobId = submitted.job.id;
    let result;
    for (let count = 0; count < 150; count++) {
      await new Promise((resolve) => setTimeout(resolve, 2000));
      result = await new LocalProductTools(options).execute('run_status', {}, context);
      if (!['queued', 'running'].includes(result.job.status)) break;
    }
    assert.equal(result.job.status, 'completed', JSON.stringify(result).slice(0, 600));
    return { jobId, modelId, textColumn };
  } finally { store.close(); }
}

function normalizedCsv(jobId) {
  return path.join(home, 'compute', jobId, 'objects', 'dataset', 'data.csv');
}

const excel = writeExcel(path.join(home, '反馈数据.xlsx'));
const jsonl = writeJsonl(path.join(home, 'feedback.jsonl'));
const results = [];
try {
  const first = await train(excel, { modelId: 'lda', textColumn: '内容', params: { num_topics: 3, max_iter: 3, vocab_size: 100, skip_eval: true, skip_viz: true, language: 'english' } });
  // The engine CSV is written with Python's csv module, whose line terminator is CRLF.
  const csv = readFileSync(normalizedCsv(first.jobId), 'utf8').trim().split(/\r?\n/u);
  assert.equal(csv[0], 'text', 'Excel 的「内容」列必须成为 text 列');
  assert.equal(csv.length - 1, ROWS);
  assert.match(csv[1], /parcel delivery/u);
  results.push({ source: 'xlsx (中文列名)', model: first.modelId, jobId: first.jobId, header: csv[0], rows: csv.length - 1, status: 'completed' });

  const second = await train(jsonl, { modelId: 'stm', textColumn: 'body', covariates: ['channel'], params: { num_topics: 3, max_iter: 5, vocab_size: 100, skip_eval: true, skip_viz: true, language: 'english' } });
  const stmCsv = readFileSync(normalizedCsv(second.jobId), 'utf8').trim().split(/\r?\n/u);
  assert.equal(stmCsv[0], 'text,cov_0', 'JSONL 的 channel 必须成为 cov_0');
  assert.equal(stmCsv.length - 1, ROWS);
  const channels = new Set(stmCsv.slice(1).map((line) => line.slice(line.lastIndexOf(',') + 1)));
  assert.deepEqual([...channels].sort(), ['phone', 'web']);
  results.push({ source: 'jsonl', model: second.modelId, jobId: second.jobId, header: stmCsv[0], rows: stmCsv.length - 1, channels: [...channels].sort(), status: 'completed' });
} finally { /* 保留 home 以便人工核对 */ }
console.log(JSON.stringify({ home, results }, null, 2));
