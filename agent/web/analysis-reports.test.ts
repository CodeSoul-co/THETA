import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, readFileSync, writeFileSync, existsSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { AnalysisReports } from './analysis-reports.js';
import type { CapabilityWorker } from '../src/adapters/python-worker.js';

test('report is click-only, deduplicates running work, preserves Markdown on PDF failure and retries only export', async t => {
  const home = mkdtempSync(path.join(os.tmpdir(), 'theta-analysis-report-'));
  let calls = 0, pdfCalls = 0;
  const worker: CapabilityWorker = { async call<T>(operation: string, input: any) {
    if (operation === 'analysis_report.evidence') {
      assert.equal(input.jobId, 'owned-job');
      return { dataset: { datasetRef: 'data' }, plan: { modelId: 'lda' }, resultHash: 'verified', description: '## 1. 数据描述性统计\n\n|记录数|10|', summary: { count: 10 } } as T;
    }
    assert.equal(operation, 'analysis_report.pdf');
    if (++pdfCalls === 1) throw new Error('PDF rendering failed');
    writeFileSync(path.join(input.directory, 'analysis.pdf'), '%PDF fixture'); return {} as T;
  } };
  const reports = new AnalysisReports(home, worker);
  t.after(async () => { await reports.close(); rmSync(home, { recursive: true, force: true }); });
  const provider = { id: 'fixture', async infer(request: any) { calls++; assert.equal(request.agentId, 'agent.theta.academic-report'); assert.match(request.input.instructions, /每个分析章节以约 1000–1800/); assert.match(JSON.stringify(request.input.messages), /研究目标测试/); assert.match(JSON.stringify(request.input), /不要以简短分点/); assert.match(JSON.stringify(request.input), /研究问题—数据特征/);  return { id: 'answer', output: { kind: 'tool_calls', toolCalls: [{ id: 'answer', name: 'respond', arguments: { message: '这是根据当前已验证的数据和模型结果生成的完整章节。'.repeat(25) } }] } }; } };
  assert.equal(reports.state('scope').status, 'idle'); assert.equal(calls, 0); assert.equal(existsSync(path.join(home, 'analysis-reports')), false);
  assert.throws(() => reports.start('scope', 'owned-job', undefined), /API/);
  reports.start('scope', 'owned-job', provider, '研究目标测试'); reports.start('scope', 'owned-job', provider);
  for (let i = 0; i < 100 && reports.state('scope').status === 'running'; i++) await new Promise(resolve => setTimeout(resolve, 5));
  assert.equal(calls, 4); assert.equal(reports.state('scope').status, 'failed'); assert.equal(reports.state('scope').markdown, true);
  assert.match(readFileSync(reports.file('scope', 'md'), 'utf8'), /## 5\. 结论分析/);
  assert.throws(() => reports.file('other-scope', 'md'), /尚未生成/);
  reports.start('scope', 'owned-job', provider);
  for (let i = 0; i < 100 && reports.state('scope').status === 'running'; i++) await new Promise(resolve => setTimeout(resolve, 5));
  assert.equal(reports.state('scope').status, 'complete'); assert.equal(calls, 4); assert.equal(pdfCalls, 2);
  reports.start('scope', 'owned-job', provider); assert.equal(calls, 4);
});
