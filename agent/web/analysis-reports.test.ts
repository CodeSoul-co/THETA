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

test('report headings and citations are normalized and original exhibits appear only once', async () => {
  const { normalizeChapter, citationProblem, insertExhibits } = await import('./report-format.js');
  const catalog = [{ id: '图1', kind: 'figure' as const, title: '主题权重', scope: '验证集', block: '![图1](assets/figure-1.png)' }, { id: '表1', kind: 'table' as const, title: '指标', scope: '验证集', block: '| 指标 | 值 |\n|---|---|\n|NPMI|0.2|' }];
  const prose = '本段围绕研究问题展开论证，结合实际权重差异解释主题分布的含义，并明确样本范围和判断依据。'.repeat(2);
  const text = normalizeChapter(`\`\`\`markdown\n# 4. 建模描述\n\n## 4.1 实际训练\n\n${prose}依据【图1】分析。\n\n[[图1]]\n\n${prose}根据【表1】比较。\n\n[[表1]]\n\n${prose}后续仍参照【图1】。\n\`\`\``, '4. 建模描述');
  assert.doesNotMatch(text, /^# 4\./m); assert.match(text, /^### 4\.1/m);
  assert.equal(citationProblem(text, catalog, true), undefined);
  assert.match(citationProblem('【图99】', catalog, true)!, /不存在/);
  assert.match(citationProblem('只引用【图1】', catalog, true)!, /本章节/);
  assert.match(citationProblem(`${prose}【图1】【表1】\n\n[[图1]]\n\n[[表1]]\n\n${prose}`, catalog, true)!, /不得相邻堆放/);
  assert.match(citationProblem(`${prose}【图1】\n\n[[图1]]`, catalog, false)!, /实质性分析段落/);
  const inserted = new Set<string>();
  const rendered = insertExhibits(text, catalog, inserted);
  assert.equal(rendered.match(/figure-1.png/g)?.length, 1);
  assert.doesNotMatch(rendered, /\[\[(图|表)/);
  assert.doesNotMatch(insertExhibits('继续分析【图1】', catalog, inserted), /!\[/);
  assert.equal(citationProblem('再次讨论【图1】与【表1】', catalog, true, inserted), undefined);

});

test('explicit regeneration keeps the existing download when model generation fails', async t => {
  const home = mkdtempSync(path.join(os.tmpdir(), 'theta-report-regenerate-'));
  const worker: CapabilityWorker = { async call<T>(operation: string, input: any) {
    if (operation === 'analysis_report.evidence') return { dataset: { datasetRef: 'data' }, plan: { modelId: 'lda' }, description: '## 1. 数据描述性统计', catalog: [] } as T;
    writeFileSync(path.join(input.directory, 'analysis.pdf'), 'pdf'); writeFileSync(path.join(input.directory, 'analysis.zip'), 'zip'); return {} as T;
  } };
  const reports = new AnalysisReports(home, worker);
  t.after(async () => { await reports.close(); rmSync(home, { recursive: true, force: true }); });
  let fail = false, calls = 0;
  const provider = { id: 'fixture', async infer() { calls++; if (fail) throw new Error('test provider unavailable'); return { id: 'response', output: { kind: 'tool_calls', toolCalls: [{ id: 'answer', name: 'respond', arguments: { message: '完整的真实证据与研究结论。'.repeat(50) } }] } }; } };
  const wait = async () => { for (let i = 0; i < 200 && reports.state('key').status === 'running'; i++) await new Promise(resolve => setTimeout(resolve, 5)); };
  reports.start('key', 'job', provider); await wait();
  assert.equal(reports.state('key').status, 'complete');
  const old = reports.file('key', 'md');
  fail = true; reports.start('key', 'job', provider, '', true); await wait();
  assert.equal(reports.state('key').status, 'failed'); assert.equal(reports.file('key', 'md'), old); assert.equal(reports.file('key', 'zip').endsWith('analysis.zip'), true);
  fail = false; reports.start('key', 'job', provider, '', true); await wait();
  assert.equal(reports.state('key').status, 'complete'); assert.notEqual(reports.file('key', 'md'), old); assert.equal(existsSync(old), true);
});
