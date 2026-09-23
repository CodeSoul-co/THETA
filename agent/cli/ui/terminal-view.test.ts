import assert from 'node:assert/strict';
import test from 'node:test';
import { AgentTerminalView, cellWidth, terminalText } from './terminal-view.js';
import type { ProductSession } from '../../src/memory/session-store.js';
import { ExecutionTrace } from '../../src/conversation/execution-events.js';

const session: ProductSession = { id: 'chat-test', title: '中文研究对话', datasetRefs: [], messages: [], updatedAt: '' };

test('terminal cards fit narrow and wide terminals with Chinese and emoji content', () => {
  for (const columns of [32, 50, 80, 140]) {
    const output: string[] = [];
    const view = new AgentTerminalView({ columns, isTTY: false, write: (line) => output.push(line), writeError() {}, question: async () => '', close() {} });
    view.welcome(session, '本地模型', '/a/very/long/path/政策文本');
    view.confirmation('这是需要确认的研究方案。🧑‍🔬'.repeat(15));
    view.composer(session, '本地模型');
    for (const line of output.join('\n').split('\n')) assert.ok(cellWidth(line) <= columns, `Overflow at ${columns}: ${line}`);
    assert.ok(!output.join('').includes('\x1b'));
  }
  assert.equal(cellWidth('主题🧑‍🔬e\u0301'), 7);
});

test('untrusted terminal controls are stripped from assistant responses and checkpoint cards', () => {
  const output: string[] = [];
  const view = new AgentTerminalView({ write: (line) => output.push(line), writeError() {}, question: async () => '', close() {} });
  const malicious = '\x1b[2J正文\x1b]52;c;YQ==\x07\r\x1b[31m内容';
  view.reply(`## 结果\n**主题**\n${malicious}\n\`\`\`python\nprint(1)\n\`\`\``);
  view.confirmation(malicious);
  assert.ok(!output.join('').includes('\x1b'));
  assert.ok(!output.join('').includes('YQ=='));
  assert.match(output.join(''), /正文内容/);
  assert.equal(terminalText('\x1b[31m可见\x1b[0m'), '可见');
});

test('completed and cached calls remain separately visible below the answer', () => {
  const output: string[] = []; const events: import('../../src/conversation/execution-events.js').ExecutionEvent[] = [];
  const view = new AgentTerminalView({ write: text => output.push(text), writeError() {}, question: async () => '', close() {} });
  const trace = new ExecutionTrace(event => { events.push(event); view.execution(event); });
  trace.start('tool', 'datasets_discover')('completed');
  trace.start('tool', 'dataset_understand')('completed');
  trace.start('tool', 'dataset_understand')('cached');
  view.reply('业务理解'); view.executionSummary(events);
  const result = output.join('\n');
  assert.match(result, /#1 datasets_discover/); assert.match(result, /#3 dataset_understand/);
  assert.ok(result.indexOf('本轮调用记录') > result.indexOf('业务理解'));
  assert.match(result, /cached/);
});

test('training status uses observed iterations and distinguishes silent logs from stale heartbeat', async () => {
  const { trainingStatusText } = await import('./terminal-view.js');
  const job: import('../../src/domain/research.js').ComputeJob = {
    id: 'job-test', status: 'running', phase: 'training', percent: 60,
    telemetry: { schemaVersion: 'theta.job-observation.v1', observedAt: 1000, percentKind: 'stage_marker',
      health: 'responding', heartbeatAgeSeconds: 1, elapsedSeconds: 125, phaseElapsedSeconds: 60,
      lastLogAgeSeconds: 2, iteration: { current: 19, total: 50, source: 'worker_log', meaning: 'reported_iteration_not_completed_count' },
      activity: 'fitting', limitation: '' },
  };
  assert.match(trainingStatusText(job), /已报告迭代 19\/50/);
  assert.match(trainingStatusText(job), /已运行 2分5秒/);
  assert.doesNotMatch(trainingStatusText(job), /60%|正常|剩余/);
  job.telemetry!.lastLogAgeSeconds = 40;
  assert.match(trainingStatusText(job), /尚不能判断算法是否推进/);
  job.telemetry!.health = 'unresponsive'; job.telemetry!.heartbeatAgeSeconds = 35;
  assert.match(trainingStatusText(job), /状态不确定/);
  job.status = 'queued';
  assert.match(trainingStatusText(job), /排队等待 worker/);
  job.telemetry = undefined;
  assert.match(trainingStatusText(job), /暂无细粒度进度/);
  for (const columns of [32, 80]) {
    let live = '';
    const view = new AgentTerminalView({ columns, trainingProgress: text => { live = text ?? ''; }, write() {}, writeError() {}, question: async () => '', close() {} });
    view.training(job);
    for (const line of live.split('\n')) assert.ok(cellWidth(line) <= columns);
    view.training(); assert.equal(live, '');
  }
});

test('completion prints the actual result directory persistently without inference or report approval', () => {
  const output: string[] = []; let progress: string | undefined;
  const view = new AgentTerminalView({ columns: 40, write: text => output.push(text), writeError() {},
    trainingProgress: text => { progress = text; }, question: async () => '', close() {} });
  const resultDir = '/actual/very-long-result-directory/中文 数据/result';
  const job: import('../../src/domain/research.js').ComputeJob = { id: 'job-one', status: 'running', phase: 'training', percent: 60, resultDir };
  view.training(job); assert.equal(output.length, 0);
  job.status = 'completed'; job.phase = 'completed';
  view.training(job);
  assert.ok(output[0].includes(`\n${resultDir}\n`), 'Preserve the complete path on a single logical line');
  for (let i = 0; i < 5; i++) view.training(job);
  assert.equal(output.length, 1, 'Repeated observations must not repeat the directory receipt');
  view.training(); assert.equal(progress, undefined);
  assert.ok(output.join('').includes(resultDir), 'Clearing live progress must not remove the directory');
  view.training({ ...job, id: 'job-two', status: 'failed' });
  assert.match(output[1], /可能不完整/);
  view.training({ ...job, id: 'job-three', resultDir: undefined });
  assert.equal(output.length, 2, 'Never invent a missing worker path');
});
