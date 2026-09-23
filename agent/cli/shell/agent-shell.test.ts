import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import type { InteractiveTerminal } from './terminal-io.js';
import type { InferenceProvider } from '../../src/providers/types.js';
import { AgentShell } from './agent-shell.js';
import { ProductSessionStore } from '../../src/memory/session-store.js';

class Terminal implements InteractiveTerminal {
  output: string[] = [];
  closed = false;
  constructor(private readonly inputs: string[]) {}
  write(value: string): void { this.output.push(value); }
  writeError(value: string): void { this.output.push(value); }
  async question(): Promise<string> { assert.ok(this.inputs.length, 'Unexpected prompt'); return this.inputs.shift()!; }
  close(): void { this.closed = true; }
}

test('standalone shell accepts ordinary conversation, persists it, and restores it without menus', async () => {
  const directory = await mkdtemp(path.join(os.tmpdir(), 'theta-shell-'));
  const provider: InferenceProvider = { id: 'fixture', infer: async () => ({ id: 'reply', output: { kind: 'tool_calls', toolCalls: [{ id: 'one', name: 'respond', arguments: { message: '可以先讨论研究问题，无需立即上传数据。' } }] } }) };
  try {
    const terminal = new Terminal(['我还没准备数据，可以先聊聊吗？', '/exit']);
    assert.equal(await new AgentShell({ directory, terminal, inferenceFactory: () => provider }).run(), 0);
    assert.ok(terminal.closed);
    assert.ok(terminal.output.some((line) => line.includes('无需立即上传')));
    const store = new ProductSessionStore(directory);
    const saved = store.list()[0];
    assert.equal(store.get(saved.id).messages.length, 2);
    store.close();
    const resumed = new Terminal(['/exit']);
    await new AgentShell({ directory, terminal: resumed, sessionId: saved.id, inferenceFactory: () => undefined }).run();
    assert.ok(resumed.output.some((line) => line.includes('无需立即上传')));
    const offline = new Terminal(['/help', '/exit']);
    await new AgentShell({ directory, terminal: offline, inferenceFactory: () => undefined }).run();
    assert.ok(offline.output.some((line) => line.includes('/model')));
  } finally { await rm(directory, { recursive: true, force: true }); }
});


test('terminal file-only and instructed uploads, new conversation and resume preserve independent context', async () => {
  const directory = await mkdtemp(path.join(os.tmpdir(), 'theta-shell-files-'));
  const file = path.join(directory, 'outside data.csv');
  await writeFile(file, 'text,channel\nlate delivery,web\nslow refunds,phone\n');
  const received: string[] = [];
  const provider: InferenceProvider = { id: 'fixture', infer: async request => {
    const messages = (request.input as { messages: Array<{ role: string; content: string }> }).messages;
    const last = messages.filter(message => message.role !== 'system').at(-1)!;
    if (last.role === 'user') received.push(last.content);
    const name = last.role === 'user' && last.content.includes('附件回执') && !last.content.includes('只说明附件是否收到') ? 'dataset_understand' : 'respond';
    return { id: 'reply', output: { kind: 'tool_calls', toolCalls: [{ id: 'one', name, arguments: name === 'respond' ? { message: '样本围绕配送延迟和退款等待，改善重点仍需业务验证。' } : {} }] } };
  } };
  try {
    const first = new Terminal(['目标是改善客服，只理解业务。', `"${file}"`, '/trace', '/context', '/exit']);
    await new AgentShell({ directory, terminal: first, inferenceFactory: () => provider }).run();
    const store = new ProductSessionStore(directory);
    const saved = store.get(store.list()[0].id);
    assert.equal(saved.datasetRefs.length, 1); assert.ok(saved.contextId);
    assert.equal(saved.lastUserIntent, '目标是改善客服，只理解业务。');
    assert.ok(first.output.some(line => line.includes('dataset_attach') || line.includes('接收数据')));
    const second = new Terminal([`/attach "${file}" 只说明附件是否收到`, '/sessions', `/resume ${saved.id}`, '继续原来的研究', '/exit']);
    await new AgentShell({ directory, terminal: second, inferenceFactory: () => provider }).run();
    assert.equal(store.list().length, 2);
    const other = store.list().find(item => item.id !== saved.id)!;
    assert.equal(store.get(other.id).lastUserIntent, '只说明附件是否收到');
    assert.equal(store.get(other.id).contextId, undefined);
    assert.equal(store.get(saved.id).contextId, saved.contextId);
    assert.ok(received.some(text => text.includes('用户附带指令：只说明附件是否收到')));
    assert.ok(second.output.some(line => line.includes('研究对话') || line.includes(saved.id)));
    store.close();
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test('training observation continues while conversational inference is pending', { timeout: 15000 }, async () => {
  const { DatabaseSync } = await import('node:sqlite');
  const { ResearchStore } = await import('../../src/memory/research-store.js');
  const directory = await mkdtemp(path.join(os.tmpdir(), 'theta-shell-observer-'));
  const previous = process.env.THETA_COMPUTE_URL; delete process.env.THETA_COMPUTE_URL;
  const store = new ProductSessionStore(directory);
  const session = store.create(); const id = `job-${'b'.repeat(64)}`;
  session.runId = 'run-observer'; session.monitorTraining = true; store.save(session); store.close();
  new ResearchStore(directory).put('run', session.runId, { id: session.runId, goal: '观察进度', jobs: [id], activeJob: id, notes: [], computeBackend: 'local' });
  const db = new DatabaseSync(path.join(directory, 'compute.sqlite'));
  db.exec('CREATE TABLE jobs (id TEXT PRIMARY KEY, request TEXT NOT NULL, state TEXT NOT NULL, value TEXT NOT NULL, cancel INTEGER NOT NULL DEFAULT 0, updated REAL NOT NULL)');
  db.prepare('INSERT INTO jobs VALUES (?,?,?,?,0,?)').run(id, '{}', 'running', JSON.stringify({ id, status: 'running', phase: 'training', percent: 60, startedAt: Date.now() / 1000 }), Date.now() / 1000);
  db.close();
  const terminal = new Terminal(['现在怎么样', '/exit']);
  let observations = 0; let release!: () => void; let inferencePending = false;
  const observed = new Promise<void>(resolve => { release = resolve; });
  Object.assign(terminal, { trainingProgress: (text?: string) => {
    if (text && inferencePending) { observations++; if (observations === 2) release(); }
  } });
  const provider: InferenceProvider = { id: 'fixture', infer: async () => {
    inferencePending = true; await observed; inferencePending = false;
    return { id: 'reply', output: { kind: 'tool_calls', toolCalls: [{ id: 'reply', name: 'respond', arguments: { message: '状态观察未被本轮对话阻塞。' } }] } };
  } };
  const safety = setTimeout(release, 10000);
  try {
    await new AgentShell({ directory, sessionId: session.id, terminal, inferenceFactory: () => provider }).run();
    assert.ok(observations >= 2, 'Status must refresh during a pending inference, not just after it');
  } finally {
    clearTimeout(safety);
    if (previous === undefined) delete process.env.THETA_COMPUTE_URL; else process.env.THETA_COMPUTE_URL = previous;
    await rm(directory, { recursive: true, force: true });
  }
});

test('a single refusal with feedback revokes the card before inference, including offline and failed inference', async () => {
  const { ResearchStore } = await import('../../src/memory/research-store.js');
  const { EffectApprovals } = await import('../../src/domain/effect-approval.js');
  for (const mode of ['online', 'offline', 'failed'] as const) {
    const directory = await mkdtemp(path.join(os.tmpdir(), 'theta-shell-refusal-'));
    const store = new ProductSessionStore(directory);
    const records = new ResearchStore(directory); const approvals = new EffectApprovals(records);
    const session = store.create(); session.runId = 'run-refusal';
    records.put('run', session.runId, { id: session.runId, goal: '探索文本', jobs: [], notes: [] });
    const card = approvals.request(session.id, { action: 'compute.submit', target: 'local', payload: {}, summary: '训练待确认' });
    session.pendingConfirmation = { kind: 'action', runId: session.runId, checkpointId: card.id, contentHash: card.hash, summary: card.summary };
    store.save(session);
    const feedback = mode === 'online' ? '拒绝，先抽样再用 LDA，最多运行 24 小时' : '/deny 不使用云 embedding';
    let called = false;
    const provider: InferenceProvider = { id: 'fixture', infer: async request => {
      called = true;
      assert.equal(approvals.get(card.id).status, 'denied');
      assert.equal(store.get(session.id).pendingConfirmation, undefined);
      assert.ok(JSON.stringify(request.input).includes(feedback));
      if (mode === 'failed') throw new Error('fixture inference unavailable');
      return { id: 'reply', output: { kind: 'tool_calls', toolCalls: [{ id: 'reply', name: 'respond', arguments: { message: '已收到修改要求，将据此调整方案。' } }] } };
    } };
    try {
      const terminal = new Terminal([feedback, '/exit']);
      await new AgentShell({ directory, sessionId: session.id, terminal, inferenceFactory: () => mode === 'offline' ? undefined : provider }).run();
      assert.equal(called, mode !== 'offline');
      assert.equal(approvals.get(card.id).status, 'denied');
      assert.equal(store.get(session.id).pendingConfirmation, undefined);
      assert.deepEqual(records.get<{ notes: string[] }>('run', session.runId).notes, [feedback]);
      assert.ok(terminal.output.some(text => text.includes('反馈已保存')));
      assert.throws(() => approvals.decide(card.id, session.id, card.hash, true), /已处理/);
    } finally { store.close(); await rm(directory, { recursive: true, force: true }); }
  }
});
