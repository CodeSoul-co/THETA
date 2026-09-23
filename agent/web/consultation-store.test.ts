import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { ConsultationStore } from './consultation-store.js';

test('consultations migrate idempotently, persist pin/delete/selection, and isolate owners and scopes', t => {
  const home = mkdtempSync(path.join(tmpdir(), 'theta-consultation-'));
  t.after(() => rmSync(home, { recursive: true, force: true }));
  const store = new ConsultationStore(home);
  const legacy = [{ id: 'legacy', title: '旧咨询', updatedAt: '2026-09-21T00:00:00Z', messages: [{ id: 'm1', role: 'user', content: '旧问题', timestamp: '10:00' }] }];
  store.import('alice', 'manual:p1', legacy, 'legacy');
  store.create('alice', 'manual:p1', 'new');
  store.patch('alice', 'manual:p1', 'legacy', { pinned: true });
  assert.equal(store.list('alice', 'manual:p1').threads[0].id, 'legacy');
  assert.equal(store.list('alice', 'manual:p1').activeId, 'new');
  store.patch('alice', 'manual:p1', 'legacy', { deleted: true });
  store.import('alice', 'manual:p1', legacy, 'legacy');
  const restored = new ConsultationStore(home).list('alice', 'manual:p1');
  assert.equal(restored.threads.length, 2);
  assert.ok(restored.threads[0].deletedAt);
  assert.equal(restored.threads[0].pinned, true);
  assert.equal(restored.threads[0].messages[0].content, '旧问题');
  assert.deepEqual(store.list('bob', 'manual:p1').threads, []);
  assert.deepEqual(store.list('alice', 'manual:p2').threads, []);
  assert.throws(() => store.patch('bob', 'manual:p1', 'legacy', { pinned: true }), /不存在/);
  assert.throws(() => store.create('alice', 'manual:p1', 'legacy'), /已删除/);
  store.patch('alice', 'manual:p1', 'legacy', { deleted: false, select: true });
  assert.equal(store.list('alice', 'manual:p1').activeId, 'legacy');
});

test('in-flight replies preserve pin/delete, are idempotent, and interrupted replies recover on restart', t => {
  const home = mkdtempSync(path.join(tmpdir(), 'theta-consultation-'));
  t.after(() => rmSync(home, { recursive: true, force: true }));
  const store = new ConsultationStore(home);
  store.create('alice', 'p', 'one');
  const payload = { charts: [{ chartName: '主题占比', sources: [{ content: 'topic,weight\n1,1' }] }] };
  store.begin('alice', 'p', 'one', 'req', '解读结果', payload);
  assert.equal(store.begin('alice', 'p', 'one', 'req', '解读结果', payload).reused, true);
  assert.throws(() => store.begin('alice', 'p', 'one', 'req', '另一个问题', payload), /不同内容/);
  assert.throws(() => store.begin('alice', 'p', 'one', 'req2', '另一个问题', payload), /正在回复/);
  store.patch('alice', 'p', 'one', { deleted: true, pinned: true });
  store.finish('alice', 'p', 'one', 'req', '主题占比为 1');
  const thread = store.list('alice', 'p').threads[0];
  assert.ok(thread.deletedAt); assert.equal(thread.pinned, true);
  assert.deepEqual(thread.messages[0].data, payload);
  assert.equal(thread.messages[1].content, '主题占比为 1');
  assert.equal(thread.messages.length, 2);
  store.patch('alice', 'p', 'one', { deleted: false });
  store.begin('alice', 'p', 'one', 'req2', '还有什么？', {});
  const restored = new ConsultationStore(home).list('alice', 'p').threads[0];
  assert.equal(restored.messages.at(-1)?.isThinking, false);
  assert.match(restored.messages.at(-1)!.content, /服务已重新启动/);
});

test('failed legacy imports are atomic and more than 100 consultations remain visible', t => {
  const home = mkdtempSync(path.join(tmpdir(), 'theta-consultation-'));
  t.after(() => rmSync(home, { recursive: true, force: true }));
  const store = new ConsultationStore(home);
  assert.throws(() => store.import('alice', 'p', [{ id: 'valid', messages: [] }, { id: '../invalid', messages: [] }]), /编号无效/);
  assert.equal(store.list('alice', 'p').threads.length, 0);
  store.import('alice', 'p', Array.from({ length: 120 }, (_, index) => ({ id: `t${index}`, messages: [] })));
  assert.equal(store.list('alice', 'p').threads.length, 120);
});
