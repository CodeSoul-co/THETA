import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import path from 'node:path';
import { tmpdir } from 'node:os';
import { ResearchContexts } from './research-context.js';
import { ResearchStore } from './research-store.js';

test('Markdown context persists, is reusable as a copy, and cannot mutate another session context', () => {
  const home = mkdtempSync(path.join(tmpdir(), 'theta-context-'));
  try {
    const contexts = new ResearchContexts(new ResearchStore(home));
    const source = contexts.save('first', { title: '售后研究', goal: '改善退款体验', understanding: '样本涉及退款等待；原因尚需验证。', questions: ['关注哪个业务环节？'], datasetRefs: ['dataset-evidence'] });
    assert.match(readFileSync(source.documentPath, 'utf8'), /改善退款体验/);
    assert.match(contexts.read(source.id).markdown, /dataset-evidence/);
    const reopened = new ResearchContexts(new ResearchStore(home));
    const copy = reopened.select('second', source.id);
    assert.notEqual(copy.id, source.id); assert.equal(copy.sourceContextId, source.id);
    reopened.save('second', { ...copy, goal: '比较新季度数据' }, copy.id);
    assert.equal(reopened.read(source.id).goal, '改善退款体验');
    assert.equal(reopened.list().length, 2);
    assert.throws(() => reopened.save('second', source, source.id), /副本/);
    assert.throws(() => reopened.read('../../.env'));
  } finally { rmSync(home, { recursive: true, force: true }); }
});
