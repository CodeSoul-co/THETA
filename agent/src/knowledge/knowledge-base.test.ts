import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdtempSync, readFileSync, writeFileSync, rmSync, copyFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import os from 'node:os';
import path from 'node:path';
import { KnowledgeBase } from './knowledge-base.js';
import { LocalProductTools } from '../tools/local-tools.js';
import { ConversationAgent } from '../conversation/conversation-agent.js';
import type { ProductSession } from '../memory/session-store.js';

const library = new KnowledgeBase();
const id = 'theta-models-parameters';

test('entire imported source, every section, all twelve models and appendix are readable without omission', () => {
  const entry = JSON.parse(readFileSync(path.join(library.directory, 'index.json'), 'utf8')).documents[0];
  const bytes = readFileSync(path.join(library.directory, entry.file));
  assert.equal(createHash('sha256').update(bytes).digest('hex'), entry.importedSha256);
  let full = ''; let offset: number | null = 0;
  while (offset !== null) { const page = library.read({ documentId: id, offset, limit: 3000 }); full += page.content; offset = page.nextOffset; }
  assert.equal(full, bytes.toString('utf8'));
  let sectionText = ''; let sectionCount = 0; offset = 0;
  while (offset !== null) {
    const page = library.list({ documentId: id, offset, limit: 10 });
    for (const item of page.items) {
      assert.ok('sectionId' in item);
      let position: number | null = 0;
      while (position !== null) { const part = library.read({ documentId: id, sectionId: item.sectionId, offset: position, limit: 3000 }); sectionText += part.content; position = part.nextOffset; }
      sectionCount++;
    }
    offset = page.nextOffset;
  }
  assert.equal(sectionText, full); assert.ok(sectionCount >= 40);
  for (const model of ['theta', 'lda', 'hdp', 'stm', 'btm', 'etm', 'ctm', 'dtm', 'nvdm', 'gsm', 'prodlda', 'bertopic']) {
    const hits = library.search(model, id);
    assert.ok(hits.total > 0, `Missing model ${model}`);
    assert.ok(hits.matches.some(item => item.matchedTerms.includes(model)));
  }
  for (const query of ['stage1_epochs', 'no_pin_memory', 'train_all', 'evolution_weight', 'covariates', 'force']) assert.ok(library.search(query).total > 0, query);
  assert.equal(library.search('zzzzUnrelatedzzzz').total, 0);
});

test('knowledge persists across instances, edits update evidence hashes, and paths/IDs cannot escape the registry', () => {
  const directory = mkdtempSync(path.join(os.tmpdir(), 'theta-knowledge-'));
  try {
    const entry = JSON.parse(readFileSync(path.join(library.directory, 'index.json'), 'utf8')).documents[0];
    copyFileSync(path.join(library.directory, entry.file), path.join(directory, 'source.md'));
    entry.file = 'source.md'; writeFileSync(path.join(directory, 'index.json'), JSON.stringify({ version: 1, documents: [entry] }));
    const first = new KnowledgeBase(directory).read({ documentId: id });
    writeFileSync(path.join(directory, 'source.md'), '# Update\nUpdated knowledge.\n```md\n## not a real heading\n```\n');
    const second = new KnowledgeBase(directory).read({ documentId: id });
    assert.notEqual(first.sha256, second.sha256); assert.equal(second.changedSinceImport, true);
    assert.equal(new KnowledgeBase(directory).list({ documentId: id }).total, 1);
    assert.throws(() => library.read({ documentId: '../../.env' }), /不存在/);
    assert.throws(() => library.read({ documentId: id, sectionId: 'missing' }), /不存在/);
    entry.file = '../outside.md'; writeFileSync(path.join(directory, 'index.json'), JSON.stringify({ version: 1, documents: [entry] }));
    assert.throws(() => new KnowledgeBase(directory).read({ documentId: id }));
    writeFileSync(path.join(directory, 'index.json'), '{bad json');
    assert.equal(new KnowledgeBase(directory).catalog().available, false);
  } finally { rmSync(directory, { recursive: true, force: true }); }
});

test('agent uses references without data, and synthesis keeps approved evidence while allowing only knowledge reads', async () => {
  const directory = mkdtempSync(path.join(os.tmpdir(), 'theta-knowledge-agent-'));
  try {
    for (const synthesis of [false, true]) {
      const session: ProductSession = { id: 'chat-reference', title: '知识验证', messages: [{ role: 'assistant', content: 'OLD_UNVERIFIED_RESULT' }], datasetRefs: [], updatedAt: '' };
      if (synthesis) session.pendingSynthesis = { id: 'synthesis', question: '解释方法边界', jobIds: ['job-existing'], contextHash: 'hash' };
      const local = new LocalProductTools({ runtimeDb: path.join(directory, 'research.sqlite'), uploadDir: path.join(directory, 'uploads') });
      const calls: string[] = []; let round = 0; let sectionId = '';
      const tools = { knowledgeCatalog: () => local.knowledgeCatalog(), execute: async (name: string, input: unknown, context: Parameters<LocalProductTools['execute']>[2]) => {
        calls.push(name); const data = await local.execute(name, input, context);
        if (name === 'knowledge_search') sectionId = (data as ReturnType<KnowledgeBase['search']>).matches[0].sectionId;
        return data;
      } };
      const agent = new ConversationAgent({ tools, save() {}, inference: { id: 'fixture', infer: async request => {
        const content = JSON.stringify(request.input);
        assert.match(content, /knowledgeCatalog/); assert.match(content, /theta-models-parameters/);
        if (synthesis) { assert.doesNotMatch(content, /OLD_UNVERIFIED_RESULT/); assert.match(content, /APPROVED_EVIDENCE/); assert.ok(request.tools!.every(tool => tool.name.startsWith('knowledge_') || tool.name === 'respond')); }
        const name = round === 0 ? 'knowledge_search' : round === 1 ? 'knowledge_read' : 'respond';
        if (round === 2) assert.match(content, /reference_document/);
        round++;
        return { id: 'response', output: { kind: 'tool_calls', toolCalls: [{ id: `call-${round}`, name, arguments: name === 'knowledge_search' ? { query: 'STM covariates' } : name === 'knowledge_read' ? { documentId: id, sectionId } : { message: '根据所读章节，STM 的协变量关联不能解释为因果。' } }] } };
      } } });
      await agent.turn(session, synthesis ? 'APPROVED_EVIDENCE' : '解释 STM 协变量，不训练');
      assert.deepEqual(calls, ['knowledge_search', 'knowledge_read']); assert.equal(session.runId, undefined); assert.equal(session.pendingConfirmation, undefined);
    }
  } finally { rmSync(directory, { recursive: true, force: true }); }
});
