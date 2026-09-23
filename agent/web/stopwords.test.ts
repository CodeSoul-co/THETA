import test from 'node:test';
import assert from 'node:assert/strict';
import { parseStopwords, exportBuiltinStopwords, STOPWORD_BYTES } from './stopwords.js';
import { repositoryRoot } from '../src/environment.js';

test('停用词表接受 UTF-8、BOM、混合语言、注释与去重', () => {
  assert.deepEqual(parseStopwords(Buffer.from('\ufeff# 注释\r\n分析\r\nThe\r\nthe\nＣＵＳＴＯＭ\nрусский\n')), ['分析', 'the', 'custom', 'русский']);
  for (const data of [Buffer.from(''), Buffer.from('# 注释\n'), Buffer.from([0xff, 0xfe]), Buffer.from('a\0b'), Buffer.alloc(STOPWORD_BYTES + 1), Buffer.from('a'.repeat(101))]) {
    assert.throws(() => parseStopwords(data));
  }
});

test('导出来自实际引擎内置词表且可以重新上传', () => {
  const exported = exportBuiltinStopwords(repositoryRoot());
  assert.match(exported, /# zh\.txt/u);
  assert.match(exported, /# en\.txt/u);
  assert.match(exported, /# ru\.txt/u);
  const words = parseStopwords(Buffer.from(exported));
  assert.ok(words.includes('the'));
  assert.ok(words.includes('的'));
  assert.ok(words.length > 5000);
});
