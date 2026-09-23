import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { parseAttachmentInput } from './attachment-input.js';

test('attachment accepts quoted/escaped paths and preserves optional user instructions', async () => {
  const root = mkdtempSync(path.join(tmpdir(), 'theta-attach-')); const file = path.join(root, '业务 数据.csv');
  writeFileSync(file, 'text\nfixture\n');
  try {
    for (const input of [`"${file}"`, `/attach '${file}'`, file.replaceAll(' ', '\\ ')]) {
      assert.deepEqual(await parseAttachmentInput(input), { filePath: file, instruction: '' });
    }
    for (const input of [`"${file}" 只回答业务问题，不训练`, `/attach ${file.replaceAll(' ', '\\ ')} -- 只回答业务问题，不训练`]) {
      assert.deepEqual(await parseAttachmentInput(input), { filePath: file, instruction: '只回答业务问题，不训练' });
    }
    assert.equal(await parseAttachmentInput('请解释 data.csv 是什么'), undefined);
    await assert.rejects(parseAttachmentInput('/attach /missing.csv 做分析'), /未找到/);
  } finally { rmSync(root, { recursive: true, force: true }); }
});
