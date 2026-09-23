import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { inspectPackage } from '../scripts/check-package.mjs';

test('packaging rejects private files and credentials embedded in compiled assets', async t => {
  const directory = mkdtempSync(path.join(tmpdir(), 'theta-package-'));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  for (const name of ['.env.local', 'settings.json', 'credentials.json']) {
    writeFileSync(path.join(directory, name), '{}');
    await assert.rejects(inspectPackage(directory), /Private configuration/);
    rmSync(path.join(directory, name));
  }
  const key = 'synthetic-private-key-for-packaging-test';
  // The value deliberately straddles the file stream chunk boundary.
  writeFileSync(path.join(directory, 'bundle.js'), ' '.repeat(65530) + key);
  await assert.rejects(inspectPackage(directory, [key]), error => {
    assert.ok(!error.message.includes(key));
    return /Local credential/.test(error.message);
  });
  writeFileSync(path.join(directory, 'bundle.js'), 'API_KEY must be entered by the user');
  assert.equal(await inspectPackage(directory, [key]), 1);
});
