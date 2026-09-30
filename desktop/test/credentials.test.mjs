import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { createCredentials } from '../credentials.cjs';
test('credentials survive updates without an OS password and reject tampering', () => {
  const home = mkdtempSync(path.join(tmpdir(), 'theta-keys-'));
  try {
    const vault = createCredentials(home, () => { throw new Error('Keychain called'); });
    const encrypted = vault.encrypt('user-entered-secret');
    assert.equal(createCredentials(home).decrypt(encrypted), 'user-entered-secret');
    assert.ok(!encrypted.includes('user-entered-secret'));
    const bytes = Buffer.from(encrypted.slice(9), 'base64'); bytes[30] ^= 1;
    assert.throws(() => vault.decrypt('local:v1:' + bytes.toString('base64')));
    if (process.platform !== 'win32') assert.equal(statSync(path.join(home, 'credentials.key')).mode & 0o777, 0o600);
  } finally { rmSync(home, { recursive: true, force: true }); }
});
