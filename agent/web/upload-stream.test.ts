import test from 'node:test';
import assert from 'node:assert/strict';
import { Readable } from 'node:stream';
import { mkdtempSync, statSync, existsSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import type { IncomingMessage } from 'node:http';
import { receiveUpload } from './upload-stream.js';

test('streaming accepts more than 200 MiB and removes truncated uploads', async () => {
  const home = mkdtempSync(path.join(tmpdir(), 'theta-stream-test-'));
  try {
    const chunk = Buffer.alloc(1024 * 1024, 97);
    const request = (count: number, expected: number) => Object.assign(Readable.from((function* () { for(let i=0;i<count;i++) yield chunk; })()), { headers: {'x-theta-file-size': String(expected)} }) as unknown as IncomingMessage;
    const file = path.join(home, 'large.txt');
    assert.equal(await receiveUpload(request(201,201*chunk.length),file),201*chunk.length);
    assert.equal(statSync(file).size,201*chunk.length);
    const bad = path.join(home,'bad.txt');
    await assert.rejects(receiveUpload(request(1,2*chunk.length),bad),/不完整/);
    assert.equal(existsSync(bad),false);
  } finally { rmSync(home,{recursive:true,force:true}); }
});
