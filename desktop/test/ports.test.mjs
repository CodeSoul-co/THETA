import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import ports from '../ports.cjs';

function servers(firstError, calls) {
  return () => {
    const server = new EventEmitter();
    server.listen = (port, host, ready) => {
      calls.push({ port, host });
      queueMicrotask(() => {
        if (calls.length === 1 && firstError) server.emit('error', Object.assign(new Error(firstError), { code: firstError }));
        else ready();
      });
    };
    server.address = () => ({ port: 18080 });
    server.close = done => queueMicrotask(() => done());
    return server;
  };
}

for (const code of ['EACCES', 'EADDRINUSE']) {
  test(`unavailable saved port (${code}) switches to an OS-selected port`, async () => {
    const calls = [];
    assert.equal(await ports.selectPort(14320, servers(code, calls)), 18080);
    assert.deepEqual(calls, [{ port: 14320, host: '127.0.0.1' }, { port: 0, host: '127.0.0.1' }]);
  });
}

test('other failures and errors on the fallback port remain visible', async () => {
  await assert.rejects(ports.selectPort(14320, servers('EADDRNOTAVAIL', [])), { code: 'EADDRNOTAVAIL' });
  await assert.rejects(ports.selectPort(0, servers('EACCES', [])), { code: 'EACCES' });
});

test('invalid persisted ports use the default, while zero requests an automatic port', () => {
  const home = mkdtempSync(path.join(tmpdir(), 'theta-port-'));
  const file = path.join(home, 'port.json');
  try {
    assert.equal(ports.preferredPort(file), 14320);
    for (const value of ['{"port":65536}', '{"port":"14320"}', '{broken']) {
      writeFileSync(file, value);
      assert.equal(ports.preferredPort(file), 14320);
    }
    writeFileSync(file, '{"port":0}');
    assert.equal(ports.preferredPort(file), 0);
    writeFileSync(file, '{"port":18080}');
    assert.equal(ports.preferredPort(file), 18080);
  } finally { rmSync(home, { recursive: true, force: true }); }
});
