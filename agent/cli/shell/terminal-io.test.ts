import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';

test('input entered while busy is buffered and wakes the host instead of being lost', async () => {
  const url = new URL('./terminal-io.js', import.meta.url).href;
  const script = `import { ReadlineInteractiveTerminal } from ${JSON.stringify(url)};
    const t = new ReadlineInteractiveTerminal();
    const pending = new Promise(resolve => t.onPendingInput(resolve));
    console.log('READY'); await pending;
    console.log('RECEIVED:' + await t.question('next')); t.close();`;
  const child = spawn(process.execPath, ['--input-type=module', '-e', script], { stdio: ['pipe', 'pipe', 'pipe'] });
  let output = ''; let sent = false;
  const timer = setTimeout(() => child.kill(), 5000);
  child.stdout.on('data', chunk => {
    output += chunk;
    if (!sent && output.includes('READY')) { sent = true; child.stdin.write('只理解业务，不训练\n'); }
  });
  try {
    const code = await new Promise<number | null>((resolve, reject) => { child.on('close', resolve); child.on('error', reject); });
    assert.equal(code, 0); assert.match(output, /RECEIVED:只理解业务，不训练/);
  } finally { clearTimeout(timer); child.kill(); }
});
