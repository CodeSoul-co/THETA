import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

test('JSON tool protocol returns stable execution events without invoking inference', () => {
  const home = mkdtempSync(path.join(tmpdir(), 'theta-json-events-'));
  try {
    const data = path.join(home, 'data'); mkdirSync(data); writeFileSync(path.join(data, 'fixture.csv'), 'text\nfixture\n');
    const script = `import { main } from ${JSON.stringify(new URL('../entry.js', import.meta.url).href)}; process.exitCode = await main(process.argv.slice(1));`;
    const run = (args: string[]) => {
      const result = spawnSync(process.execPath, ['--disable-warning=ExperimentalWarning', '--input-type=module', '-e', script, ...args], {
        env: { ...process.env, THETA_AGENT_HOME: home, THETA_DATA_DIR: data }, input: '{}', encoding: 'utf8', timeout: 10000,
      });
      assert.equal(result.status, 0, result.stderr); return JSON.parse(result.stdout);
    };
    const session = run(['tools', 'call', 'session_create']);
    const result = run(['tools', 'call', 'datasets_discover', '--session', session.sessionId]);
    assert.equal(result.data.datasets[0].name, 'fixture.csv');
    assert.deepEqual(result.executionEvents.map((event: { status: string }) => event.status), ['started', 'completed']);
    assert.equal(result.executionEvents[0].callId, result.executionEvents[1].callId);
    assert.ok(result.executionEvents[1].durationMs >= 0);
  } finally { rmSync(home, { recursive: true, force: true }); }
});
