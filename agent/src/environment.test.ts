import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import test from 'node:test';

test('Agent environment precedence: process > package > repository; explicit file is exclusive', () => {
  const root = mkdtempSync(path.join(tmpdir(), 'theta-env-'));
  const packageRoot = path.join(root, 'agent'); mkdirSync(packageRoot);
  try {
    writeFileSync(path.join(root, '.env.local'), 'ROOT_ONLY=yes\nCHOICE=root\n');
    writeFileSync(path.join(packageRoot, '.env'), 'PACKAGE_ONLY=yes\nCHOICE=package\n');
    writeFileSync(path.join(packageRoot, '.env.local'), 'CHOICE=local\n');
    const script = `import { loadThetaProjectEnvironment } from ${JSON.stringify(new URL('./environment.js', import.meta.url).href)}; loadThetaProjectEnvironment({ packageRoot: ${JSON.stringify(packageRoot)} }); console.log(JSON.stringify(Object.fromEntries(['CHOICE','ROOT_ONLY','PACKAGE_ONLY'].map(key => [key, process.env[key] ?? null]))));`;
    const run = (env: NodeJS.ProcessEnv = {}) => {
      const result = spawnSync(process.execPath, ['--input-type=module', '-e', script], { env, encoding: 'utf8' });
      assert.equal(result.status, 0, result.stderr); return JSON.parse(result.stdout);
    };
    assert.deepEqual(run(), { CHOICE: 'local', ROOT_ONLY: 'yes', PACKAGE_ONLY: 'yes' });
    assert.equal(run({ CHOICE: 'process' }).CHOICE, 'process');
    assert.deepEqual(run({ THETA_ENV_FILE: path.join(packageRoot, '.env') }), { CHOICE: 'package', ROOT_ONLY: null, PACKAGE_ONLY: 'yes' });
    assert.equal(run({ THETA_ENV_FILE: path.join(root, 'missing') }).CHOICE, null);
  } finally { rmSync(root, { recursive: true, force: true }); }
});
