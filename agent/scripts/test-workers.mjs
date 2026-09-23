import { spawnSync } from 'node:child_process';
import { loadThetaProjectEnvironment, packageRoot } from '../dist/src/environment.js';
import { pythonExecutable } from '../dist/src/adapters/python-worker.js';

loadThetaProjectEnvironment();
const python = pythonExecutable();
console.log(`Worker tests: ${python}`);
const result = spawnSync(python, ['-m', 'unittest', 'discover', '-s', 'workers/tests', '-t', '.'], {
  cwd: packageRoot, stdio: 'inherit', env: process.env,
});
if (result.error) console.error(result.error.message);
process.exit(result.status ?? 1);
