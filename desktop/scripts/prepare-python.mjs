import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, readdirSync, renameSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const desktop = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const runtime = path.join(desktop, 'runtime');
const pythonHome = path.join(runtime, 'python');
const python = path.join(pythonHome, process.platform === 'win32' ? 'python.exe' : 'bin/python3');
function run(command, args, env = {}) {
  const result = spawnSync(command, args, { cwd: desktop, stdio: 'inherit', env: { ...process.env, ...(process.platform === 'win32' ? { UV_NO_CACHE: '1' } : {}), ...env } });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`${command} exited ${result.status}`);
}
mkdirSync(runtime, { recursive: true });
if (!existsSync(python)) {
  const staging = path.join(desktop, '.python-download');
  run('uv', ['python', 'install', '3.12.13', '--install-dir', staging, '--no-bin', '--no-registry']);
  const installation = readdirSync(staging).find(name => name.startsWith('cpython-3.12.13-'));
  if (!installation) throw new Error('Managed Python installation not found');
  renameSync(path.join(staging, installation), pythonHome);
  rmSync(staging, { recursive: true, force: true });
}
const lock = path.join(desktop, `requirements-${process.platform}-${process.arch}${process.platform === 'win32' ? '-cpu' : ''}.txt`);
if (!existsSync(lock)) {
  run('uv', ['pip', 'compile', 'requirements.in', '--python', python, '--output-file', lock, '--custom-compile-command', 'npm run prepare:python',
    ...(process.platform === 'win32' ? ['--constraint', 'requirements-windows.in'] : [])]);
}
run('uv', ['pip', 'install', '--python', python, '--system', '--break-system-packages', '--link-mode', 'copy', '-r', lock]);
if (process.platform === 'win32') run(python, ['-I', '-c', 'import torch; assert torch.version.cuda is None and torch.__version__.startswith("2.11.0"), "Windows CPU runtime must match the optional CUDA component"; print("Bundled CPU torch:", torch.__version__)']);
if (process.platform === 'darwin') run('uvx', ['--from', 'delocate', 'python', 'scripts/repair-macos.py', pythonHome]);
run(python, ['-I', '-c', 'import sys, ssl, sqlite3, numpy, pandas, scipy, torch, sklearn, gensim, jieba, nltk, transformers; print("Bundled Python:", sys.version, sys.prefix)']);
const inventory = spawnSync('uv', ['pip', 'freeze', '--python', python], { encoding: 'utf8' });
if (inventory.status !== 0) throw new Error(inventory.stderr);
writeFileSync(path.join(pythonHome, 'THETA-dependencies.txt'), inventory.stdout);
console.log(`Bundled Python ready: ${python}`);
