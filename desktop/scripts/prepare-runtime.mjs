import { spawnSync } from 'node:child_process';
import { cpSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const desktop = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const root = path.dirname(desktop);
const runtime = path.join(desktop, 'runtime');
const run = (args, cwd) => {
  const result = spawnSync(process.platform === 'win32' ? 'npm.cmd' : 'npm', args, {
    cwd, stdio: 'inherit', shell: process.platform === 'win32', env: { ...process.env, NEXT_TELEMETRY_DISABLED: '1' },
  });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`npm ${args.join(' ')} failed`);
};
run(['run', 'build'], path.join(root, 'agent'));
run(['run', 'build', '--', '--webpack'], path.join(root, 'frontend'));
mkdirSync(runtime, { recursive: true });
const allowed = source => {
  const name = path.basename(source);
  return !['__pycache__', '.DS_Store', '.venv', '.git', '.local', 'node_modules'].includes(name)
    && !(name.startsWith('.env') && name !== '.env.example') && !name.endsWith('.pyc');
};
for (const name of ['agent', 'frontend', 'src', 'trainning', 'config', 'skills']) rmSync(path.join(runtime, name), { recursive: true, force: true });
for (const name of ['agent/dist', 'agent/workers', 'agent/skills', 'agent/knowledge', 'agent/cli/bin', 'src', 'trainning/worker', 'config', 'skills']) {
  cpSync(path.join(root, name), path.join(runtime, name), { recursive: true, filter: allowed });
}
const pkg = JSON.parse(readFileSync(path.join(root, 'agent/package.json')));
for (const name of Object.keys(pkg.dependencies)) {
  pkg.dependencies[name] = JSON.parse(readFileSync(path.join(root, 'agent/node_modules', name, 'package.json'))).version;
}
delete pkg.devDependencies; delete pkg.scripts;
writeFileSync(path.join(runtime, 'agent/package.json'), JSON.stringify(pkg, null, 2));
run(['install', '--omit=dev', '--ignore-scripts', '--no-audit', '--no-fund'], path.join(runtime, 'agent'));
const standalone = path.join(root, 'frontend/.next/standalone');
if (!existsSync(path.join(standalone, 'server.js'))) throw new Error('Expected frontend/.next/standalone/server.js; check Next tracing root');
cpSync(standalone, path.join(runtime, 'frontend'), { recursive: true, dereference: true,
  filter: source => !(path.basename(source).startsWith('.env')) });
cpSync(path.join(root, 'frontend/.next/static'), path.join(runtime, 'frontend/.next/static'), { recursive: true });
cpSync(path.join(root, 'frontend/public'), path.join(runtime, 'frontend/public'), { recursive: true });
cpSync(path.join(root, 'LICENSE'), path.join(runtime, 'LICENSE'));
cpSync(path.join(root, 'frontend/public/theta-assets/brand/theta-cat-favicon-master-v1.png'), path.join(desktop, 'ui/icon.png'));
writeFileSync(path.join(runtime, 'build.json'), JSON.stringify({ version: pkg.version, platform: process.platform, arch: process.arch, builtAt: new Date().toISOString() }, null, 2));
console.log(`Runtime prepared: ${runtime}`);
