#!/usr/bin/env node
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
const root = fileURLToPath(new URL('../../', import.meta.url));
const entry = new URL('../../dist/cli/entry.js', import.meta.url);
if (!existsSync(entry)) {
  const tsc = new URL('../../node_modules/typescript/bin/tsc', import.meta.url);
  if (!existsSync(tsc)) { console.error('请先在 agent/ 运行 pnpm install，再运行 pnpm start。'); process.exit(1); }
  const result = spawnSync(process.execPath, [fileURLToPath(tsc), '-p', 'tsconfig.json'], { cwd: root, stdio: 'inherit' });
  if (result.status !== 0) process.exit(result.status ?? 1);
}
const { main } = await import(entry.href);
process.exitCode = await main(process.argv.slice(2));
