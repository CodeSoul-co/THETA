import { readdirSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const root = fileURLToPath(new URL('../', import.meta.url));
const files = (directory) => readdirSync(directory, { withFileTypes: true }).flatMap((entry) => entry.isDirectory() ? files(path.join(directory, entry.name)) : [path.join(directory, entry.name)]);
for (const file of files(path.join(root, 'src')).filter((name) => name.endsWith('.ts'))) {
  const text = readFileSync(file, 'utf8');
  if (/(?:from\s+|import\s*\()\s*['"][^'"]*(?:\/cli\/|theta_project|theta\.code-soul\.com|readline)/u.test(text)) throw new Error(`Core imports UI/legacy runtime: ${file}`);
}
if (files(path.join(root, 'cli')).some((file) => path.relative(root, file).split(path.sep).includes('src'))) throw new Error('CLI must not have a src directory');
console.log('Architecture boundaries: core independent from CLI/Web/legacy runtime; CLI outside src.');
