import { createReadStream } from 'node:fs';
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { homedir } from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

// Compare values in memory; never print credentials, their hashes, or file contents.
export async function inspectPackage(root, privateValues = []) {
  const needles = [...new Set(privateValues.filter(value => typeof value === 'string' && value.length >= 8))].map(value => Buffer.from(value));
  const overlap = Math.max(0, ...needles.map(value => value.length - 1));
  let files = 0;
  async function walk(directory) {
    for (const entry of readdirSync(directory, { withFileTypes: true })) {
      const file = path.join(directory, entry.name);
      const relative = path.relative(root, file);
      if ((entry.name.startsWith('.env') && entry.name !== '.env.example')
        || ['settings.json', 'credentials.json', '.npmrc', '.pypirc', '.smoke-home'].includes(entry.name)) {
        throw new Error(`Private configuration found in package: ${relative}`);
      }
      if (entry.isDirectory()) { await walk(file); continue; }
      if (!entry.isFile()) continue;
      files++;
      if (!needles.length) continue;
      let tail = Buffer.alloc(0);
      for await (const chunk of createReadStream(file)) {
        const data = Buffer.concat([tail, chunk]);
        if (needles.some(value => data.includes(value))) throw new Error(`Local credential found in package: ${relative}`);
        tail = overlap ? data.subarray(-overlap) : Buffer.alloc(0);
      }
    }
  }
  await walk(root);
  return files;
}

export function localCredentials(root) {
  const values = Object.entries(process.env)
    .filter(([key]) => /^(OPENAI|DEEPSEEK|MINIMAX|GLM|EMBEDDING|THETA_INFERENCE)_.*(KEY|TOKEN|SECRET)$/i.test(key))
    .map(([, value]) => value);
  for (const directory of [root, path.join(root, 'agent'), path.join(root, 'frontend')]) {
    for (const name of readdirSync(directory).filter(name => name.startsWith('.env') && name !== '.env.example')) {
      for (const line of readFileSync(path.join(directory, name), 'utf8').split(/\r?\n/)) {
        const match = line.match(/^\s*(?:export\s+)?[\w]*(?:KEY|TOKEN|SECRET|PASSWORD)[\w]*\s*=\s*(.*?)\s*$/i);
        if (match) values.push(match[1].replace(/^(['"])(.*)\1$/, '$2'));
      }
    }
  }
  const appData = process.platform === 'win32' ? process.env.APPDATA : path.join(homedir(), 'Library/Application Support');
  const settingsFile = appData && path.join(appData, 'THETA/settings.json');
  if (settingsFile && existsSync(settingsFile)) {
    const settings = JSON.parse(readFileSync(settingsFile, 'utf8'));
    values.push(settings.encryptedKey, settings.embedding?.encryptedKey,
      ...Object.values(settings.providers ?? {}).map(provider => provider.encryptedKey));
  }
  return values;
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  const desktop = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
  const root = path.dirname(desktop);
  const values = localCredentials(root);
  const count = await inspectPackage(path.join(desktop, 'runtime'), values);
  for (const name of ['main.cjs', 'preload.cjs', 'services.mjs', 'settings.cjs']) {
    const contents = readFileSync(path.join(desktop, name));
    if (values.some(value => typeof value === 'string' && value.length >= 8 && contents.includes(Buffer.from(value)))) {
      throw new Error(`Local credential found in application file: ${name}`);
    }
  }
  if (!readFileSync(path.join(desktop, 'ui/icon.png')).equals(readFileSync(path.join(root, 'frontend/public/theta-assets/brand/theta-cat-favicon-master-v1.png')))) {
    throw new Error('Desktop icon must match the approved ragdoll cat logo');
  }
  console.log(`Package check passed: ${count} runtime files; no private configuration or matching local credential; ragdoll cat icon verified.`);
}
