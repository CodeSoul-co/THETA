import { existsSync } from 'node:fs';
import { loadEnvFile } from 'node:process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const packageRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
export const repositoryRoot = (): string => path.resolve(process.env.THETA_PROJECT_ROOT ?? path.join(packageRoot, '..'));
export const loadThetaProjectEnvironment = (options: { packageRoot?: string } = {}): string | undefined => {
  const root = options.packageRoot ?? packageRoot;
  const candidates = process.env.THETA_ENV_FILE !== undefined ? [path.resolve(process.env.THETA_ENV_FILE)]
    : [root, path.resolve(root, '..')].flatMap((dir) => ['.env.local', '.env'].map((name) => path.join(dir, name)));
  let first: string | undefined;
  for (const file of candidates) {
    if (!existsSync(file)) continue;
    loadEnvFile(file); first ??= file;
  }
  return first;
};
