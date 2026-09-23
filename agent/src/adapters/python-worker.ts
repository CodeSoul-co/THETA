import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { packageRoot, repositoryRoot } from '../environment.js';

/** JSON capability transport. Swap this port for RPC without changing conversation/tools. */
export interface CapabilityWorker { call<T = Record<string, unknown>>(operation: string, input: unknown, signal?: AbortSignal): Promise<T> }
export const pythonExecutable = (): string => {
  const local = path.join(packageRoot, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
  return process.env.THETA_PYTHON ?? (existsSync(local) ? local : 'python3');
};
export class PythonCapabilityWorker implements CapabilityWorker {
  async call<T>(operation: string, input: unknown, signal?: AbortSignal): Promise<T> {
    return new Promise((resolve, reject) => {
      const child = spawn(pythonExecutable(), ['-m', 'workers', operation], { cwd: packageRoot, env: { ...process.env, THETA_PROJECT_ROOT: repositoryRoot(), THETA_WORKER_CONTROL_PYTHON: pythonExecutable(), PYTHONNOUSERSITE: '1' }, stdio: ['pipe', 'pipe', 'pipe'] });
      let stdout = ''; let stderr = ''; let failure: Error | undefined;
      let forcedStop: ReturnType<typeof setTimeout> | undefined;
      const stop = (error: Error): void => {
        if (failure) return;
        failure = error; child.kill('SIGTERM');
        forcedStop = setTimeout(() => child.kill('SIGKILL'), 2000);
      };
      const abort = (): void => stop(new Error('能力调用已中断'));
      // Status is a bounded read. A stuck observer must not freeze interactive input.
      // Report views run the native visualization, which can exceed two minutes on a
      // large corpus; allow bounded native rendering without extending training.
      const defaultTimeoutMs = operation === 'compute.status' ? 5000 : Number(process.env.THETA_CAPABILITY_TIMEOUT_MS ?? 120000);
      const timeoutMs = ['compute.results', 'figure.adjust'].includes(operation) ? Number(process.env.THETA_REPORT_TIMEOUT_MS ?? 600000) : defaultTimeoutMs;
      const timer = setTimeout(() => stop(new Error(`能力调用超过 ${timeoutMs / 1000} 秒`)), timeoutMs);
      const killTimer = setTimeout(() => child.kill('SIGKILL'), timeoutMs + 2000);
      signal?.addEventListener('abort', abort, { once: true });
      child.stdout.on('data', (chunk) => { stdout += chunk; if (Buffer.byteLength(stdout) > 2 * 1024 * 1024) stop(new Error('Worker 输出超出预算')); });
      child.stderr.on('data', (chunk) => { stderr = (stderr + chunk).slice(-4000); });
      child.stdin.on('error', () => {});
      child.on('error', (error) => { failure = error; });
      child.on('close', (code) => {
        clearTimeout(timer); clearTimeout(killTimer); clearTimeout(forcedStop); signal?.removeEventListener('abort', abort);
        if (failure) return reject(failure);
        try { const result = JSON.parse(stdout); if (!result.ok) throw new Error(result.error ?? 'Worker failed'); if (code !== 0) throw new Error(`Worker exited ${code}`); resolve(result.data as T); }
        catch (error) { reject(new Error(`计算能力 ${operation}：${error instanceof Error ? error.message : String(error)}${stdout ? '' : ` (${stderr.slice(-500)})`}`)); }
      });
      child.stdin.end(JSON.stringify(input));
      if (signal?.aborted) abort();
    });
  }
}
