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
export class ProcessCapabilityWorker implements CapabilityWorker {
  async call<T>(operation: string, input: unknown, signal?: AbortSignal): Promise<T> {
    return new Promise((resolve, reject) => {
      const child = spawn(pythonExecutable(), ['-m', 'workers', operation], { cwd: packageRoot, env: { ...process.env, THETA_PROJECT_ROOT: repositoryRoot(), THETA_WORKER_CONTROL_PYTHON: pythonExecutable(), PYTHONNOUSERSITE: '1', PYTHONUTF8: '1' }, windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'] });
      child.stdout.setEncoding('utf8'); child.stderr.setEncoding('utf8');
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
      const timeoutMs = ['compute.results', 'figure.adjust', 'analysis_report.evidence', 'analysis_report.pdf'].includes(operation) ? Number(process.env.THETA_REPORT_TIMEOUT_MS ?? 600000) : defaultTimeoutMs;
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

let localApi: Promise<{ url: string; token: string }> | undefined;
/** Web, desktop and CLI share the same HTTP contract. Python stays inside the worker. */
export class PythonCapabilityWorker implements CapabilityWorker {
  async call<T>(operation: string, input: unknown, signal?: AbortSignal): Promise<T> {
    signal?.throwIfAborted();
    const endpoint = process.env.THETA_WORKER_API_URL;
    const connection = endpoint ? { url: endpoint, token: process.env.THETA_WORKER_API_TOKEN ?? '' }
      : await (localApi ??= import('../../web/worker-server.js').then(module => module.startLocalWorkerApi()));
    const url = new URL(connection.url);
    if (url.username || url.password || url.search || url.hash || !(url.protocol === 'https:' || url.protocol === 'http:' && ['127.0.0.1', 'localhost', '[::1]'].includes(url.hostname))) throw new Error('Worker API requires HTTPS or a loopback HTTP address');
    if (!connection.token) throw new Error('Worker API token is missing');
    const timeout = operation === 'compute.status' ? 7000 : ['compute.results', 'figure.adjust', 'analysis_report.evidence', 'analysis_report.pdf'].includes(operation) ? Number(process.env.THETA_REPORT_TIMEOUT_MS ?? 600000) + 5000 : Number(process.env.THETA_CAPABILITY_TIMEOUT_MS ?? 120000) + 5000;
    try {
      const response = await fetch(connection.url.replace(/\/$/, '') + '/v1/capabilities/' + encodeURIComponent(operation), {
        method: 'POST', headers: { authorization: `Bearer ${connection.token}`, 'content-type': 'application/json' },
        body: JSON.stringify(input), signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(timeout)]) : AbortSignal.timeout(timeout),
      });
      const result = await response.json() as { ok: boolean; data?: T; error?: string };
      if (!response.ok || !result.ok) throw new Error(result.error ?? `Worker API HTTP ${response.status}`);
      return result.data as T;
    } catch (error) {
      if (signal?.aborted) throw new Error('能力调用已中断');
      throw error;
    }
  }
}
