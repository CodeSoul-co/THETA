import { createServer } from 'node:http';
import { randomBytes, timingSafeEqual } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { ProcessCapabilityWorker, type CapabilityWorker } from '../src/adapters/python-worker.js';
import { loadThetaProjectEnvironment } from '../src/environment.js';

export function createWorkerServer(token: string, worker: CapabilityWorker = new ProcessCapabilityWorker()) {
  if (token.length < 32) throw new Error('Worker API needs a private access token');
  const server = createServer(async (req, res) => {
    const reply = (status: number, value: unknown) => { res.writeHead(status, { 'content-type': 'application/json', 'cache-control': 'no-store' }); res.end(JSON.stringify(value)); };
    const auth = Buffer.from(req.headers.authorization ?? ''), expected = Buffer.from(`Bearer ${token}`);
    if (req.headers.origin || auth.length !== expected.length || !timingSafeEqual(auth, expected)) return reply(403, { ok: false, error: 'Worker API access denied' });
    if (req.method === 'GET' && req.url === '/health') return reply(200, { ok: true, data: { service: 'theta-worker', protocol: 1 } });
    const match = req.url?.match(/^\/v1\/capabilities\/([a-z_]+\.[a-z_]+)$/);
    if (req.method !== 'POST' || !match) return reply(404, { ok: false, error: 'Unknown worker endpoint' });
    const controller = new AbortController();
    res.on('close', () => { if (!res.writableEnded) controller.abort(); });
    try {
      const chunks: Buffer[] = []; let size = 0;
      for await (const chunk of req) { size += chunk.length; if (size > 2 * 1024 * 1024) return reply(413, { ok: false, error: 'Worker request exceeds 2 MiB' }); chunks.push(Buffer.from(chunk)); }
      const input = JSON.parse(Buffer.concat(chunks).toString('utf8'));
      const data = await worker.call(match[1], input, controller.signal);
      if (!controller.signal.aborted) reply(200, { ok: true, data });
    } catch (error) { if (!controller.signal.aborted) reply(400, { ok: false, error: error instanceof Error ? error.message : String(error) }); }
  });
  server.requestTimeout = 0;
  return server;
}

export async function startLocalWorkerApi() {
  const token = randomBytes(32).toString('hex'), server = createWorkerServer(token);
  await new Promise<void>((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
  server.unref();
  const address = server.address();
  if (!address || typeof address === 'string') throw new Error('Worker API did not bind');
  return { url: `http://127.0.0.1:${address.port}`, token };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  loadThetaProjectEnvironment();
  const host = process.env.THETA_WORKER_API_HOST ?? '127.0.0.1';
  const port = Number(process.env.THETA_WORKER_API_PORT ?? 4319);
  createWorkerServer(process.env.THETA_WORKER_API_TOKEN ?? '').listen(port, host, () => console.log(`THETA worker API http://${host}:${port}`));
}
