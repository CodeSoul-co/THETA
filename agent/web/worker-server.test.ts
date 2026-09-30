import test from 'node:test';
import assert from 'node:assert/strict';
import { createWorkerServer } from './worker-server.js';
import { PythonCapabilityWorker } from '../src/adapters/python-worker.js';

test('worker HTTP contract covers submit, progress, results, cancellation and rejects untrusted requests', async () => {
  const token = 'a'.repeat(64), calls: string[] = [];
  let cancelled = false;
  const server = createWorkerServer(token, { async call<T>(operation: string, input: unknown, signal?: AbortSignal): Promise<T> {
    calls.push(operation);
    if (operation === 'compute.preview') await new Promise<void>(resolve => { signal?.addEventListener('abort', () => { cancelled = true; resolve(); }, { once: true }); });
    return { operation, input } as T;
  } });
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve));
  const address = server.address() as {port:number}, base = `http://127.0.0.1:${address.port}`;
  const previous = { url: process.env.THETA_WORKER_API_URL, token: process.env.THETA_WORKER_API_TOKEN };
  process.env.THETA_WORKER_API_URL = base; process.env.THETA_WORKER_API_TOKEN = token;
  try {
    assert.equal((await fetch(base + '/health')).status, 403);
    assert.equal((await fetch(base + '/health', {headers:{authorization:`Bearer ${token}`,origin:'https://external.example'}})).status,403);
    assert.equal((await fetch(base + '/health', {headers:{authorization:`Bearer ${token}`}})).status,200);
    const client = new PythonCapabilityWorker();
    for (const operation of ['compute.submit','compute.status','compute.results','compute.cancel']) {
      const data = await client.call<{operation:string;input:unknown}>(operation, {jobId:'owned-job'});
      assert.equal(data.operation,operation); assert.deepEqual(data.input,{jobId:'owned-job'});
    }
    const controller = new AbortController();
    const pending=client.call('compute.preview',{},controller.signal);
    while (!calls.includes('compute.preview')) await new Promise(resolve=>setTimeout(resolve,10));
    controller.abort(); await assert.rejects(pending,/中断/);
    await new Promise(resolve=>setTimeout(resolve,50)); assert.equal(cancelled,true);
    process.env.THETA_WORKER_API_URL='http://external.example';
    await assert.rejects(client.call('compute.status',{}),/HTTPS/);
  } finally {
    for(const [key,value] of [['THETA_WORKER_API_URL',previous.url],['THETA_WORKER_API_TOKEN',previous.token]]) {if(value===undefined)delete process.env[key!];else process.env[key!]=value;}
    server.closeAllConnections();await new Promise<void>(resolve=>server.close(()=>resolve()));
  }
});
