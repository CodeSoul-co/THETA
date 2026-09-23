import assert from 'node:assert/strict';
import test from 'node:test';
import { createHash } from 'node:crypto';
import { mkdtempSync, writeFileSync, rmSync, mkdirSync, readFileSync, symlinkSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { GoComputeGateway, goConfigurationFingerprint } from './go-compute-gateway.js';
import { EffectApprovals } from '../domain/effect-approval.js';
import { ResearchStore } from '../memory/research-store.js';
import { c as createTar } from 'tar';

test('Go gateway binds host-owned dataset/model mappings, restores task IDs and fences uncertain submissions', async () => {
  const home = mkdtempSync(path.join(os.tmpdir(), 'theta-go-'));
  const previous = process.env.THETA_COMPUTE_BINDINGS;
  const bindings = path.join(home, 'bindings.json');
  process.env.THETA_COMPUTE_BINDINGS = bindings;
  writeFileSync(bindings, JSON.stringify({ userId: 'user', projectId: 'project', datasets: { dataset: { ref: 'registered', sha256: 'sha', textColumn: 'text', timeColumn: 'year', covariates: ['channel'] } }, models: { lda: { modelId: 7 }, theta: { modelId: 8 } } }));
  const store = new ResearchStore(home);
  let posts = 0;
  const fetcher: typeof fetch = async (_url, init) => {
    if (init?.method === 'POST') {
      posts++; const body = JSON.parse(String(init.body)); assert.equal(body.dataset_ref, 'registered');
      assert.equal(body.params['input.text_column'], 'text');
      assert.equal(body.params['input.time_column'], 'year');
      assert.deepEqual(body.params['input.covariates'], ['channel']);
      assert.equal(body.params['model.random_state'], 7);
      if (body.model_id === 8) assert.equal(body.params.embedding_provider, 'local');
    }
    return Response.json({ id: 42, status: 'running', phase: 'training', progress: 60 });
  };
  const request = { jobId: 'job-one', runId: 'run', execution: { computeConfigurationFingerprint: goConfigurationFingerprint() }, dataset: { datasetRef: 'dataset', sha256: 'sha', fileName: 'data.csv', managedPath: '/local', sizeBytes: 10 }, plan: { modelId: 'lda', textColumn: 'text', timeColumn: 'year', covariates: ['channel'], params: { 'model.random_state': 7 }, rationale: 'baseline', timeoutSeconds: 30 } };
  try {
    const approvals = new EffectApprovals(store);
    const permit = (payload: typeof request) => { const value = approvals.request('session', { action: 'compute.submit', target: 'http://127.0.0.1:8080', payload, summary: 'test task' }); return approvals.decide(value.id, 'session', value.hash, true); };
    const receipt = permit(request);
    const gateway = new GoComputeGateway('http://127.0.0.1:8080', store, fetcher);
    await assert.rejects(gateway.submit(request), /确认/); assert.equal(posts, 0);
    assert.equal((await gateway.submit(request, receipt)).percent, 60);
    await new GoComputeGateway('http://127.0.0.1:8080', store, fetcher).submit(request, receipt);
    assert.equal(posts, 1);
    const theta = { ...request, jobId: 'job-theta', plan: { ...request.plan, modelId: 'theta' } };
    await gateway.submit(theta, permit(theta));
    assert.equal(posts, 2);
    const unsafePath = { ...request, jobId: 'job-path', plan: { ...request.plan, params: { ...request.plan.params, 'embedding.model_path': '/host/private/model' } } };
    await assert.rejects(gateway.submit(unsafePath, permit(unsafePath)), /本地 embedding 路径/);
    await assert.rejects(gateway.submit({ ...request, plan: { ...request.plan, params: { num_topics: 2 } } }, receipt), /变化/);
    const failing = new GoComputeGateway('http://127.0.0.1:8080', store, async () => { throw new Error('network interrupted'); });
    const uncertain = { ...request, jobId: 'job-uncertain' }; const uncertainReceipt = permit(uncertain);
    await assert.rejects(failing.submit(uncertain, uncertainReceipt), /network/);
    await assert.rejects(failing.submit(uncertain, uncertainReceipt), /不确定/);
  } finally { if (previous === undefined) delete process.env.THETA_COMPUTE_BINDINGS; else process.env.THETA_COMPUTE_BINDINGS = previous; rmSync(home, { recursive: true, force: true }); }
});

test('Go gateway returns structured metrics, artifact object keys and the model download', async () => {
  const home = mkdtempSync(path.join(os.tmpdir(), 'theta-go-results-'));
  const store = new ResearchStore(home);
  store.put('remote-job', 'job-complete', { hash: 'request-hash', taskId: '42' });
  const fetcher: typeof fetch = async input => {
    const route = new URL(String(input)).pathname;
    if (route.endsWith('/result')) return Response.json({ model_weights_path: 'training-results/42/model.tar.gz', metrics: { NPMI: 0.21 }, output_files: { manifest: 'training-results/42/manifest.json' } });
    if (route.endsWith('/model/download')) return Response.json({ task_id: 42, download_url: 'https://storage.example/model.tar.gz' });
    return Response.json({ id: 42, status: 'completed', phase: 'completed', progress: 100 });
  };
  try {
    const result = await new GoComputeGateway('http://127.0.0.1:8080', store, fetcher).results('job-complete', 'summary') as { metrics: { NPMI: number }; outputFiles: { manifest: string }; download: { download_url: string } };
    assert.equal(result.metrics.NPMI, 0.21);
    assert.equal(result.outputFiles.manifest, 'training-results/42/manifest.json');
    assert.equal(result.download.download_url, 'https://storage.example/model.tar.gz');
  } finally { rmSync(home, { recursive: true, force: true }); }
});

test('Go gateway uploads an unbound managed dataset once before task submission', async () => {
  const home = mkdtempSync(path.join(os.tmpdir(), 'theta-go-upload-'));
  const previous = process.env.THETA_COMPUTE_BINDINGS;
  const bindings = path.join(home, 'bindings.json');
  const managed = path.join(home, 'OPC.xlsx');
  const bytes = Buffer.from('xlsx-fixture');
  const sha256 = createHash('sha256').update(bytes).digest('hex');
  writeFileSync(managed, bytes);
  process.env.THETA_COMPUTE_BINDINGS = bindings;
  writeFileSync(bindings, JSON.stringify({ userId: 'user', projectId: 'project', datasets: {}, models: { lda: { modelId: 7 } } }));
  const store = new ResearchStore(home);
  let uploads = 0; let tasks = 0;
  const fetcher: typeof fetch = async (input, init) => {
    const route = new URL(String(input)).pathname;
    if (route === '/api/v1/datasets') {
      uploads++;
      assert.equal(init?.method, 'POST');
      const form = init?.body as FormData;
      assert.equal(form.get('dataset_ref'), 'dataset-uploaded');
      assert.equal(form.get('sha256'), sha256);
      return Response.json({ dataset_ref: 'registered-upload', sha256 }, { status: 201 });
    }
    if (route === '/api/v1/tasks') {
      tasks++;
      const body = JSON.parse(String(init?.body));
      assert.equal(body.dataset_ref, 'registered-upload');
      return Response.json({ id: 84, status: 'queued', phase: 'scheduling', progress: 0 });
    }
    throw new Error(`unexpected route ${route}`);
  };
  const request = { jobId: 'job-upload', runId: 'run', execution: { computeConfigurationFingerprint: goConfigurationFingerprint() }, dataset: { datasetRef: 'dataset-uploaded', sha256, fileName: 'OPC.xlsx', managedPath: managed, sizeBytes: bytes.length }, plan: { modelId: 'lda', textColumn: '正文', params: { num_topics: 2 }, rationale: 'baseline', timeoutSeconds: 30 } };
  try {
    const approvals = new EffectApprovals(store);
    const pending = approvals.request('session', { action: 'compute.submit', target: 'http://127.0.0.1:8080', payload: request, summary: 'upload and train' });
    const receipt = approvals.decide(pending.id, 'session', pending.hash, true);
    const result = await new GoComputeGateway('http://127.0.0.1:8080', store, fetcher).submit(request, receipt);
    assert.equal(result.status, 'queued');
    assert.equal(uploads, 1);
    assert.equal(tasks, 1);
  } finally {
    if (previous === undefined) delete process.env.THETA_COMPUTE_BINDINGS; else process.env.THETA_COMPUTE_BINDINGS = previous;
    rmSync(home, { recursive: true, force: true });
  }
});

test('Go gateway verifies and safely extracts the Worker archive before building a row-aware report', async () => {
  const home = mkdtempSync(path.join(os.tmpdir(), 'theta-go-hydrate-'));
  const source = path.join(home, 'source.csv'); writeFileSync(source, 'text\nhello\n');
  const input = path.join(home, 'archive-input');
  for (const directory of ['result', 'workspace', 'input']) mkdirSync(path.join(input, directory), { recursive: true });
  writeFileSync(path.join(input, 'result', 'metrics.json'), '{}'); writeFileSync(path.join(input, 'workspace', 'vocab.json'), '[]');
  writeFileSync(path.join(input, 'input', 'data.csv'), 'text\nhello\n');
  const archivePath = path.join(home, 'model.tar.gz'); await createTar({ gzip: true, file: archivePath, cwd: input }, ['result', 'workspace', 'input']);
  const archive = readFileSync(archivePath); const sha256 = createHash('sha256').update(archive).digest('hex');
  const normalized = createHash('sha256').update('text\nhello\n').digest('hex');
  const request = { jobId: 'job-complete', runId: 'run', dataset: { datasetRef: 'dataset', sha256: 'source-hash', fileName: 'source.csv', managedPath: source, sizeBytes: 11 }, plan: { modelId: 'lda', textColumn: 'text', params: { num_topics: 2 }, rationale: 'test', timeoutSeconds: 30 } };
  const store = new ResearchStore(home); store.put('remote-job', request.jobId, { hash: 'request-hash', taskId: '42', request });
  let workerCalls = 0;
  const worker = { call: async <T>(operation: string, payload: unknown) => { workerCalls++; assert.equal(operation, 'compute.remote_report'); assert.match(String((payload as {bundleRoot:string}).bundleRoot), /remote-results/); return { schemaVersion: 'theta.result-report.v2', jobId: request.jobId, resultHash: 'result-hash', reportStatus: 'complete', reportPath: '/report.html', resultDir: '/result', files: [], evidence: { evidence: [], tables: [], figures: [], matrices: [] }, trainingPlan: request.plan, missingEvidence: [] } as T; } };
  const fetcher: typeof fetch = async input => {
    const url = String(input);
    if (url === 'https://storage.example/model.tar.gz') return new Response(archive, { headers: { 'content-length': String(archive.length) } });
    const route = new URL(url).pathname;
    if (route.endsWith('/result')) return Response.json({ output_files: { integrity: { archive_sha256: sha256, archive_size_bytes: archive.length, normalized_input_sha256: normalized } } });
    if (route.endsWith('/model/download')) return Response.json({ download_url: 'https://storage.example/model.tar.gz', sha256, size_bytes: archive.length });
    return Response.json({ id: 42, status: 'completed', phase: 'completed', progress: 100 });
  };
  try {
    const gateway = new GoComputeGateway('http://127.0.0.1:8080', store, fetcher, home, worker);
    const report = await gateway.results(request.jobId, 'report') as { schemaVersion: string };
    assert.equal(report.schemaVersion, 'theta.result-report.v2'); assert.equal(workerCalls, 1);
    await gateway.results(request.jobId, 'summary'); assert.equal(workerCalls, 1, 'verified report is cached');
  } finally { rmSync(home, { recursive: true, force: true }); }
});

test('Go gateway rejects links in a signed archive before invoking the report worker', async () => {
  const home = mkdtempSync(path.join(os.tmpdir(), 'theta-go-unsafe-')); const input = path.join(home, 'input');
  for (const directory of ['result','workspace','input']) mkdirSync(path.join(input,directory),{recursive:true});
  writeFileSync(path.join(input,'input','data.csv'),'text\nhello\n'); symlinkSync('/etc/passwd',path.join(input,'result','escape'));
  const file = path.join(home,'unsafe.tar.gz'); await createTar({gzip:true,file,cwd:input},['result','workspace','input']);
  const archive=readFileSync(file); const sha256=createHash('sha256').update(archive).digest('hex'); const normalized=createHash('sha256').update('text\nhello\n').digest('hex');
  const request={jobId:'job-unsafe',runId:'run',dataset:{datasetRef:'d',sha256:'s',fileName:'d.csv',managedPath:'/d',sizeBytes:1},plan:{modelId:'lda',textColumn:'text',params:{},rationale:'',timeoutSeconds:30}};
  const store=new ResearchStore(home);store.put('remote-job',request.jobId,{hash:'h',taskId:'42',request});let calls=0;
  const fetcher:typeof fetch=async input=>{const url=String(input);if(url==='https://storage.example/unsafe.tar.gz')return new Response(archive,{headers:{'content-length':String(archive.length)}});const route=new URL(url).pathname;if(route.endsWith('/result'))return Response.json({output_files:{integrity:{archive_sha256:sha256,archive_size_bytes:archive.length,normalized_input_sha256:normalized}}});if(route.endsWith('/model/download'))return Response.json({download_url:'https://storage.example/unsafe.tar.gz',sha256,size_bytes:archive.length});return Response.json({id:42,status:'completed',phase:'completed',progress:100});};
  const worker={call:async<T>()=>{calls++;return{} as T;}};
  try {await assert.rejects(new GoComputeGateway('http://127.0.0.1:8080',store,fetcher,home,worker).results(request.jobId,'report'),/不安全/);assert.equal(calls,0);}
  finally{rmSync(home,{recursive:true,force:true});}
});
