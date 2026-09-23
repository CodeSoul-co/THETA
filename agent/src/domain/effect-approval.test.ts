import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdtempSync, rmSync } from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { EffectApprovals } from './effect-approval.js';
import { ResearchStore } from '../memory/research-store.js';
import { LocalComputeGateway } from '../adapters/compute-gateway.js';

test('effect approvals bind target, payload, session and expiry; gateway cannot bypass host consent', async () => {
  const home = mkdtempSync(path.join(os.tmpdir(), 'theta-consent-'));
  try {
    const store = new ResearchStore(home); const approvals = new EffectApprovals(store);
    const payload = { operation: 'embedding', dataHash: 'data', maxRequests: 4 };
    const request = approvals.request('session', { action: 'external.api', target: 'https://fixture', payload, summary: 'fixture request' });
    assert.throws(() => approvals.assert({ id: request.id, hash: request.hash }, request.action, request.target, payload), /变化/);
    assert.throws(() => approvals.decide(request.id, 'wrong-session', request.hash, true), /不匹配/);
    const receipt = approvals.decide(request.id, 'session', request.hash, true);
    approvals.assert(receipt, request.action, request.target, payload);
    assert.throws(() => approvals.assert(receipt, request.action, request.target, { ...payload, maxRequests: 5 }), /变化/);
    assert.throws(() => approvals.assert(receipt, request.action, 'https://another', payload), /变化/);
    store.put('effect-approval', request.id, { ...approvals.get(request.id), expiresAt: 0 });
    assert.throws(() => approvals.assert(receipt, request.action, request.target, payload), /变化/);
    let calls = 0;
    const gateway = new LocalComputeGateway(home, { call: async <T>() => { calls++; return {} as T; } });
    await assert.rejects(async () => gateway.submit({ jobId: 'job', runId: 'run', dataset: {} as never, plan: {} as never }), /确认/);
    assert.equal(calls, 0);
  } finally { rmSync(home, { recursive: true, force: true }); }
});
