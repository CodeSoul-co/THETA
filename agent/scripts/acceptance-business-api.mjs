// Explicit opt-in only. Live Business API acceptance: account, project and dataset hash.
// 只发送文档允许的公开请求；写操作必须显式 --confirm-live-writes；不打印凭据、Cookie 或 CSRF。
import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { loadThetaProjectEnvironment } from '../dist/src/environment.js';
import { BusinessApiComputeGateway, businessApiConfigurationFingerprint } from '../dist/src/adapters/business-api-gateway.js';
import { ResearchStore } from '../dist/src/memory/research-store.js';
import { writeSupportFixture } from './support-fixture.mjs';

loadThetaProjectEnvironment();
const baseUrl = process.env.THETA_BUSINESS_API_URL;
if (!baseUrl) throw new Error('未配置 THETA_BUSINESS_API_URL；本脚本只在 Business API 模式下运行。');
if (!process.env.THETA_BUSINESS_IDENTIFIER || !process.env.THETA_BUSINESS_PASSWORD) throw new Error('缺少 THETA_BUSINESS_IDENTIFIER / THETA_BUSINESS_PASSWORD：写入本机私有配置 agent/.env.local，不要提交。');
if (!process.env.THETA_BUSINESS_BINDINGS) throw new Error('缺少 THETA_BUSINESS_BINDINGS：需要宿主提供的项目与 model_id/runtime_id 映射。');

const writes = process.argv.includes('--confirm-live-writes');
const home = path.resolve(process.argv.slice(2).find(argument => !argument.startsWith('--')) ?? '.theta_agent/acceptance/business');
mkdirSync(home, { recursive: true });
const gateway = new BusinessApiComputeGateway(new ResearchStore(home), fetch, { baseUrl });
const report = { checkedAt: new Date().toISOString(), baseUrl, configurationFingerprint: businessApiConfigurationFingerprint({ baseUrl }), liveWrites: writes, steps: [] };
const record = (step, value) => { report.steps.push({ step, ...value }); console.log(JSON.stringify({ step, ...value })); };

const account = await gateway.authenticate();
record('auth/me', { ok: true, user: account.user, csrfAvailable: account.csrfAvailable, expiresAt: account.expiresAt });

if (!writes) {
  record('project', { ok: false, skipped: '需要 --confirm-live-writes：可能创建项目或上传数据。' });
  record('dataset', { ok: false, skipped: '需要 --confirm-live-writes。' });
} else {
  const projectId = await gateway.ensureProject();
  record('project', { ok: true, projectId });
  const fixture = path.join(home, 'fixtures', 'business-acceptance.csv');
  writeSupportFixture(fixture);
  const bytes = readFileSync(fixture);
  const sha256 = createHash('sha256').update(bytes).digest('hex');
  const datasetRef = await gateway.ensureDataset({ datasetRef: 'business-acceptance', sha256, fileName: path.basename(fixture), managedPath: fixture, sizeBytes: bytes.length });
  record('dataset', { ok: true, datasetRef, sha256, sizeBytes: bytes.length, remoteHashVerified: true });
}

// 如实报告“现有方案原样提交”在公开参数契约下是否可行；不修改任何参数来制造成功。
const sample = {
  jobId: 'acceptance-plan', runId: 'acceptance',
  dataset: { datasetRef: 'business-acceptance', sha256: '', fileName: 'business-acceptance.csv', managedPath: path.join(home, 'fixtures', 'business-acceptance.csv'), sizeBytes: 0 },
  plan: { modelId: 'lda', textColumn: 'text', device: 'cpu', params: { num_topics: 5, max_iter: 3, 'prepare.clean': false, lang: 'en' }, rationale: '最近一次本地 LDA 方案，原样提交核对', timeoutSeconds: 43200 },
};
try {
  const params = gateway.representableParams(sample);
  record('current-plan', { ok: true, params, note: '公开参数契约可以表达该方案；仍需真实 model_id 才能提交。' });
} catch (error) {
  record('current-plan', { ok: false, reason: error instanceof Error ? error.message : String(error) });
}

const bindings = JSON.parse(readFileSync(process.env.THETA_BUSINESS_BINDINGS, 'utf8'));
record('bindings', { ok: true, models: Object.keys(bindings.models ?? {}), projectId: bindings.projectId ?? null, projectTitle: bindings.projectTitle ?? null, allowProjectCreate: bindings.allowProjectCreate === true });
console.log(JSON.stringify({ ...report, scope: '只核验公开 Business API 的账号、项目与数据集 hash；未提交训练、未读取结果、未改动算法或参数。' }, null, 2));
