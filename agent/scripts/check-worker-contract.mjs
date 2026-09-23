// Read-only worker interface check. Never starts training or writes user data.
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, writeFileSync, rmSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadThetaProjectEnvironment, repositoryRoot } from '../dist/src/environment.js';
import { PythonCapabilityWorker, pythonExecutable } from '../dist/src/adapters/python-worker.js';

loadThetaProjectEnvironment();
const root = repositoryRoot();
const agentRoot = fileURLToPath(new URL('../', import.meta.url));
const worker = new PythonCapabilityWorker();
const checks = [];
const expect = (condition, message) => { if (!condition) throw new Error(message); };
const hasKeys = (value, keys, label) => { for (const key of keys) expect(value && typeof value === 'object' && key in value, label + ' 缺少字段 ' + key); };
const check = async (name, run) => {
  try { checks.push({ name, ok: true, detail: await run() }); }
  catch (error) { checks.push({ name, ok: false, detail: error instanceof Error ? error.message : String(error) }); }
};
const MODEL_IDS = ['etm', 'nvdm', 'gsm', 'prodlda', 'lda', 'btm', 'hdp', 'dtm', 'stm', 'ctm', 'bertopic', 'theta'];
const PROFILES = { classic: ['lda', 'btm', 'hdp', 'dtm', 'stm'], neural: ['etm', 'nvdm', 'gsm', 'prodlda', 'ctm', 'bertopic', 'theta'], reports: [], statistics: [] };
const fixture = path.join(root, 'trainning', 'testdata', 'lda_smoke.csv');
const home = mkdtempSync(path.join(os.tmpdir(), 'theta-contract-'));
const plan = { modelId: 'lda', textColumn: 'text', params: { num_topics: 3, max_iter: 2 }, rationale: 'contract check', timeoutSeconds: 180 };

await check('models.list', async () => {
  const models = await worker.call('models.list', {});
  expect(Array.isArray(models), 'models.list 必须返回数组');
  expect(models.length === MODEL_IDS.length, '模型数量变化：' + models.length);
  for (const model of models) hasKeys(model, ['modelId', 'description'], 'models.list 条目');
  expect(MODEL_IDS.every((id) => models.some((model) => model.modelId === id)), '模型清单缺少已有模型');
  return models.map((model) => model.modelId).join(',');
});
await check('models.inspect', async () => {
  const lda = await worker.call('models.inspect', { modelId: 'lda' });
  hasKeys(lda, ['modelId', 'description', 'runtimeProfile', 'requiresTime', 'requiresCovariates', 'requiresWeights', 'embeddingOptions', 'supportedParameters', 'parameters', 'parameterChoices', 'engine'], 'models.inspect');
  expect(lda.runtimeProfile === 'classic', 'lda 必须路由到 classic');
  expect((await worker.call('models.inspect', { modelId: 'dtm' })).requiresTime === true, 'dtm 必须声明 requiresTime');
  expect((await worker.call('models.inspect', { modelId: 'stm' })).requiresCovariates === true, 'stm 必须声明 requiresCovariates');
  return Object.keys(lda).length + ' 个字段';
});
await check('runtime.environments', async () => {
  const profiles = await worker.call('runtime.environments', {});
  expect(Array.isArray(profiles) && profiles.length === 4, '必须返回 4 个 profile');
  for (const entry of profiles) {
    hasKeys(entry, ['profile', 'revision', 'python', 'mode', 'available', 'configurationVariable', 'models'], 'runtime.environments');
    expect(entry.configurationVariable === 'THETA_WORKER_' + entry.profile.toUpperCase() + '_PYTHON', '解释器变量名变化：' + entry.configurationVariable);
    expect(JSON.stringify(entry.models) === JSON.stringify(PROFILES[entry.profile]), entry.profile + ' 的模型归属变化');
  }
  return profiles.map((entry) => entry.profile + '=' + (entry.available ? 'ready' : 'missing')).join(' ');
});
await check('runtime.environment', async () => {
  const identity = await worker.call('runtime.environment', { profile: 'classic' });
  hasKeys(identity, ['profile', 'revision', 'python', 'pythonVersion', 'prefix', 'dependencyFingerprint', 'isolatedVenv', 'inheritsSystemPackages', 'sandbox'], 'runtime.environment');
  expect(/^[0-9a-f]{64}$/u.test(identity.dependencyFingerprint), '依赖指纹必须是 sha256');
  return identity.python + ' ' + identity.dependencyFingerprint.slice(0, 12);
});
await check('runtime.config', async () => {
  const configuration = await worker.call('runtime.config', {});
  hasKeys(configuration, ['embedding'], 'runtime.config');
  hasKeys(configuration.embedding, ['preferredMode', 'configured'], 'runtime.config.embedding');
  return configuration.embedding.preferredMode;
});
let dataset;
await check('dataset.import', async () => {
  dataset = await worker.call('dataset.import', { filePath: fixture, uploadDir: path.join(home, 'uploads') });
  hasKeys(dataset, ['datasetRef', 'sha256', 'fileName', 'managedPath', 'sizeBytes'], 'dataset.import');
  const expected = createHash('sha256').update(readFileSync(fixture)).digest('hex');
  expect(dataset.sha256 === expected, '托管副本 hash 与源文件不一致');
  return dataset.fileName + ' ' + dataset.sha256.slice(0, 12);
});
await check('dataset.profile', async () => {
  const profile = await worker.call('dataset.profile', { dataset });
  expect(Array.isArray(profile.columns) && profile.columns.includes('text'), 'dataset.profile 必须报告列清单');
  hasKeys(profile, ['candidateRoles'], 'dataset.profile');
  return profile.columns.join(',');
});
await check('plan.validate', async () => {
  await worker.call('plan.validate', { dataset, plan });
  let rejected = false;
  try { await worker.call('plan.validate', { dataset, plan: { ...plan, params: { num_topics: -1 } } }); } catch { rejected = true; }
  expect(rejected, '非法参数必须被拒绝');
  return '合法接受 / 非法拒绝';
});
await check('compute.preview', async () => {
  const preview = await worker.call('compute.preview', { plan, dataset });
  hasKeys(preview, ['execution', 'readiness', 'rowCount'], 'compute.preview');
  hasKeys(preview.execution, ['embedding', 'runtime'], 'compute.preview.execution');
  hasKeys(preview.execution.embedding, ['mode'], 'compute.preview.execution.embedding');
  hasKeys(preview.execution.runtime, ['profile', 'revision', 'python', 'dependencyFingerprint'], 'compute.preview.execution.runtime');
  expect(preview.rowCount > 0, 'rowCount 必须为正');
  return 'rows=' + preview.rowCount + ' runtime=' + preview.execution.runtime.profile;
});
await check('ExecutionSpec v2 与规范化列契约', async () => {
  const program = path.join(home, 'contract-check.py');
  writeFileSync(program, "import json, sys, tempfile\nfrom pathlib import Path\n\nroot, payload_json, agent_root = sys.argv[1], sys.argv[2], sys.argv[3]\nsys.path.insert(0, str(Path(root) / \"trainning\"))\nsys.path.insert(0, agent_root)\n\nimport statistics\nfrom worker.protocol import ExecutionSpec\n\npayload = json.loads(payload_json)\nspec = ExecutionSpec.from_dict(payload)\nresult = {\n    \"identity\": spec.identity,\n    \"roundTripExecutor\": spec.runtime.executor,\n    \"objectKey\": spec.dataset.object_key,\n    \"params\": spec.params,\n    \"stdlibStatistics\": hasattr(statistics, \"NormalDist\"),\n}\nmutations = {\n    \"Executor\": {\"runtime\": {**payload[\"runtime\"], \"executor\": \"other\"}},\n    \"Schema\": {\"schema_version\": 1},\n    \"Filename\": {\"dataset\": {**payload[\"dataset\"], \"filename\": \"../escape.csv\"}},\n}\nfor name, change in mutations.items():\n    try:\n        ExecutionSpec.from_dict({**payload, **change})\n        result[\"reject\" + name] = \"accepted\"\n    except Exception:\n        result[\"reject\" + name] = \"rejected\"\n\nfrom workers.capabilities import dataset_import\nfrom workers.local_compute import normalize_dataset\n\nhome = Path(tempfile.mkdtemp(prefix=\"theta-contract-\"))\nsource = home / \"records.jsonl\"\nsource.write_text(\"\\n\".join(json.dumps({\"内容\": \"退款 账单 重复扣费 客服\", \"渠道\": \"web\"}, ensure_ascii=False) for _ in range(3)), encoding=\"utf-8\")\ndataset = dataset_import({\"filePath\": str(source), \"uploadDir\": str(home / \"uploads\")})\ndestination = home / \"normalized.csv\"\nnormalize_dataset({\"dataset\": dataset, \"plan\": {\"modelId\": \"stm\", \"textColumn\": \"内容\", \"covariates\": [\"渠道\"], \"params\": {}}}, destination)\nlines = [line for line in destination.read_text(encoding=\"utf-8\").splitlines() if line]\nresult[\"normalizedHeader\"] = lines[0]\nresult[\"normalizedRows\"] = len(lines) - 1\nprint(json.dumps(result, ensure_ascii=False))");
  const payload = {
    schema_version: 2, type: 'job.ready', event_id: 'contract-check', task_id: 1, attempt: 1, task_version: 1,
    user_id: 'contract-check', priority: 0,
    dataset: { ref: 'contract-check', project_id: 'contract-check', object_key: 'dataset/data.csv', filename: 'data.csv', format: 'csv', sha256: 'a'.repeat(64) },
    model: { id: 1, name: 'lda', framework: 'theta' },
    runtime: { id: 1, key: 'classic-cpu', version: 'v1', executor: 'theta_pipeline', image: 'local' },
    params: { num_topics: 3 },
    resources: { accelerator: 'none', cpu_cores: 2, memory_mb: 4096, gpu_count: 0, gpu_memory_mb: 0, timeout_seconds: 180 },
    output_prefix: 'job-contract/', created_at: '2026-01-01T00:00:00Z',
  };
  const result = spawnSync(pythonExecutable(), [program, root, JSON.stringify(payload), agentRoot], { encoding: 'utf8', cwd: agentRoot });
  expect(result.status === 0, '契约脚本失败：' + ((result.stderr || result.stdout || '') + '').slice(-300));
  const parsed = JSON.parse(result.stdout.trim().split('\n').pop());
  expect(parsed.roundTripExecutor === 'theta_pipeline', 'executor 回读不一致：' + parsed.roundTripExecutor);
  expect(parsed.identity === '1:1', 'identity 形状变化：' + parsed.identity);
  expect(parsed.objectKey === 'dataset/data.csv', 'object_key 归一化变化');
  expect(parsed.rejectExecutor === 'rejected', '非 theta_pipeline 执行器必须被拒绝');
  expect(parsed.rejectSchema === 'rejected', 'schema_version 必须严格校验');
  expect(parsed.rejectFilename === 'rejected', 'dataset.filename 带路径必须被拒绝');
  expect(parsed.stdlibStatistics === true, 'worker 侧标准库 statistics 被本地包遮蔽');
  expect(parsed.normalizedHeader === 'text,cov_0', '规范化列契约变化：' + parsed.normalizedHeader);
  expect(parsed.normalizedRows === 3, '规范化行数不一致：' + parsed.normalizedRows);
  return 'spec v2 + ' + parsed.normalizedHeader;
});

rmSync(home, { recursive: true, force: true });
const failed = checks.filter((entry) => !entry.ok);
console.log(JSON.stringify({ python: pythonExecutable(), ok: failed.length === 0, checks }, null, 2));
if (failed.length) process.exitCode = 1;
