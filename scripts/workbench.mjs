#!/usr/bin/env node
import { spawn, execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync, openSync, closeSync, unlinkSync } from 'node:fs';
import { createServer } from 'node:net';
import { loadEnvFile } from 'node:process';
import { randomUUID } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const script = fileURLToPath(import.meta.url);
export const root = path.resolve(path.dirname(script), '..');
const runtime = path.join(root, '.local/workbench');
const stateFile = path.join(runtime, 'services.json');
const lockFile = path.join(runtime, 'operation.lock');
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const readJson = file => { try { return JSON.parse(readFileSync(file, 'utf8')); } catch { return undefined; } };
const alive = pid => { try { process.kill(pid, 0); return true; } catch { return false; } };

export function serviceEnvironment(repository, inherited) {
  return { ...inherited,
    NODE_ENV: 'development', THETA_PROJECT_ROOT: repository,
    THETA_AGENT_HOME: path.join(repository, '.theta_agent'),
    THETA_AGENT_SERVICE_MODE: 'local', THETA_AGENT_API_HOST: '127.0.0.1',
    THETA_AGENT_API_PORT: '4318', THETA_MANUAL_PORT: '4321',
    THETA_AGENT_API_URL: 'http://127.0.0.1:4318',
    THETA_MANUAL_LOCAL_API_URL: 'http://127.0.0.1:4321',
    THETA_LOCAL_AUTH_ENABLED: 'true', NEXT_PUBLIC_LOCAL_NO_AUTH: 'false',
    THETA_COMPUTE_URL: '', THETA_BUSINESS_API_URL: '',
    THETA_PYTHON: inherited.THETA_PYTHON || path.join(repository, 'agent/.venv/bin/python'),
    PYTHONNOUSERSITE: '1',
  };
}

const services = [
  { name: '对话 API', key: 'agent', port: 4318, cwd: 'agent', args: ['--disable-warning=ExperimentalWarning', 'dist/web/server.js'], health: '/api/v3/health' },
  { name: '手动 API', key: 'manual', port: 4321, cwd: 'agent', args: ['--disable-warning=ExperimentalWarning', 'dist/web/manual-server.js'], health: '/health' },
  { name: '前端', key: 'frontend', port: 4320, cwd: 'frontend', args: ['node_modules/next/dist/bin/next', 'dev', '-H', '127.0.0.1', '-p', '4320'], health: '/workbench?mode=conversation' },
];

function configuredEnvironment() {
  // Match Agent precedence, preserving private provider/model/runtime settings.
  const files = process.env.THETA_ENV_FILE ? [path.resolve(process.env.THETA_ENV_FILE)]
    : ['agent/.env.local', 'agent/.env', '.env.local', '.env'].map(file => path.join(root, file));
  for (const file of files) if (existsSync(file)) loadEnvFile(file);
  if (process.env.THETA_ENV_FILE) process.env.THETA_ENV_FILE = files[0];
  return serviceEnvironment(root, process.env);
}

export async function portAvailable(port) {
  return new Promise(resolve => {
    const server = createServer();
    server.once('error', () => resolve(false));
    server.listen(port, '127.0.0.1', () => server.close(() => resolve(true)));
  });
}

function owned(state) {
  if (!state || state.root !== root || !Number.isInteger(state.pid) || state.pid < 2 || !state.token) return false;
  try {
    const command = execFileSync('ps', ['-p', String(state.pid), '-o', 'command='], { encoding: 'utf8' });
    return command.includes(`${script} supervise ${state.token}`);
  } catch { return false; }
}

async function health(service) {
  try {
    const response = await fetch(`http://127.0.0.1:${service.port}${service.health}`, { signal: AbortSignal.timeout(5000) });
    if (!response.ok) return false;
    if (service.key === 'frontend') { await response.body?.cancel(); return true; }
    const body = await response.json();
    return service.key === 'agent' ? body.ok === true && body.data?.service === 'theta-agent' : body.service === 'theta-manual';
  } catch { return false; }
}

async function status() {
  const state = readJson(stateFile);
  const managed = owned(state);
  console.log(`后台管理：${managed ? `运行中（PID ${state.pid}）` : '未运行（已有端口服务不会被自动接管）'}`);
  const checks = await Promise.all(services.map(health));
  services.forEach((service, i) => console.log(`${service.name} · ${service.port} · ${checks[i] ? '可访问' : '不可访问'}`));
  console.log(`页面：http://127.0.0.1:4320/workbench?mode=conversation\n日志：${runtime}\n对话数据：${path.join(root, '.theta_agent')}\n手动数据：${path.join(root, '.local/manual-workbench')}`);
  return managed && checks.every(Boolean);
}

async function prepare(env) {
  const [major, minor] = process.versions.node.split('.').map(Number);
  if (major < 22 || major === 22 && minor < 13) throw new Error('需要 Node.js ≥ 22.13。');
  for (const file of ['agent/node_modules/typescript/bin/tsc', 'frontend/node_modules/next/dist/bin/next']) {
    if (!existsSync(path.join(root, file))) throw new Error(`缺少 ${file}，请先按 frontend/README.md 安装依赖。`);
  }
  if (!existsSync(env.THETA_PYTHON)) throw new Error(`Python 不存在：${env.THETA_PYTHON}；请准备 agent/.venv 或设置 THETA_PYTHON 绝对路径。`);
  console.log('编译当前 Agent 源码，并检查本仓库 Worker 契约（不训练、不请求外部模型）…');
  await run(process.execPath, ['node_modules/typescript/bin/tsc', '-p', 'tsconfig.json'], path.join(root, 'agent'), env);
  await run(process.execPath, ['--disable-warning=ExperimentalWarning', 'scripts/check-worker-contract.mjs'], path.join(root, 'agent'), env);
}

function run(command, args, cwd, env) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { cwd, env, stdio: 'inherit' });
    child.once('error', reject);
    child.once('exit', code => code === 0 ? resolve() : reject(new Error(`预检失败，退出码 ${code}；现有服务未停止。`)));
  });
}

async function stop() {
  const state = readJson(stateFile);
  if (!owned(state)) { console.log('没有本入口管理的服务；不按端口杀进程。'); return; }
  process.kill(state.pid, 'SIGTERM');
  for (let i = 0; i < 100 && owned(state); i++) await sleep(200);
  if (owned(state)) throw new Error('服务仍在退出，请查看 supervisor.log；没有强制终止训练。');
  console.log('网页与 API 已停止；未删除项目、上传文件、模型或结果。后台训练不在停止范围内。');
}

async function start(env, prepared = false) {
  if (owned(readJson(stateFile))) {
    if (!await status()) throw new Error('管理进程存在但服务不健康，请检查日志后使用 restart。');
    return;
  }
  for (const service of services) {
    if (!await portAvailable(service.port)) throw new Error(`${service.port} 已被其他进程占用。请先核对并停止旧服务；此入口不会杀死或接管未知进程。`);
  }
  if (!prepared) await prepare(env);
  const token = randomUUID();
  const log = openSync(path.join(runtime, 'supervisor.log'), 'a');
  const child = spawn(process.execPath, [script, 'supervise', token], { cwd: root, env, detached: true, stdio: ['ignore', log, log] });
  const spawnError = new Promise((_, reject) => child.once('error', reject));
  child.unref(); closeSync(log);
  await Promise.race([spawnError, (async () => {
    const deadline = Date.now() + 90_000;
    while (Date.now() < deadline) {
      const state = readJson(stateFile);
      if (state?.token === token && owned(state) && (await Promise.all(services.map(health))).every(Boolean)) {
        await status(); return;
      }
      if (!alive(child.pid)) throw new Error(`启动进程已退出，请检查 ${runtime}/supervisor.log。`);
      await sleep(500);
    }
    throw new Error(`启动就绪检查超时；服务未被强制终止。请运行 status 并检查 ${runtime} 中的日志。`);
  })()]);
}

function supervise(env, token) {
  const children = [];
  let stopping = false;
  const shutdown = () => {
    if (stopping) return;
    stopping = true;
    for (const child of children) if (child.exitCode === null && child.signalCode === null) child.kill('SIGTERM');
    const timer = setInterval(() => {
      if (children.every(child => child.exitCode !== null || child.signalCode !== null)) {
        clearInterval(timer);
        if (readJson(stateFile)?.token === token) unlinkSync(stateFile);
        process.exit(0);
      }
    }, 100);
  };
  process.on('SIGTERM', shutdown); process.on('SIGINT', shutdown);
  for (const service of services) {
    const log = openSync(path.join(runtime, `${service.key}.log`), 'a');
    const child = spawn(process.execPath, service.args, { cwd: path.join(root, service.cwd), env, stdio: ['ignore', log, log] });
    children.push(child); closeSync(log);
    child.once('error', error => { console.error(`${service.name} 启动失败：${error.message}`); shutdown(); });
    child.once('exit', (code, signal) => {
      console.log(`${new Date().toISOString()} ${service.name} 退出：${code ?? signal}`);
      if (!stopping) shutdown();
    });
  }
  writeFileSync(stateFile, JSON.stringify({ root, pid: process.pid, token, startedAt: new Date().toISOString(), children: children.map((child, index) => ({ name: services[index].name, pid: child.pid })) }, null, 2));
}

async function main() {
  const command = process.argv[2] ?? 'start';
  if (!['start', 'stop', 'restart', 'status', 'check', 'supervise'].includes(command)) {
    console.log('用法：./theta-web [start|status|restart|stop|check]\n本机开发专用。start 后台启动，check 只编译和检查，不启动训练。');
    process.exitCode = command === '--help' || command === 'help' ? 0 : 1; return;
  }
  mkdirSync(runtime, { recursive: true });
  if (command === 'status') { process.exitCode = await status() ? 0 : 1; return; }
  const env = configuredEnvironment();
  if (command === 'supervise') {
    if (!process.argv[3]) throw new Error('缺少后台启动标识。');
    supervise(env, process.argv[3]); return;
  }
  const lock = readJson(lockFile);
  if (lock?.pid && !alive(lock.pid)) unlinkSync(lockFile);
  let fd;
  try { fd = openSync(lockFile, 'wx'); }
  catch { throw new Error('另一条启动/停止命令正在执行，请稍后重试。'); }
  writeFileSync(fd, JSON.stringify({ pid: process.pid })); closeSync(fd);
  try {
    if (command === 'check') await prepare(env);
    else if (command === 'stop') await stop();
    else if (command === 'restart') { await prepare(env); await stop(); await start(env, true); }
    else await start(env);
  } finally { unlinkSync(lockFile); }
}

if (process.argv[1] && path.resolve(process.argv[1]) === script) {
  main().catch(error => { console.error(`启动管理失败：${error.message}`); process.exitCode = 1; });
}
