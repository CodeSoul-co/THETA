const { app, BrowserWindow, Menu, dialog, ipcMain, utilityProcess, session, shell, safeStorage } = require('electron');
const { existsSync, mkdirSync, createWriteStream, readFileSync, writeFileSync } = require('node:fs');
const { spawn } = require('node:child_process');
const { randomBytes } = require('node:crypto');
const { createServer } = require('node:net');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { readSettings, saveInference, saveEmbedding, embeddingSettings } = require('./settings.cjs');
const smoke = process.argv.includes('--smoke-test');
app.setName('THETA');
const { prepareDataHome, selectDataHome, saveDataLocation, clearUpgradeCaches } = require('./data-home.cjs');
const locationFile = path.join(app.getPath('home'), '.theta-desktop-location.json');
const requestedDataHome = process.argv.find(argument => argument.startsWith('--data-dir='))?.slice('--data-dir='.length);
let home, settingsFile, startupError;
try {
  home = smoke ? path.resolve(process.env.THETA_DESKTOP_TEST_HOME || path.join(__dirname, '.smoke-home'))
    : selectDataHome({ requested: requestedDataHome, locationFile, legacyHome: path.join(app.getPath('appData'), 'THETA'), installDirectory: path.dirname(app.getPath('exe')) });
  home = prepareDataHome(home);
  app.setPath('userData', home);
  app.setPath('sessionData', home);
} catch (error) { startupError = error; }
const runtime = app.isPackaged ? path.join(process.resourcesPath, 'runtime') : path.join(__dirname, 'runtime');
const python = path.join(runtime, 'python', process.platform === 'win32' ? 'python.exe' : 'bin/python3');
let providerSpecs = [];
let window, service, origin, token, serviceEnv, stopped = false, restarting = false, quitting = false;
app.on('second-instance', () => { window?.restore(); window?.focus(); });

function pythonCall(args, options = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(python, args, { cwd: path.join(runtime, 'agent'), env: serviceEnv, windowsHide: true, ...options });
    let stdout = '', stderr = '';
    const timer = setTimeout(() => { child.kill(); reject(new Error('Python check timed out')); }, 120000);
    child.stdout.setEncoding('utf8'); child.stderr.setEncoding('utf8');
    child.stdout.on('data', data => { stdout += data; }); child.stderr.on('data', data => { stderr = (stderr + data).slice(-4000); });
    child.once('error', error => { clearTimeout(timer); reject(error); });
    child.once('exit', code => { clearTimeout(timer); code === 0 ? resolve(stdout) : reject(new Error(stderr || `Python exited ${code}`)); });
    child.stdin.end();
  });
}
async function selectPort(preferred) {
  const server = createServer();
  try {
    await new Promise((resolve, reject) => { server.once('error', reject); server.listen(preferred, '127.0.0.1', resolve); });
    const port = server.address().port;
    await new Promise(resolve => server.close(resolve)); return port;
  } catch (error) {
    if (preferred && error.code === 'EADDRINUSE') return selectPort(0);
    throw error;
  }
}
function environment(port) {
  const settings = readSettings(settingsFile);
  const env = { ...process.env };
  for (const key of Object.keys(env)) {
    if (/^(THETA_|PYTHON|OPENAI_|DEEPSEEK_|MINIMAX_|GLM_|EMBEDDING_|QWEN_|SBERT_)/.test(key)) delete env[key];
  }
  Object.assign(env, {
    NODE_ENV: 'production', NEXT_TELEMETRY_DISABLED: '1', PORT: String(port), HOSTNAME: '127.0.0.1',
    THETA_PROJECT_ROOT: runtime, THETA_DESKTOP_HOME: home, THETA_DESKTOP_TOKEN: token,
    THETA_AGENT_HOME: path.join(home, 'agent'), THETA_AGENT_SERVICE_MODE: 'local',
    THETA_ENV_FILE: path.join(home, '.desktop-no-env'), THETA_LOCAL_AUTH_ENABLED: 'true',
    THETA_PYTHON: python, THETA_WORKER_CONTROL_PYTHON: python,
    PYTHONNOUSERSITE: '1', PYTHONDONTWRITEBYTECODE: '1', PYTHONUTF8: '1',
    DATA_DIR: path.join(home, 'data'), RESULT_DIR: path.join(home, 'results'), WORKSPACE_DIR: path.join(home, 'workspace'),
    HF_HOME: path.join(home, 'cache/huggingface'), MPLCONFIGDIR: path.join(home, 'cache/matplotlib'),
    NUMBA_CACHE_DIR: path.join(home, 'cache/numba'), XDG_CACHE_HOME: path.join(home, 'cache'),
    TMPDIR: path.join(home, 'tmp'), TMP: path.join(home, 'tmp'), TEMP: path.join(home, 'tmp'),
    WORKER_JOB_ROOT: path.join(home, 'workspace', 'jobs'), OBJECT_STORAGE_FILESYSTEM_ROOT: path.join(home, 'workspace', 'objects'),
    // Desktop model downloads are explicit. Point settings at a complete local model.
    HF_HUB_OFFLINE: '1', TRANSFORMERS_OFFLINE: '1', TOKENIZERS_PARALLELISM: 'false',
    OMP_NUM_THREADS: '2', OPENBLAS_NUM_THREADS: '2',
  });
  for (const profile of ['CLASSIC', 'NEURAL', 'REPORTS', 'STATISTICS']) env[`THETA_WORKER_${profile}_PYTHON`] = python;
  const embedding = embeddingSettings(settings);
  if (embedding.localPath) {
    for (const key of ['QWEN_MODEL_PATH', 'QWEN_MODEL_0_6B', 'QWEN_MODEL_4B', 'QWEN_MODEL_8B']) env[key] = embedding.localPath;
  }
  if (embedding.sbertPath) env.SBERT_MODEL_PATH = embedding.sbertPath;
  Object.assign(env, { EMBEDDING_PROVIDER: embedding.mode, EMBEDDING_CLOUD_PROVIDER: embedding.provider,
    EMBEDDING_MODEL: embedding.model, EMBEDDING_API_BASE: embedding.baseUrl, EMBEDDING_API_KEY_ENV: 'EMBEDDING_API_KEY' });
  if (embedding.dimensions) env.EMBEDDING_DIMENSIONS = String(embedding.dimensions);
  if (embedding.encryptedKey) env.EMBEDDING_API_KEY = safeStorage.decryptString(Buffer.from(embedding.encryptedKey, 'base64'));
  const selected = settings.providers[settings.providerId];
  for (const spec of providerSpecs) {
    const id = spec.id === 'gpt' ? 'openai' : spec.id;
    const profile = settings.providers[id];
    if (id !== settings.providerId || !profile?.encryptedKey) continue;
    env[spec.envApiKey] = safeStorage.decryptString(Buffer.from(profile.encryptedKey, 'base64'));
    env[spec.envBaseUrl] = profile.baseUrl;
    env[spec.envModel] = profile.model;
  }
  if (selected?.encryptedKey) {
    env.THETA_INFERENCE_PROVIDER = settings.providerId;
    if (settings.providerId === 'openai-compatible') {
      env.THETA_INFERENCE_BASE_URL = selected.baseUrl;
      env.THETA_INFERENCE_MODEL = selected.model;
      env.THETA_INFERENCE_API_KEY = safeStorage.decryptString(Buffer.from(selected.encryptedKey, 'base64'));
    }
  }

  return env;
}
async function stopService() {
  const child = service; service = undefined;
  if (!child || !child.pid) return;
  await new Promise(resolve => {
    const timer = setTimeout(() => { child.kill(); resolve(); }, 5000);
    child.once('exit', () => { clearTimeout(timer); resolve(); });
    child.postMessage('shutdown');
  });
}
async function startService() {
  if (!existsSync(python) || !existsSync(path.join(runtime, 'frontend/server.js'))) throw new Error('安装包缺少运行环境。请运行 prepare:python 和 prepare:runtime 后重新打包。');
  let previousPort = 14320;
  try { previousPort = JSON.parse(readFileSync(path.join(home, 'port.json'))).port; } catch {}
  const port = await selectPort(smoke ? 0 : previousPort);
  writeFileSync(path.join(home, 'port.json'), JSON.stringify({ port }));
  token = randomBytes(32).toString('hex');
  serviceEnv = environment(port);
  const child = utilityProcess.fork(path.join(__dirname, 'services.mjs'), [], {
    cwd: runtime, env: serviceEnv, stdio: 'pipe', serviceName: 'THETA local services',
  });
  service = child;
  const log = createWriteStream(path.join(home, 'logs/services.log'), { flags: 'a', mode: 0o600 });
  child.stdout?.pipe(log, { end: false }); child.stderr?.pipe(log, { end: false });
  child.once('exit', () => log.end());
  return new Promise((resolve, reject) => {
    let ready = false;
    const timer = setTimeout(() => { child.kill(); reject(new Error('本地服务启动超时，请查看日志。')); }, 60000);
    child.once('exit', code => {
      clearTimeout(timer);
      if (!ready) reject(new Error(`本地服务退出 (${code})，请查看 ${path.join(home, 'logs/services.log')}`));
      else if (!quitting && !restarting && service === child) {
        dialog.showErrorBox('THETA 服务已停止', '请从应用菜单重新启动服务。数据保留在应用数据目录。');
      }
    });
    child.on('message', message => {
      if (message?.type !== 'ready') return;
      clearTimeout(timer); ready = true; origin = message.url; resolve(message);
    });
  });
}
async function restart() {
  restarting = true;
  try { await stopService(); await startService(); await window?.loadURL(origin + '/workbench?mode=conversation'); }
  finally { restarting = false; }
}
function openSettings() {
  window?.focus(); window?.webContents.send('desktop:open-settings');
}
async function applySettings() {
  serviceEnv = environment(Number(serviceEnv.PORT));
  const requestId = randomBytes(12).toString('hex');
  await new Promise((resolve, reject) => {
    const child = service;
    if (!child?.pid) { reject(new Error('本地服务未运行，请重新启动服务')); return; }
    const cleanup = () => { clearTimeout(timer); child.removeListener('message', listener); };
    const timer = setTimeout(() => { cleanup(); reject(new Error('应用配置超时，请重新启动服务')); }, 5000);
    const listener = message => { if (message?.type === 'configured' && message.requestId === requestId) { cleanup(); resolve(); } };
    child.on('message', listener);
    child.postMessage({ type: 'configure', requestId, env: serviceEnv });
  });
}
function registerSettings() {
  function trusted(event) {
    if (event.sender !== window?.webContents || event.senderFrame !== window.webContents.mainFrame || new URL(event.senderFrame.url).origin !== origin) throw new Error('Desktop workbench required');
  }
  ipcMain.handle('settings:read', event => {
    trusted(event); const value = readSettings(settingsFile);
    const { encryptedKey, ...embedding } = embeddingSettings(value);
    return { embedding: { ...embedding, apiKeyConfigured: Boolean(encryptedKey) }, home, python: '3.12.13' };
  });
  ipcMain.handle('settings:catalog', event => {
    trusted(event); const value = readSettings(settingsFile);
    const specs = [...providerSpecs, { id: 'openai-compatible', displayName: '自定义 / 本地服务', defaultBaseUrl: 'http://127.0.0.1:11434/v1', defaultModel: '', defaultModels: [] }];
    return { kind: 'inference.provider.list', selection: value.providers[value.providerId]?.encryptedKey ? { providerId: value.providerId, model: value.providers[value.providerId].model, source: 'desktop' } : null,
      providers: specs.map(spec => {
        const id = spec.id === 'gpt' ? 'openai' : spec.id;
        const profile = value.providers[id]; const configured = Boolean(profile?.encryptedKey);
        return { id, displayName: spec.displayName, baseUrl: profile?.baseUrl ?? spec.defaultBaseUrl,
          configured, credentialConfigured: configured, configuredModel: profile?.model ?? spec.defaultModel,
          selected: id === value.providerId, local: false, category: 'compatible', models: profile?.models ?? spec.defaultModels,
          capabilities: { streaming: false, reasoning: true, reasoningEffort: false } };
      }) };
  });
  ipcMain.handle('settings:save-inference', async (event, value) => {
    trusted(event);
    saveInference(settingsFile, value, key => {
      if (!safeStorage.isEncryptionAvailable()) throw new Error('系统密钥存储不可用，无法保存 API Key');
      return safeStorage.encryptString(key).toString('base64');
    });
    await applySettings();
  });
  ipcMain.handle('settings:save-embedding', async (event, value) => {
    trusted(event);
    saveEmbedding(settingsFile, value, key => {
      if (!safeStorage.isEncryptionAvailable()) throw new Error('系统密钥存储不可用，无法保存 API Key');
      return safeStorage.encryptString(key).toString('base64');
    });
    await applySettings();
  });
  ipcMain.handle('settings:select-model', async event => {
    trusted(event);
    const result = await dialog.showOpenDialog(window, { title: '选择包含 config.json 和模型权重的目录', properties: ['openDirectory'] });
    return result.canceled ? null : result.filePaths[0];
  });
  ipcMain.handle('settings:open-models', (event, kind) => {
    trusted(event);
    return shell.openExternal(kind === 'sbert' ? 'https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2' : 'https://huggingface.co/Qwen/Qwen3-Embedding-0.6B');
  });
  ipcMain.handle('settings:open-data', event => { trusted(event); return shell.openPath(home); });
}

async function runSmoke(services) {
  const assert = require('node:assert/strict');
  const headers = { 'x-theta-desktop-token': token };
  const get = async (base, route, extra = {}) => fetch(base + route, { headers, ...extra });
  assert.equal((await fetch(origin + '/api/v3/health')).status, 403);
  assert.equal((await fetch(services.agentUrl + '/api/v3/health')).status, 403);
  assert.equal((await fetch(services.manualUrl + '/health')).status, 403);
  assert.equal((await get(origin, '/api/v3/health')).status, 200);
  assert.equal((await get(origin, '/api/backend/health')).status, 200);
  assert.equal((await get(origin, '/api/backend/api/projects')).status, 200);
  assert.equal((await get(origin, '/api/backend/api/auth/me')).status, 404);
  assert.equal((await get(origin, '/api/v3/projects', { headers: { ...headers, origin: 'https://evil.example' } })).status, 403);
  assert.equal((await get(origin, '/workbench?mode=conversation')).status, 200);
  // Verify homepage images through the authenticated Electron session. Internal
  // optimizer fetches do not carry its token and previously left these blank.
  await window.loadURL(origin + '/');
  const homepageImages = await window.webContents.executeJavaScript(`(async () => {
    const images = [...document.querySelectorAll('img')].filter(image => image.getAttribute('src')?.includes('/screenshots/'));
    await Promise.race([
      Promise.all(images.map(image => { image.loading = 'eager'; return image.decode(); })),
      new Promise((_, reject) => setTimeout(() => reject(new Error('Homepage images timed out')), 20000)),
    ]);
    return images.map(image => ({ src: image.getAttribute('src'), width: image.naturalWidth }));
  })()`);
  assert.ok(homepageImages.length >= 4, 'Homepage must include actual product screenshots');
  assert.ok(homepageImages.every(image => image.width > 0 && image.src.startsWith('/screenshots/')), 'Homepage screenshots must load directly');
  await window.loadURL(origin + '/workbench?mode=conversation');
  // Exercise the same sandboxed preload and IPC used by the settings dialog.
  const embedding = await window.webContents.executeJavaScript('window.thetaDesktop.read().then(value => value.embedding)');
  const catalog = await window.webContents.executeJavaScript('window.thetaDesktop.catalog()');
  assert.ok(catalog.providers.every(provider => !provider.credentialConfigured));
  assert.equal(embedding.apiKeyConfigured, false);
  assert.equal(embedding.localModel, 'Qwen/Qwen3-Embedding-0.6B');
  const cloud = { ...embedding, mode: 'cloud', model: 'embedding-3', dimensions: 1024 };
  await window.webContents.executeJavaScript(`window.thetaDesktop.saveEmbedding(${JSON.stringify(cloud)})`);
  const configurationResponse = await get(origin, '/api/v3/training-runtime');
  const configurationPayload = await configurationResponse.json();
  assert.equal(configurationResponse.status, 200, JSON.stringify(configurationPayload));
  const configured = configurationPayload.data;
  assert.equal(configured.embedding.preferredMode, 'cloud');
  assert.equal(configured.embedding.provider, 'zhipu');
  assert.equal(configured.embedding.model, 'embedding-3');
  assert.equal(configured.embedding.endpoint, 'https://open.bigmodel.cn/api/paas/v4');
  assert.equal(configured.embedding.configured, false);
  await window.webContents.executeJavaScript(`window.thetaDesktop.saveEmbedding(${JSON.stringify(embedding)})`);
  const manual = await get(origin, '/api/backend/api/projects', { method: 'POST', headers: { ...headers, 'content-type': 'application/json' }, body: JSON.stringify({ name: `Desktop smoke ${Date.now()}` }) });
  assert.equal(manual.status, 201);
  const project = await manual.json();
  // Reproduce column selection through the production frontend proxy, not just the backend.
  const fixture = await pythonCall(['-I', '-c', 'import io,base64; from openpyxl import Workbook; w=Workbook(); s=w.active; s.append(["正文","时间","标签"]); s.append(["本地列预览测试","2026-09-23","测试"]); b=io.BytesIO(); w.save(b); import zipfile; z=zipfile.ZipFile(b, "a"); z.writestr("theta-padding.bin", b"x" * (12 * 1024 * 1024), compress_type=zipfile.ZIP_STORED); z.close(); print(base64.b64encode(b.getvalue()).decode())']);
  const upload = await get(origin, '/api/backend/api/upload?' + new URLSearchParams({ filename: '列预览测试.xlsx', dataset_name: project.dataset_name }), {
    method: 'POST', headers: { ...headers, 'content-type': 'application/octet-stream' }, body: Buffer.from(fixture.trim(), 'base64'),
  });
  const uploaded = await upload.json();
  assert.equal(upload.status, 201, JSON.stringify(uploaded));
  const preview = await get(origin, `/api/backend/api/datasets/${encodeURIComponent(project.dataset_name)}/preview?file_id=${uploaded.id}`);
  const table = await preview.json();
  assert.equal(preview.status, 200, JSON.stringify(table));
  assert.deepEqual(table.columns, ['正文', '时间', '标签']);
  assert.deepEqual(table.rows[0], ['本地列预览测试', '2026-09-23', '测试']);
  const wordFixture = await pythonCall(['-I', '-c', 'import io,base64,zipfile; from docx import Document; d=Document(); d.add_paragraph("中文路径和大文件正文测试"); b=io.BytesIO(); d.save(b); z=zipfile.ZipFile(b, "a"); z.writestr("theta-padding.bin", b"x" * (12 * 1024 * 1024), compress_type=zipfile.ZIP_STORED); z.close(); print(base64.b64encode(b.getvalue()).decode())']);
  const wordBytes = Buffer.from(wordFixture.trim(), 'base64');
  const wordUpload = await get(origin, '/api/backend/api/upload?' + new URLSearchParams({ filename: '中文 Word 大文件.docx', dataset_name: project.dataset_name }), {
    method: 'POST', headers: { ...headers, 'content-type': 'application/octet-stream', 'x-theta-file-size': String(wordBytes.length) }, body: wordBytes,
  });
  const wordFile = await wordUpload.json(); assert.equal(wordUpload.status, 201, JSON.stringify(wordFile));
  const wordPreview = await get(origin, `/api/backend/api/datasets/${encodeURIComponent(project.dataset_name)}/preview?file_id=${wordFile.id}`);
  const wordData = await wordPreview.json(); assert.equal(wordPreview.status, 200, JSON.stringify(wordData));
  assert.equal(wordData.segments[0].text, '中文路径和大文件正文测试');
  const textUpload = await get(origin, '/api/backend/api/upload?' + new URLSearchParams({ filename: '正文.txt', dataset_name: project.dataset_name }), {
    method: 'POST', headers: { ...headers, 'content-type': 'application/octet-stream' }, body: '第一条正文\n第二条正文',
  });
  assert.equal(textUpload.status, 201);
  const textFile = await textUpload.json();
  const textPreview = await (await get(origin, `/api/backend/api/datasets/${encodeURIComponent(project.dataset_name)}/preview?file_id=${textFile.id}`)).json();
  assert.equal(textPreview.inputKind, 'text');
  assert.equal(textPreview.textColumn, 'text');
  assert.deepEqual(textPreview.segments.map(segment => segment.text), ['第一条正文', '第二条正文']);
  const preprocessing = await get(origin, `/api/backend/api/preprocessing/check/${encodeURIComponent(project.dataset_name)}`);
  assert.equal(preprocessing.status, 200);
  assert.equal((await preprocessing.json()).managed_by_training, true);
  assert.equal((await get(origin, '/api/backend/api/projects', { headers: { ...headers, origin: 'https://evil.example' } })).status, 403);
  const inventory = await pythonCall(['-I', '-c', 'import json,sys,sqlite3,ssl,numpy,pandas,scipy,torch,sklearn,gensim,transformers; assert (torch.ones((2,2)) @ torch.ones((2,2))).sum().item() == 8; print(json.dumps({"executable":sys.executable,"prefix":sys.prefix,"version":sys.version.split()[0],"cuda":torch.version.cuda}))']);
  const parsed = JSON.parse(inventory);
  if (process.platform === 'win32') assert.equal(parsed.cuda, null, 'Windows base installer must run independently of optional CUDA components');
  assert.equal(path.resolve(parsed.prefix), path.resolve(runtime, 'python'));
  const model = await get(origin, '/api/backend/api/models/lda');
  assert.equal(model.status, 200);
  assert.equal((await model.json()).modelId, 'lda');
  console.log(JSON.stringify({ ok: true, app: app.getVersion(), node: process.versions.node, electron: process.versions.electron, python: parsed, checks: ['production UI', 'authenticated homepage screenshots', 'agent API', 'manual API', 'project write', 'XLSX upload and column preview through frontend proxy', 'local preprocessing status', 'desktop authentication', 'origin rejection', 'sandboxed settings bridge', 'live GLM embedding configuration', 'bundled Python imports', 'Python model inspection'] }, null, 2));
}
app.whenReady().then(async () => {
  if (startupError) {
    if (smoke) throw startupError;
    while (true) {
      const answer = await dialog.showMessageBox({ type: 'warning', title: 'THETA 数据目录不可写',
        message: '无法在当前数据目录保存文件，请选择有写入权限的文件夹。',
        detail: `${home || ''}\n${startupError.message}\n\n可选择 D 盘等位置的专用文件夹。原项目与配置不会删除；选择已有的 THETA 数据目录可继续使用原数据。`,
        buttons: ['选择数据目录', '退出'], cancelId: 1 });
      if (answer.response !== 0) { app.quit(); return; }
      const chosen = await dialog.showOpenDialog({ title: '选择 THETA 数据目录', properties: ['openDirectory', 'createDirectory'] });
      if (chosen.canceled || !chosen.filePaths[0]) continue;
      try { home = prepareDataHome(chosen.filePaths[0]); break; }
      catch (error) { startupError = error; }
    }
    // Chromium's storage path must be chosen before ready; restart with the selected location.
    app.relaunch({ args: [...process.argv.slice(1).filter(argument => !argument.startsWith('--data-dir=')), `--data-dir=${home}`] });
    app.exit(0); return;
  }
  if (!app.requestSingleInstanceLock()) { app.quit(); return; }
  await clearUpgradeCaches(home, app.getVersion());
  settingsFile = path.join(home, 'settings.json');
  if (!smoke) {
    try { saveDataLocation(locationFile, home); }
    catch { await dialog.showMessageBox({ type: 'warning', message: '本次已使用所选数据目录，但无法记住该位置。', detail: `下次请使用 --data-dir="${home}" 启动 THETA。` }); }
  }
  if (process.platform === 'darwin') app.dock.setIcon(path.join(__dirname, 'ui/icon.png'));
  mkdirSync(path.join(home, 'logs'), { recursive: true });
  providerSpecs = (await import(pathToFileURL(path.join(runtime, 'agent/dist/src/providers/provider-registry.js')).href)).inferenceProviderSpecs;
  registerSettings();
  const desktopSession = session.fromPartition(smoke ? 'theta-smoke' : 'persist:theta');
  desktopSession.setPermissionRequestHandler((_contents, _permission, callback) => callback(false));
  desktopSession.setPermissionCheckHandler(() => false);
  desktopSession.webRequest.onBeforeSendHeaders((details, callback) => {
    if (origin && new URL(details.url).origin === origin) details.requestHeaders['x-theta-desktop-token'] = token;
    callback({ requestHeaders: details.requestHeaders });
  });
  Menu.setApplicationMenu(Menu.buildFromTemplate([
    { label: 'THETA', submenu: [{ label: '设置…', accelerator: 'CmdOrCtrl+,', click: openSettings },
      { label: '打开数据目录', click: () => shell.openPath(home) }, { label: '打开日志目录', click: () => shell.openPath(path.join(home, 'logs')) },
      { label: '重新启动服务', click: () => restart().catch(error => dialog.showErrorBox('启动失败', error.message)) }, { type: 'separator' }, { role: 'quit' }] },
    { role: 'editMenu' }, { role: 'viewMenu' }, { role: 'windowMenu' },
  ]));
  const services = await startService();
  window = new BrowserWindow({ show: !smoke, width: 1440, height: 960, minWidth: 1024, minHeight: 680, title: 'THETA', backgroundColor: '#faf9f6',
    icon: path.join(__dirname, 'ui/icon.png'), webPreferences: { preload: path.join(__dirname, 'preload.cjs'), session: desktopSession, sandbox: true, contextIsolation: true, nodeIntegration: false } });
  window.on('page-title-updated', event => event.preventDefault());
  window.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https:\/\//.test(url)) void shell.openExternal(url);
    return { action: 'deny' };
  });
  window.webContents.on('will-navigate', (event, url) => { if (new URL(url).origin !== origin) event.preventDefault(); });
  await window.loadURL(origin + '/workbench?mode=conversation');
  if (smoke) { await runSmoke(services); app.quit(); }
}).catch(error => {
  console.error(error.stack || error.message);
  if (!smoke) dialog.showErrorBox('THETA 启动失败', error.message);
  process.exitCode = 1; app.quit();
});
app.on('window-all-closed', () => app.quit());
app.on('before-quit', event => {
  if (stopped) return;
  event.preventDefault(); quitting = true;
  stopService().finally(() => { stopped = true; app.exit(process.exitCode || 0); });
});
