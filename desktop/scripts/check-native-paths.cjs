const { spawnSync } = require('node:child_process');
const { mkdtempSync, writeFileSync, rmSync } = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const electron = require('electron');
const root = mkdtempSync(path.join(os.tmpdir(), 'theta-native-paths-'));
const probe = path.join(root, 'main.cjs');
writeFileSync(probe, `
console.log('native-path: main entered');
const { app, session } = require('electron');
const fs = require('node:fs');
const path = require('node:path');
console.log = (...values) => fs.writeSync(1, values.join(' ') + '\\n');
const { prepareDataHome, clearUpgradeCaches } = require(${JSON.stringify(path.resolve(__dirname, '../data-home.cjs'))});
console.log('native-path: prepare data');
const home = prepareDataHome(process.env.THETA_NATIVE_PATH_HOME);
console.log('native-path: set Chromium paths');
app.setPath('userData', home);
app.setPath('sessionData', home);
app.whenReady().then(async () => {
  console.log('native-path: ready');
  app.requestSingleInstanceLock();
  console.log('native-path: caches');
  console.log('native-path: canonical path');
  console.log(fs.realpathSync.native(home));
  console.log('native-path: cache directory');
  const cache = path.join(home, 'Cache', 'Cache_Data');
  fs.mkdirSync(cache, { recursive: true });
  fs.writeFileSync(path.join(cache, 'entry'), 'disposable cache');
  console.log('native-path: cache removal');
  await clearUpgradeCaches(home, 'test');
  if (fs.existsSync(cache)) throw new Error('Upgrade cache was not removed');
  console.log('native-path: session');
  session.fromPartition('theta-path-test');
  console.log('native-path: complete');
  app.exit(0);
}).catch(error => { console.error(error); app.exit(1); });
`);
try {
  for (const name of ['ascii', '中文路径']) {
    console.log(`Checking native Electron startup: ${name}`);
    const result = spawnSync(electron, [probe], {
      env: { ...process.env, THETA_NATIVE_PATH_HOME: path.join(root, name) },
      encoding: 'utf8', timeout: 45000,
    });
    process.stdout.write(result.stdout || '');
    process.stderr.write(result.stderr || '');
    if (result.status !== 0) throw new Error(`Native ${name} startup failed: ${result.status}; ${result.error || result.signal || ''}`);
  }
} finally { rmSync(root, { recursive: true, force: true }); }
