const fs = require('node:fs');
const path = require('node:path');
const { randomUUID } = require('node:crypto');

function prepareDataHome(directory) {
  if (!path.isAbsolute(directory) || path.dirname(directory) === directory) throw new Error('请选择磁盘中的专用文件夹，例如 D:\\THETA-data，而不是磁盘根目录。');
  const home = path.resolve(directory);
  for (const relative of ['', 'logs', 'data', 'results', 'workspace', 'agent', 'manual', 'cache', 'tmp']) {
    const folder = path.join(home, relative);
    fs.mkdirSync(folder, { recursive: true });
    const probe = path.join(folder, `.theta-write-check-${randomUUID()}`);
    try { fs.writeFileSync(probe, '', { flag: 'wx', mode: 0o600 }); }
    finally { if (fs.existsSync(probe)) fs.unlinkSync(probe); }
  }
  return home;
}
function readDataLocation(file, fallback) {
  if (!fs.existsSync(file)) return fallback;
  const value = JSON.parse(fs.readFileSync(file, 'utf8'));
  if (typeof value.directory !== 'string' || !path.isAbsolute(value.directory)) throw new Error('数据目录记录无效，请使用 --data-dir 指定已有数据目录。');
  return value.directory;
}
function saveDataLocation(file, directory) {
  fs.writeFileSync(file, JSON.stringify({ directory }), { mode: 0o600 });
}
module.exports = { prepareDataHome, readDataLocation, saveDataLocation };

// Only disposable Chromium caches; project databases, drafts, credentials and model caches stay intact.
function clearUpgradeCaches(home, version) {
  const marker = path.join(home, '.desktop-version');
  if (fs.existsSync(marker) && fs.readFileSync(marker, 'utf8') === version) return;
  const root = fs.realpathSync(home);
  for (const relative of ['Cache/Cache_Data', 'Code Cache', 'GPUCache', 'ShaderCache', 'GrShaderCache', 'DawnGraphiteCache', 'DawnWebGPUCache', 'Partitions/theta/Cache', 'Partitions/theta/Code Cache', 'Partitions/theta/GPUCache']) {
    const target = path.join(home, relative);
    if (!fs.existsSync(target)) continue;
    const resolved = path.relative(root, fs.realpathSync(target));
    if (resolved.startsWith('..') || path.isAbsolute(resolved)) continue;
    try { fs.rmSync(target, { recursive: true, force: true }); } catch { /* A locked cache can be retried next upgrade. */ }
  }
  fs.writeFileSync(marker, version);
}
module.exports.clearUpgradeCaches = clearUpgradeCaches;
