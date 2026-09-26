const path = require('node:path');
const fs = require('node:fs');
const { createHash } = require('node:crypto');
const { Readable, Transform } = require('node:stream');
const { pipeline } = require('node:stream/promises');
const REPOSITORY = 'https://github.com/CodeSoul-co/THETA';
const API = 'https://api.github.com/repos/CodeSoul-co/THETA/releases?per_page=100';
const versionParts = value => /^\d+\.\d+\.\d+$/.test(value) ? value.split('.').map(Number) : null;
function newer(a, b) {
  const left = versionParts(a), right = versionParts(b);
  if (!left || !right) return false;
  for (let i = 0; i < 3; i++) if (left[i] !== right[i]) return left[i] > right[i];
  return false;
}
function selectRelease(releases, current, platform, arch) {
  if (!Array.isArray(releases)) throw new Error('更新服务返回的数据无效');
  const suffix = platform === 'darwin' ? `mac-${arch}.dmg` : `win-${arch}.exe`;
  const candidates = releases.filter(release => !release.draft && /^desktop-v\d+\.\d+\.\d+$/.test(release.tag_name))
    .sort((a, b) => newer(a.tag_name.slice(9), b.tag_name.slice(9)) ? -1 : 1);
  for (const release of candidates) {
    const version = release.tag_name.slice(9);
    if (!newer(version, current)) continue;
    const name = `THETA-${version}-${suffix}`;
    const asset = release.assets?.find(item => item.name === name);
    const metadata = platform === 'darwin' ? 'latest-mac.yml' : 'latest.yml';
    if (!asset || !release.assets.some(item => item.name === metadata)) continue;
    const url = `${REPOSITORY}/releases/download/${release.tag_name}/${name}`;
    if (asset.browser_download_url !== url || !/^sha256:[a-f0-9]{64}$/.test(asset.digest || '') || !Number.isSafeInteger(asset.size) || asset.size <= 0 || asset.size > 4 * 1024 ** 3) throw new Error('更新文件缺少可信的地址或校验信息');
    return { version, url, name, size: asset.size, sha256: asset.digest.slice(7), feed: `${REPOSITORY}/releases/download/${release.tag_name}/` };
  }
  return null;
}
async function hashFile(file) {
  const digest = createHash('sha256');
  for await (const chunk of fs.createReadStream(file)) digest.update(chunk);
  return digest.digest('hex');
}
async function downloadInstaller(release, directory, progress, fetcher = fetch) {
  await fs.promises.mkdir(directory, { recursive: true });
  const target = path.join(directory, release.name), partial = target + '.part';
  if (fs.existsSync(target) && await hashFile(target) === release.sha256) return target;
  try {
    const response = await fetcher(release.url, { signal: AbortSignal.timeout(30 * 60 * 1000) });
    if (!response.ok || !response.body) throw new Error(`下载更新失败（HTTP ${response.status}）`);
    let received = 0;
    const digest = createHash('sha256');
    await pipeline(Readable.fromWeb(response.body), new Transform({ transform(chunk, _encoding, done) {
      received += chunk.length;
      if (received > release.size) return done(new Error('更新文件大小与发布记录不符'));
      digest.update(chunk); progress(Math.floor(received / release.size * 100)); done(null, chunk);
    } }), fs.createWriteStream(partial, { flags: 'w', mode: 0o600 }));
    if (received !== release.size || digest.digest('hex') !== release.sha256) throw new Error('更新文件校验失败，请重新下载');
    await fs.promises.rm(target, { force: true });
    await fs.promises.rename(partial, target);
    return target;
  } finally { await fs.promises.rm(partial, { force: true }); }
}
function createUpdates({ version, platform, arch, home, enabled, nativeFactory, manualMac, emit, notify, prepareInstall, recoverInstall, openPath, fetcher = fetch }) {
  const config = path.join(home, 'updates.json');
  let automatic = true;
  try { automatic = JSON.parse(fs.readFileSync(config, 'utf8')).automatic !== false; } catch { /* First launch. */ }
  let state = { status: enabled ? 'idle' : 'disabled', currentVersion: version, automatic, manualInstall: manualMac, percent: 0 };
  let release, native, downloaded, busy = false, installing = false, timer, startup;
  const update = values => { state = { ...state, ...values }; emit({ ...state }); };
  const fail = error => {
    update({ status: 'error', error: error.message || String(error) });
    if (installing) { installing = false; void recoverInstall(); }
  };
  async function check(announce = false) {
    if (!enabled || busy || ['ready', 'installing', 'downloading'].includes(state.status)) return { ...state };
    busy = true; release = undefined; native?.removeAllListeners(); native = undefined;
    update({ status: 'checking', error: undefined, percent: 0, availableVersion: undefined });
    try {
      const response = await fetcher(API, { headers: { Accept: 'application/vnd.github+json' }, signal: AbortSignal.timeout(20000) });
      if (!response.ok) throw new Error(`无法连接更新服务（HTTP ${response.status}），请检查网络后重试`);
      const candidate = selectRelease(await response.json(), version, platform, arch);
      if (!candidate) { update({ status: 'current', availableVersion: undefined }); return { ...state }; }
      if (!manualMac) {
        native = nativeFactory(candidate.feed);
        native.autoDownload = false; native.autoInstallOnAppQuit = false; native.allowDowngrade = false;
        native.on('error', fail);
        native.on('download-progress', data => update({ percent: Math.floor(data.percent) }));
        const result = await native.checkForUpdates();
        if (result?.updateInfo.version !== candidate.version) throw new Error('更新清单版本与发布版本不一致');
      }
      release = candidate;
      update({ status: 'available', availableVersion: release.version });
      if (announce) void Promise.resolve(notify('available', { ...state })).catch(() => {});
    } catch (error) { fail(error); }
    finally { busy = false; }
    return { ...state };
  }
  async function download() {
    if (busy || !release || !['available', 'error'].includes(state.status)) return { ...state };
    busy = true; update({ status: 'downloading', error: undefined, percent: 0 });
    try {
      if (manualMac) downloaded = await downloadInstaller(release, path.join(home, 'updates'), percent => { if (state.percent !== percent) update({ percent }); }, fetcher);
      else {
        if (!native) throw new Error('请先重新检查更新');
        const files = await native.downloadUpdate();
        if (!files?.length) throw new Error('更新下载未完成，请重试');
      }
      update({ status: 'ready', percent: 100 });
      void Promise.resolve(notify('ready', { ...state })).catch(() => {});
    } catch (error) { fail(error); }
    finally { busy = false; }
    return { ...state };
  }
  async function install() {
    if (state.status !== 'ready' || busy) return;
    busy = true;
    try {
      if (manualMac) {
        if (!downloaded || await hashFile(downloaded) !== release.sha256) throw new Error('安装包校验失败，请重新下载');
        const error = await openPath(downloaded);
        if (error) throw new Error(error);
      } else {
        if (!await prepareInstall()) return;
        installing = true; update({ status: 'installing' });
        native.quitAndInstall(true, true);
      }
    } catch (error) { fail(error); }
    finally { busy = false; }
  }
  function configure(value) {
    if (typeof value !== 'boolean') throw new Error('无效的更新偏好');
    fs.writeFileSync(config, JSON.stringify({ automatic: value }), { mode: 0o600 });
    update({ automatic: value }); return { ...state };
  }
  return { state: () => ({ ...state }), check, download, install, configure,
    start() { if (!enabled) return;
      // Remove only installers for versions already installed, never project data.
      const directory = path.join(home, 'updates');
      try { for (const name of fs.readdirSync(directory)) {
        const match = /^THETA-(\d+\.\d+\.\d+)-mac-(?:arm64|x64)\.dmg(?:\.part)?$/.exec(name);
        if (match && !newer(match[1], version)) { try { fs.unlinkSync(path.join(directory, name)); } catch {} }
      } } catch {}
      startup = setTimeout(() => { if (state.automatic) void check(true); }, 30000); startup.unref(); timer = setInterval(() => { if (state.automatic) void check(true); }, 6 * 3600000); timer.unref(); },
    stop() { clearTimeout(startup); clearInterval(timer); },
  };
}
module.exports = { newer, selectRelease, downloadInstaller, createUpdates };
