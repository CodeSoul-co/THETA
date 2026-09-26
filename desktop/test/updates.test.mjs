import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { mkdtempSync, rmSync, existsSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { createHash } from 'node:crypto';
import updates from '../updates.cjs';
const { newer, selectRelease, downloadInstaller, createUpdates } = updates;
const bytes = Buffer.from('test installer');
const digest = createHash('sha256').update(bytes).digest('hex');
function release(version = '0.3.9', suffix = 'mac-arm64.dmg') {
  const name = `THETA-${version}-${suffix}`;
  return { tag_name: `desktop-v${version}`, prerelease: true, draft: false, assets: [
    { name, browser_download_url: `https://github.com/CodeSoul-co/THETA/releases/download/desktop-v${version}/${name}`, digest: `sha256:${digest}`, size: bytes.length },
    { name: suffix.startsWith('mac') ? 'latest-mac.yml' : 'latest.yml' },
  ] };
}
function temp(t) { const home = mkdtempSync(path.join(tmpdir(), 'theta-update-')); t.after(() => rmSync(home, { recursive: true, force: true })); return home; }
test('selects only a newer compatible complete desktop release, including preview releases', () => {
  assert.equal(newer('0.3.10', '0.3.9'), true);
  assert.equal(newer('0.3.9', '0.3.9'), false);
  assert.equal(selectRelease([release('0.3.10'), release()], '0.3.8', 'darwin', 'arm64').version, '0.3.10');
  assert.equal(selectRelease([release()], '0.3.9', 'darwin', 'arm64'), null);
  assert.equal(selectRelease([release()], '0.3.8', 'darwin', 'x64'), null);
  assert.equal(selectRelease([{ ...release(), draft: true }], '0.3.8', 'darwin', 'arm64'), null);
  const incomplete = release(); incomplete.assets.pop();
  assert.equal(selectRelease([incomplete], '0.3.8', 'darwin', 'arm64'), null);
  const unsafe = release(); unsafe.assets[0].browser_download_url = 'https://example.com/app.dmg';
  assert.throws(() => selectRelease([unsafe], '0.3.8', 'darwin', 'arm64'), /校验信息/);
});
test('installer downloads are size and checksum verified; invalid partial files are removed', async t => {
  const home = temp(t), info = selectRelease([release()], '0.3.8', 'darwin', 'arm64');
  await assert.rejects(downloadInstaller(info, home, () => {}, async () => new Response('tampered')), /校验失败/);
  assert.equal(existsSync(path.join(home, info.name)), false);
  assert.equal(existsSync(path.join(home, info.name + '.part')), false);
  const progress = [];
  const file = await downloadInstaller(info, home, value => progress.push(value), async () => new Response(bytes));
  assert.deepEqual(readFileSync(file), bytes); assert.equal(progress.at(-1), 100);
  assert.equal(await downloadInstaller(info, home, () => {}, async () => { throw Error('should reuse cache'); }), file);
});
test('native update checks, downloads and installs only after explicit confirmation', async t => {
  const home = temp(t), native = new EventEmitter(); let installed = 0, allowed = false;
  native.checkForUpdates = async () => ({ updateInfo: { version: '0.3.9' } });
  native.downloadUpdate = async () => { native.emit('download-progress', { percent: 65 }); return ['update.exe']; };
  native.quitAndInstall = () => installed++;
  const manager = createUpdates({ version: '0.3.8', platform: 'win32', arch: 'x64', home, enabled: true, manualMac: false,
    nativeFactory: () => native, emit() {}, notify() {}, prepareInstall: async () => allowed, recoverInstall() {}, openPath() {},
    fetcher: async () => Response.json([release('0.3.9', 'win-x64.exe')]),
  });
  assert.equal((await manager.check()).status, 'available');
  assert.equal(native.autoInstallOnAppQuit, false); assert.equal(native.allowDowngrade, false);
  assert.equal((await manager.download()).status, 'ready');
  await manager.install(); assert.equal(installed, 0); assert.equal(manager.state().status, 'ready');
  allowed = true; await manager.install(); assert.equal(installed, 1);
  manager.configure(false); assert.equal(JSON.parse(readFileSync(path.join(home, 'updates.json'))).automatic, false);
});
test('failed network checks are recoverable and development smoke never checks the network', async t => {
  const home = temp(t); let count = 0;
  const options = { version: '0.3.8', platform: 'win32', arch: 'x64', home, enabled: true, emit() {}, notify() {}, recoverInstall() {},
    fetcher: async () => { count++; throw Error('offline'); },
  };
  const manager = createUpdates(options);
  assert.equal((await manager.check()).status, 'error'); assert.equal((await manager.check()).status, 'error'); assert.equal(count, 2);
  const disabled = createUpdates({ ...options, enabled: false }); disabled.start(); await disabled.check(); disabled.stop(); assert.equal(count, 2);
});

test('unsigned Mac opens only a verified installer and preserves update preferences', async t => {
  const home = temp(t); let opened = 0;
  const options = { version: '0.3.8', platform: 'darwin', arch: 'arm64', home, enabled: true, manualMac: true,
    emit() {}, notify() {}, recoverInstall() {}, openPath: async () => { opened++; return ''; },
    fetcher: async url => url.includes('api.github.com') ? Response.json([release()]) : new Response(bytes),
  };
  const manager = createUpdates(options);
  await manager.check(); await manager.download(); await manager.install(); assert.equal(opened, 1);
  const { writeFileSync } = await import('node:fs');
  writeFileSync(path.join(home, 'updates', 'THETA-0.3.9-mac-arm64.dmg'), 'modified');
  await manager.install(); assert.equal(opened, 1); assert.equal(manager.state().status, 'error');
  await manager.download(); await manager.install(); assert.equal(opened, 2);
  manager.configure(false); assert.equal(createUpdates(options).state().automatic, false);
});
test('inconsistent native metadata cannot become an installable candidate', async t => {
  const native = new EventEmitter(); let downloads = 0;
  native.checkForUpdates = async () => ({ updateInfo: { version: '9.9.9' } });
  native.downloadUpdate = async () => { downloads++; return []; };
  const manager = createUpdates({ version: '0.3.8', platform: 'win32', arch: 'x64', home: temp(t), enabled: true,
    nativeFactory: () => native, emit() {}, notify() {}, fetcher: async () => Response.json([release('0.3.9', 'win-x64.exe')]),
  });
  await manager.check(); assert.equal(manager.state().status, 'error');
  await manager.download(); assert.equal(downloads, 0);
});
