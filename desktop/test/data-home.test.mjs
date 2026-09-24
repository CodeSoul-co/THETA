import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, existsSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { prepareDataHome, readDataLocation, saveDataLocation, clearUpgradeCaches } from '../data-home.cjs';

test('writable data homes can be outside the installation and retain their location', t => {
  const root = mkdtempSync(path.join(tmpdir(), 'theta-home-'));
  t.after(() => rmSync(root, {recursive:true, force:true}));
  const selected = path.join(root, '用户数据');
  assert.equal(prepareDataHome(selected), selected);
  for (const folder of ['agent', 'manual', 'data', 'results', 'workspace', 'cache', 'tmp']) assert.ok(existsSync(path.join(selected, folder)));
  const pointer = path.join(root, 'location.json');
  assert.equal(readDataLocation(pointer, 'default'), 'default');
  saveDataLocation(pointer, selected);
  assert.equal(readDataLocation(pointer, 'default'), selected);
  assert.throws(() => prepareDataHome(path.parse(root).root));
  const blocked = path.join(root, 'blocked'); writeFileSync(blocked, 'file');
  assert.throws(() => prepareDataHome(blocked));
  assert.equal(readFileSync(blocked, 'utf8'), 'file');
});
test('upgrade cleanup removes disposable caches but retains datasets, keys, drafts and models', async t => {
  const root = mkdtempSync(path.join(tmpdir(), 'theta-cleanup-'));
  t.after(() => rmSync(root, {recursive:true, force:true}));
  const preserve = ['settings.json', 'manual/manual.sqlite', 'agent/research.sqlite', 'data/source.txt', 'results/result.csv', 'cache/huggingface/weights.bin', 'Partitions/theta/Local Storage/draft'];
  for (const file of [...preserve, 'Cache/Cache_Data/stale', 'Partitions/theta/Code Cache/stale']) {
    const target = path.join(root,file); mkdirSync(path.dirname(target), {recursive:true}); writeFileSync(target,'keep');
  }
  await clearUpgradeCaches(root, 'test-new-version');
  assert.ok(!existsSync(path.join(root,'Cache/Cache_Data')));
  assert.ok(!existsSync(path.join(root,'Partitions/theta/Code Cache')));
  for (const file of preserve) assert.equal(readFileSync(path.join(root,file),'utf8'), 'keep');
  mkdirSync(path.join(root,'Cache'), {recursive:true}); writeFileSync(path.join(root,'Cache/current'),'active');
  await clearUpgradeCaches(root, 'test-new-version');
  assert.ok(existsSync(path.join(root,'Cache/current')));
});

test('Windows first launch uses installation data folder, falls back when blocked, and upgrades retain prior data', async t => {
  const { selectDataHome } = await import('../data-home.cjs');
  const root = mkdtempSync(path.join(tmpdir(), 'theta-install-home-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const options = { platform: 'win32', locationFile: path.join(root, 'location.json'), legacyHome: path.join(root, 'roaming', 'THETA'), installDirectory: path.join(root, 'installed') };
  assert.equal(selectDataHome(options), path.join(options.installDirectory, 'THETA-data'));
  mkdirSync(options.legacyHome, { recursive: true });
  writeFileSync(path.join(options.legacyHome, 'settings.json'), '{}');
  assert.equal(selectDataHome(options), options.legacyHome);
  saveDataLocation(options.locationFile, path.join(root, 'custom'));
  assert.equal(selectDataHome(options), path.join(root, 'custom'));
  const blocked = path.join(root, 'blocked'); writeFileSync(blocked, 'not a directory');
  const fallback = path.join(root, 'fallback');
  assert.equal(selectDataHome({ ...options, locationFile: path.join(root, 'absent'), legacyHome: fallback, installDirectory: blocked }), fallback);
  assert.throws(() => selectDataHome({ ...options, requested: blocked }));
});
