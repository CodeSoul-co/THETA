import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, existsSync, rmSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { stageDelivery, createDeliveryZip } from './delivery-archive.js';
import { execFileSync } from 'node:child_process';
import { pythonExecutable } from '../src/adapters/python-worker.js';

test('ZIP 中文目录与图名按 UTF-8 跨平台保存', async () => {
  const root = mkdtempSync(path.join(os.tmpdir(), 'theta-zip-'));
  try {
    const stage = path.join(root, 'stage');
    mkdirSync(path.join(stage, '训练结果'), { recursive: true });
    writeFileSync(path.join(stage, '训练结果/主题占比.csv'), '主题,占比\n一,1');
    const archive = path.join(root, 'result.zip');
    await createDeliveryZip(stage, archive);
    const result = execFileSync(pythonExecutable(), ['-c', "from zipfile import ZipFile; import sys,json; z=ZipFile(sys.argv[1]); print(json.dumps(z.namelist(),ensure_ascii=False)); assert all(i.flag_bits & 0x800 for i in z.infolist())", archive], { encoding: 'utf8' });
    assert.deepEqual(JSON.parse(result), ['训练结果/主题占比.csv']);
  } finally { rmSync(root, { recursive: true, force: true }); }
});

test('交付包含模型、数据、固定加载代码和说明，不收集环境密钥', () => {
  const root = mkdtempSync(path.join(os.tmpdir(), 'theta-delivery-'));
  try {
    mkdirSync(path.join(root, 'src/models'), { recursive: true });
    writeFileSync(path.join(root, 'src/models/model_delivery.py'), '# fixed loader');
    writeFileSync(path.join(root, 'src/models/.env'), 'SECRET=do-not-export');
    writeFileSync(path.join(root, 'trained_model.joblib'), 'trusted test checkpoint');
    const stage = path.join(root, 'stage');
    stageDelivery(stage, [{ name: 'training/trained_model.joblib', path: path.join(root, 'trained_model.joblib') }], root);
    assert.ok(existsSync(path.join(stage, 'training/trained_model.joblib')));
    assert.ok(existsSync(path.join(stage, '代码/src/models/model_delivery.py')));
    assert.equal(existsSync(path.join(stage, '代码/src/models/.env')), false);
    assert.match(readFileSync(path.join(stage, '使用方法.md'), 'utf8'), /load_trained_model/u);
    assert.throws(() => stageDelivery(stage, [{ name: '../escape', path: path.join(root, 'trained_model.joblib') }], root), /路径无效/u);
  } finally { rmSync(root, { recursive: true, force: true }); }
});
