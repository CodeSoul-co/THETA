import assert from 'node:assert/strict'
import test from 'node:test'
import { computationNotice, errorGuidance, LARGE_DATASET_BYTES } from './workbench-guidance.ts'

test('large-data warning is strictly above 20 MB, independent of model selection', () => {
  assert.equal(computationNotice(LARGE_DATASET_BYTES, ['lda']), undefined)
  assert.match(computationNotice(LARGE_DATASET_BYTES + 1, ['lda'])!, /超过 20 MB/)
  assert.match(computationNotice(LARGE_DATASET_BYTES + 1, [], 'en-US')!, /exceeds 20 MB/)
})
test('compute-heavy models warn with small inputs without claiming an exact duration', () => {
  for (const model of ['THETA', 'ctm', 'bertopic', 'dtm', 'nvdm', 'gsm', 'prodlda', 'etm']) assert.match(computationNotice(100, [model])!, /计算量较大/)
  assert.equal(computationNotice(100, ['lda', 'btm']), undefined)
  assert.match(computationNotice(LARGE_DATASET_BYTES + 1, ['theta'])!, /超过 20 MB，且/)
})
test('configuration errors lead to the correct settings; unrelated errors remain intact', () => {
  for (const raw of ['API Key 未配置', 'invalid API key', '尚未配置 OPENAI_API_KEY']) assert.equal(errorGuidance(raw).settingsTab, 'inference')
  assert.equal(errorGuidance('云端嵌入未配置').settingsTab, 'embedding')
  assert.equal(errorGuidance('本地嵌入模型尚未准备好').settingsTab, 'embedding')
  assert.equal(errorGuidance('modelAssetsReady: false').settingsTab, 'embedding')
  assert.match(errorGuidance('CUDA out of memory').message, /内存不足/)
  assert.deepEqual(errorGuidance('本地服务验证失败，请重新启动 THETA'), { message: '本地服务验证失败，请重新启动 THETA' })
  assert.deepEqual(errorGuidance('CSV column text is missing'), { message: 'CSV column text is missing' })
})
