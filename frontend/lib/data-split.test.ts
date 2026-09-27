import test from 'node:test'
import assert from 'node:assert/strict'
import { defaultDataSplit, splitError } from './data-split.ts'

test('关闭自定义划分不受保留的未完成设置影响；开启后必须完整配置', () => {
  assert.equal(splitError({ ...defaultDataSplit, ratios: [0, 0, 0], mode: 'upload' }), undefined)
  assert.equal(splitError({ ...defaultDataSplit, enabled: true }), undefined)
  assert.match(splitError({ ...defaultDataSplit, enabled: true, ratios: [.8, .2, .1] })!, /100%/)
  assert.match(splitError({ ...defaultDataSplit, enabled: true, ratios: [1, 0, 0] })!, /大于 0/)
  assert.match(splitError({ ...defaultDataSplit, enabled: true, mode: 'upload' })!, /分别上传/)
  const sources = { train: { fileId: '1', textColumn: 'body' }, validation: { fileId: '2', textColumn: 'text' }, test: { fileId: '3', textColumn: '正文' } }
  assert.equal(splitError({ ...defaultDataSplit, enabled: true, mode: 'upload', sources }), undefined)
  assert.match(splitError({ ...defaultDataSplit, enabled: true, mode: 'upload', sources: { ...sources, test: sources.train } })!, /不同/)
})
