import test from 'node:test'
import assert from 'node:assert/strict'
import { readDraft, writeDraft } from './workbench-draft.ts'
import { editableParameters, parseParameter, parameterGroup, type ParameterSpec } from './model-parameters.ts'

test('项目与文件草稿分离，刷新读取不会提交任务', () => {
  const data = new Map<string, string>()
  Object.defineProperty(globalThis, 'localStorage', { configurable: true, value: {
    getItem: (key: string) => data.get(key) ?? null,
    setItem: (key: string, value: string) => data.set(key, value),
  } })
  const columns = { textColumn: '正文', timeColumn: '发布时间', metaColumns: ['公众号名称'] }
  writeDraft('project-1:columns:14', columns)
  writeDraft('project-1:config', { parameters: { lda: { max_iter: 10, 'model.alpha': .2 }, dtm: { epochs: 10 } } })
  assert.deepEqual(readDraft('project-1:columns:14', {}), columns)
  assert.deepEqual(readDraft('project-2:columns:14', {}), {})
  assert.deepEqual(readDraft('project-1:columns:15', {}), {})
  assert.equal(data.size, 2)
  data.set('theta:workbench:v1:broken', '{')
  assert.equal(readDraft('broken', 'default'), 'default')
  data.set('theta:workbench:v1:future', JSON.stringify({ version: 2, value: 'unknown' }))
  assert.equal(readDraft('future', 'default'), 'default')
  data.set('theta:workbench:v1:wrong-type', JSON.stringify({ version: 1, value: 'invalid' }))
  assert.deepEqual(readDraft('wrong-type', {}), {})
  data.set('theta:workbench:v1:wrong-array', JSON.stringify({ version: 1, value: {} }))
  assert.deepEqual(readDraft('wrong-array', []), [])
})

test('存储不可用时仍可操作，参数不转换成另一模型的通用参数', () => {
  Object.defineProperty(globalThis, 'localStorage', { configurable: true, get: () => { throw new Error('denied') } })
  assert.doesNotThrow(() => writeDraft('p', {}))
  assert.equal(readDraft('p', 10), 10)
  const spec = { type: 'float', default: .1, target: 'constructor', source: 'model.py' } as ParameterSpec
  assert.equal(parseParameter('0', spec), 0)
  assert.equal(parseParameter('', spec), undefined)
  assert.equal(parseParameter('0.2', spec), .2)
  assert.equal(parseParameter('null', { ...spec, nullable: true }), null)
  assert.throws(() => parseParameter('abc', spec))
  assert.throws(() => parseParameter('2.5', { ...spec, type: 'int' }))
  assert.deepEqual(parseParameter('128, 64', { ...spec, type: 'array' }), [128, 64])
  const entries = editableParameters({ modelId: 'lda', description: '', parameters: { 'model.alpha': spec, embedding_api_key_env: spec, language: spec } })
  assert.deepEqual(entries.map(([key]) => key), ['model.alpha'])
  assert.equal(parameterGroup('prepare.time_slices'), '预处理参数')
})
