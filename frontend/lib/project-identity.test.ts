import test from 'node:test'
import assert from 'node:assert/strict'
import { reconcileProjectIdentity } from './project-identity.ts'

test('同名历史数据库项目刷新时不共享标签 ID 或模型配置', () => {
  const previous = [{ id: 'proj-db-28', dbProjectId: 28, name: '测试', datasetName: '测试', models: ['prodlda'] }]
  const next = [
    { id: 'proj-db-31', dbProjectId: 31, name: '测试', datasetName: '测试', models: ['theta'] },
    { ...previous[0], models: ['theta'] },
  ]
  const result = reconcileProjectIdentity(next, previous)
  assert.equal(result[0].id, 'proj-db-31')
  assert.deepEqual(result[0].models, ['theta'])
  assert.equal(result[1].id, 'proj-db-28')
  assert.deepEqual(result[1].models, ['prodlda'])
})

test('唯一临时项目可迁移，歧义数据集不可将多个项目合并为一个标签', () => {
  const previous = [{ id: 'new-temp', name: '临时', datasetName: 'data', models: ['lda'] }]
  const next = [{ id: 'proj-db-1', dbProjectId: 1, name: '新项目', datasetName: 'data' }]
  assert.equal(reconcileProjectIdentity(next, previous)[0].id, 'new-temp')
  const ambiguous = [...next, { ...next[0], id: 'proj-db-2', dbProjectId: 2 }]
  assert.deepEqual(reconcileProjectIdentity(ambiguous, previous).map(p => p.id), ['proj-db-1', 'proj-db-2'])
  assert.equal(reconcileProjectIdentity(next, [...previous, { ...previous[0], id: 'another-temp' }])[0].id, 'proj-db-1')
})

test('数据库身份优先于名称，归档后重建不可继承旧标签', () => {
  const previous = [{ id: 'tab-1', dbProjectId: 1, name: '旧名', datasetName: 'data' }]
  assert.equal(reconcileProjectIdentity([{ ...previous[0], id: 'proj-db-1', name: '新名' }], previous)[0].id, 'tab-1')
  assert.equal(reconcileProjectIdentity([{ ...previous[0], id: 'proj-db-2', dbProjectId: 2 }], previous)[0].id, 'proj-db-2')
})
