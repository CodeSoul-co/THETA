import { trainingSummary } from '../components/theta-workbench/panels/training-summary.ts'
import test from 'node:test'
import assert from 'node:assert/strict'
import { shouldRefreshResultCatalog, resultReadinessNotice, resultDisplayStatus } from './result-readiness.ts'

test('DTM 训练完成但报告晚到时继续同步，产物发布后停止轮询', () => {
  const completed = { status: 'completed', reportStatus: 'not_requested' }
  assert.equal(shouldRefreshResultCatalog([completed]), true)
  assert.match(resultReadinessNotice(completed, true)!, /整理结果/)
  const ready = { ...completed, reportStatus: 'ready', artifacts: { fileCount: 275 } }
  assert.equal(shouldRefreshResultCatalog([ready]), false)
  assert.equal(resultReadinessNotice(ready, true), undefined)
})

test('多个模型中只要仍有训练或等待产物的任务就继续同步', () => {
  const ready = { status: 'completed', artifacts: { fileCount: 236 } }
  assert.equal(shouldRefreshResultCatalog([ready, { status: 'running' }]), true)
  assert.equal(shouldRefreshResultCatalog([ready, { status: 'queued' }]), true)
  assert.equal(shouldRefreshResultCatalog([ready, { status: 'completed' }]), true)
})

test('失败、取消和报告整理失败不伪装成正在生成，也不无限轮询', () => {
  assert.equal(shouldRefreshResultCatalog([]), false)
  assert.equal(shouldRefreshResultCatalog([{ status: 'failed' }, { status: 'cancelled' }]), false)
  const incomplete = { status: 'completed', reportStatus: 'incomplete' }
  assert.equal(shouldRefreshResultCatalog([incomplete]), false)
  assert.match(resultReadinessNotice(incomplete, true)!, /整理不完整/)
  assert.match(resultReadinessNotice({ status: 'completed' }, false)!, /结果文件同步/)
})

test('训练结束、报告生成和浏览器读取结果完成后才计为完成', () => {
  const result = { jobId: 'one', modelId: 'lda', status: 'completed', percent: 100, reportStatus: 'not_requested' }
  assert.equal(resultDisplayStatus(result), 'running')
  assert.equal(resultDisplayStatus({ ...result, reportStatus: 'generating' }), 'running')
  const ready = { ...result, reportStatus: 'ready', artifacts: { fileCount: 12 } }
  assert.equal(resultDisplayStatus(ready, true), 'running')
  assert.equal(resultDisplayStatus(ready), 'completed')
  assert.equal(resultDisplayStatus(ready, false, true), 'failed')
  const pending = trainingSummary(undefined, { results: [result] } as never)
  assert.equal(pending.completed, 0); assert.equal(pending.state, 'running'); assert.equal(pending.percent, 99)
  assert.equal(trainingSummary(undefined, { results: [ready] } as never).completed, 1)
})
