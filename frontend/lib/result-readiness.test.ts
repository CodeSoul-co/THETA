import test from 'node:test'
import assert from 'node:assert/strict'
import { shouldRefreshResultCatalog, resultReadinessNotice } from './result-readiness.ts'

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
