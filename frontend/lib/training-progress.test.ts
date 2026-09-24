import test from 'node:test'
import assert from 'node:assert/strict'
import { trainingEventText, trainingElapsed } from './training-progress.ts'

test('renders actual epoch and batch values without treating a running batch as a completed epoch', () => {
  const batch = trainingEventText({ id: 1, at: null, kind: 'batch', current: 3, total: 50, batch: { current: 2, total: 8 }, metrics: { loss: .25 } })
  assert.match(batch, /训练轮次 3\/50.*批次 2\/8.*损失 0.25/)
  assert.doesNotMatch(batch, /完成/)
  assert.match(trainingEventText({ id: 2, at: 1, kind: 'epoch', current: 3, total: 50, stage: 'stage2', metrics: { train_loss: 1, val_loss: .1 } }), /第二阶段.*本轮完成.*训练损失 1.*验证损失 0.1/)
  assert.equal(trainingEventText({ id: 3, at: null, kind: 'iteration', current: 7, total: 100 }), '迭代 7/100')
  assert.equal(trainingEventText({ id: 4, at: null, kind: 'early_stop', current: 8 }), '第 8 轮触发提前停止')
  assert.equal(trainingElapsed(125), '2 分 5 秒')
})

test('embedding counts show completed texts, chunks, and batches separately', () => {
  assert.equal(trainingEventText({ id: 1, at: 0, kind: 'embedding', source: 'cloud', scope: 'documents', current: 1, total: 4, chunks: 3, chunkTotal: 8, completedBatches: 2, totalBatches: 5 }), '云端嵌入 · 已完成文本 1/4 · 25% · 已编码文本块 3/8 · 已完成批次 2/5')
})

test('device events explain automatic CPU fallback', () => {
  assert.match(trainingEventText({ id: 1, at: 0, kind: 'device', device: 'cuda:0', status: 'selected' }), /NVIDIA GPU/)
  assert.match(trainingEventText({ id: 2, at: 1, kind: 'device', device: 'cpu', status: 'unavailable' }), /自动使用 CPU/)
  assert.match(trainingEventText({ id: 3, at: 2, kind: 'device', device: 'cpu', status: 'fallback' }), /回退到 CPU 重新执行/)
})

test('optional GPU setup shows download progress and actionable CPU fallback', () => {
  assert.match(trainingEventText({ id: 1, at: 0, kind: 'device', device: 'cpu', status: 'downloading', current: 25, total: 100 }), /25%/)
  assert.match(trainingEventText({ id: 2, at: 0, kind: 'device', device: 'cpu', status: 'disk_space' }), /9 GB.*CPU/)
  assert.match(trainingEventText({ id: 3, at: 0, kind: 'device', device: 'cpu', status: 'setup_failed' }), /本次使用 CPU/)
})
