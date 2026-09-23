import test from 'node:test'
import assert from 'node:assert/strict'
import { chartReferenceKey, mergeChartReferences } from './chart-reference.ts'

const chart = (referenceId: string) => ({ referenceId, chartName: '主题占比', chartPath: 'topic_proportion', dataset: 'test', model: 'LDA' })
test('same-titled figures in different jobs remain distinct, re-quoting replaces the same figure', () => {
  const first = chart('job-1:proportion')
  const second = chart('job-2:proportion')
  const refs = mergeChartReferences([first], [second])
  assert.equal(refs.length, 2)
  assert.deepEqual(mergeChartReferences(refs, [{ ...first, model: 'updated' }]), [{ ...first, model: 'updated' }, second])
  const removed = refs.filter(item => chartReferenceKey(item) !== chartReferenceKey(first))
  assert.equal(removed.some(item => chartReferenceKey(item) === chartReferenceKey(first)), false)
  assert.equal(removed.some(item => chartReferenceKey(item) === chartReferenceKey(second)), true)
})
test('only the latest four references remain selected and older drafts have stable keys', () => {
  assert.deepEqual(mergeChartReferences([chart('1'), chart('2'), chart('3'), chart('4')], [chart('5')]).map(chartReferenceKey), ['2', '3', '4', '5'])
  const legacy = { chartName: 'plot', chartPath: 'plot.svg', dataset: 'a', model: 'LDA' }
  assert.equal(chartReferenceKey(legacy), chartReferenceKey({ ...legacy }))
  assert.notEqual(chartReferenceKey(legacy), chartReferenceKey({ ...legacy, dataset: 'b' }))
})
