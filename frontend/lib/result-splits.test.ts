import test from 'node:test'
import assert from 'node:assert/strict'
import { filesForDataset, formatSplitValue } from './result-splits.ts'
import { resolveChartDataFiles } from './chart-data.ts'

test('dataset scope isolates figures, metrics and reference source tables', () => {
  const files = ['zh/global/topic_proportions.png', 'zh/global/topic_proportions.csv', 'splits/train/theta.npy', 'zh/splits/train/global/topic_proportions.png', 'zh/splits/train/global/topic_proportions.csv', 'zh/splits/validation/global/topic_proportions.csv'].map(name => ({ name, path: name }))
  assert.equal(filesForDataset(files, 'all').length, 2)
  const train = filesForDataset(files, 'train')
  assert.equal(train.length, 3)
  assert.deepEqual(resolveChartDataFiles(train[1], train).map(file => file.path), ['zh/splits/train/global/topic_proportions.csv'])
  assert.deepEqual(filesForDataset(files, 'test'), [])
})
test('small signed differences remain visible rather than rounding to the same four digits', () => {
  assert.notEqual(formatSplitValue(.003516786964610219), formatSplitValue(.0035171546041965485))
  assert.equal(formatSplitValue(-.008033580146729946), '-0.0080335801')
})
