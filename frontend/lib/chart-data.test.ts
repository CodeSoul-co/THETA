import test from 'node:test'
import assert from 'node:assert/strict'
import { resolveChartDataFiles } from './chart-data.ts'

test('两工作台的中文图表匹配相同真实数据，不使用无关指标兜底', () => {
  const files = ['topic_proportions.csv', 'evaluation_metrics.csv', 'topic_beta_similarity.csv', 'document_projection.csv']
    .map(name => ({ name, path: 'zh/global/' + name, url: name }))
  const resolve = (name: string) => resolveChartDataFiles({ name, path: 'zh/global/' + name }, files).map(file => file.name)
  assert.deepEqual(resolve('主题占比分布.png'), ['topic_proportions.csv'])
  assert.deepEqual(resolve('7项核心指标.png'), ['evaluation_metrics.csv'])
  assert.deepEqual(resolve('主题相似度图.png'), ['topic_beta_similarity.csv'])
  assert.deepEqual(resolve('文档主题UMAP图.png'), ['document_projection.csv'])
  assert.deepEqual(resolve('尚未支持的图.png'), [])
})

test('分主题图只读取本主题的绘图数据', () => {
  const files = [1, 2].map(id => ({ name: 'word_importance.csv', path: `zh/topic/topic_${id}/word_importance.csv` }))
  assert.deepEqual(resolveChartDataFiles({ name: '主题2 词云.png', path: 'zh/topic/topic_2/主题2 词云.png' }, files), [files[1]])
})
