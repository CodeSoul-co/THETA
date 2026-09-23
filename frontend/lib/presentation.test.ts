import assert from 'node:assert/strict'
import { test } from 'node:test'
import { artifactLabel, fieldLabel, modeLabel, phaseLabel, statusLabel, systemText, advisoryErrorMessage, conversationDisplayText } from './presentation.ts'

test('引用和 job ID 不显示在消息正文，图片链接及原始请求保持完整', () => {
  const job = 'job-' + 'a'.repeat(64)
  const raw = `改成英文\n[[cite]][{"jobId":"${job}","figure":"native/global/topic_wordcloud_grid_3.svg"}][[/cite]]`
  assert.equal(conversationDisplayText(raw), '改成英文')
  assert.ok(raw.includes(job))
  assert.equal(conversationDisplayText(`已完成 ${job}`), '已完成 本次任务')
  const image = `![调整版图](/api/artifacts/${job}/figure.png)`
  assert.equal(conversationDisplayText(image), image)
  assert.equal(conversationDisplayText('消息[[cite]][{"jobId":"job-abcd'), '消息')
})

test('共享结果助手将供应商错误展示为中文，不伪装成分析结果', () => {
  assert.match(advisoryErrorMessage(new Error('DeepSeek HTTP 503: Service is too busy')), /模型服务繁忙（503）/u)
  assert.match(advisoryErrorMessage(new Error('请求超时')), /响应超时/u)
  assert.match(advisoryErrorMessage(new Error('HTTP 429 rate limit')), /请求过于频繁/u)
  assert.match(advisoryErrorMessage(new Error('unknown upstream error')), /无法完成结果咨询/u)
})

test('图表、数据表及带后缀的文件展示中文名称', () => {
  const cases = [
    ['native/global/Topic Evolution.png', '主题演变'],
    ['native/global/evaluation_metrics.csv', '评估指标'],
    ['Topic 12 Word Cloud.svg', '主题 12 · 词云'],
    ['topic_wordcloud_grid_10.png', '主题词云总览 · 10'],
    ['adjustments/abc123abc123/global/topic_wordcloud_grid_3.png', '主题词云总览 · 3（调整版）'],
    ['Topic Evolution-adjusted-4.png', '主题演变（调整版 4）'],
    ['theta_k20.npy', '文档主题矩阵 · 20'],
    ['topic_words_20260918_192843.json', '主题词 · 20260918 192843'],
    ['config_zero_shot.json', '训练配置（零样本分析）'],
    ['年度趋势.png', '年度趋势'],
  ]
  for (const [name, expected] of cases) assert.equal(artifactLabel(name), expected)
})

test('状态和阶段保持不同语义，未知值不伪装为成功', () => {
  assert.equal(statusLabel('cancelled'), '已取消')
  assert.equal(statusLabel('failed'), '失败')
  assert.equal(statusLabel('queued'), '排队中')
  assert.equal(statusLabel('new_future_state'), '状态待确认')
  assert.equal(phaseLabel('completed'), '已完成')
  assert.equal(phaseLabel('preprocessing'), '数据预处理')
  assert.equal(modeLabel('unsupervised'), '无监督分析')
})

test('系统字段和指标中文化，用户数据字段不被篡改', () => {
  assert.equal(fieldLabel('iRBO'), '主题差异度')
  assert.equal(fieldLabel('T12'), '主题 12')
  assert.equal(fieldLabel('learningRate'), '学习率')
  assert.equal(fieldLabel('matrix_row'), '矩阵行号')
  assert.equal(fieldLabel('customer_segment'), 'customer_segment')
  assert.equal(fieldLabel('年份'), '年份')
})

test('训练提示翻译不影响原始记录', () => {
  const raw = 'Epoch 5/10 loss=0.25'
  assert.equal(systemText(raw), '训练轮次 5 / 10 损失=0.25')
  assert.equal(raw, 'Epoch 5/10 loss=0.25')
  assert.equal(systemText('unrecognized message', '读取失败，请重试'), '读取失败，请重试')
})

test('未知产物具有可区分的中文展示名，原名仍可用于请求', () => {
  const file = { name: 'unregistered_figure.png', url: '/original/path' }
  assert.match(artifactLabel(file.name, 'figure'), /^其他图表 · \d{6}$/)
  assert.notEqual(artifactLabel(file.name), artifactLabel('another_file.png'))
  assert.equal(file.name, 'unregistered_figure.png')
  assert.equal(file.url, '/original/path')
})
