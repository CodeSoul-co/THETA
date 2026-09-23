import assert from 'node:assert/strict'
import test from 'node:test'
import { configFromPlans, plansFromConfig, type EditorTrainingPlan } from './analysis-config.ts'

test('shared editor prefills distinct models, retains recommendations and submits only the final selection', () => {
  const originals: EditorTrainingPlan[] = [
    { modelId: 'lda', textColumn: '正文', params: { num_topics: 8, alpha: 0.1, max_iter: 20 }, rationale: '词袋基线', timeoutSeconds: 600 },
    { modelId: 'prodlda', textColumn: '正文', params: { num_topics: 12, epochs: 80, learning_rate: 0.002 }, rationale: '神经模型比较', timeoutSeconds: 800 },
  ]
  const config = configFromPlans(originals)
  assert.deepEqual(config.models, ['lda', 'prodlda'])
  assert.equal(config.parameters.lda.num_topics, 8)
  assert.equal(config.parameters.prodlda.num_topics, 12)
  config.parameters.lda.alpha = 0.2; config.parameters.prodlda.epochs = 3
  const final = plansFromConfig(config, originals)
  assert.equal(final[0].params.alpha, 0.2); assert.equal(final[0].params.epochs, undefined)
  assert.equal(final[1].params.epochs, 3); assert.equal(final[1].params.alpha, undefined)
  assert.equal(final[1].timeoutSeconds, 800)
  config.models = ['prodlda', 'dtm']; config.timeColumn = '发布时间'; config.parameters.dtm = { num_topics: 10 }
  assert.deepEqual(plansFromConfig(config, originals).map(plan => plan.modelId), ['prodlda', 'dtm'])
  assert.equal(plansFromConfig(config, originals)[1].timeColumn, '发布时间')
})

test('cloud selection needs fresh consent and shared controls override hidden legacy parameters', () => {
  const originals = [{ modelId: 'theta', textColumn: '正文', params: { num_topics: 9, mode: 'zero_shot', embedding_provider: 'cloud', 'prepare.vocab_size': 7000, embedding_api_base: 'https://old.invalid' }, rationale: '推荐', timeoutSeconds: 1000, externalRequestLimit: 200 }]
  const config = configFromPlans(originals)
  assert.equal(config.embeddingProvider, 'cloud'); assert.equal(config.cloudConfirmed, false)
  config.vocabSize = 3000; config.embeddingProvider = 'local'
  const final = plansFromConfig(config, originals)[0]
  assert.equal(final.params.vocab_size, 3000); assert.equal(final.params['prepare.vocab_size'], undefined)
  assert.equal(final.params.embedding_provider, 'local'); assert.equal(final.params.embedding_api_base, undefined)
  assert.equal(final.externalRequestLimit, 200)
})
