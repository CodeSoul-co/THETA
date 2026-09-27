import test from 'node:test'
import assert from 'node:assert/strict'
import { analysisReportEndpoint } from './analysis-report.ts'
import { isManualWorkspacePath } from './local-development.ts'

test('report status, generation and downloads use the correct conversation/manual proxy', () => {
  const conversation = analysisReportEndpoint('run', 'run with space', 'job/1')
  const manual = analysisReportEndpoint('dataset', '中文项目', 'job/1')
  assert.match(conversation, /^\/api\/v3\/runs\/run%20with%20space\/analysis-report\/job%2F1$/)
  assert.ok(!conversation.startsWith('/api/backend'), 'The manual proxy rejects conversation routes')
  const path = new URL(manual, 'http://localhost').pathname.replace('/api/backend/', '').split('/').map(decodeURIComponent)
  assert.ok(isManualWorkspacePath(path))
  assert.equal(new URL(manual, 'http://localhost').searchParams.get('job_id'), 'job/1')
})
