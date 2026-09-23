import assert from 'node:assert/strict'
import test from 'node:test'
import { allowLocalSession, isLocalRequestOrigin, isManualWorkspacePath } from './local-development.ts'

test('仅本机开发模式允许免登录，生产和远程后端永远拒绝', () => {
  const local = 'http://127.0.0.1:4320/api/backend/api/auth/me'
  assert.equal(allowLocalSession(local, 'http://127.0.0.1:8000', 'development', undefined), true)
  assert.equal(allowLocalSession(local, 'http://localhost:8000', 'development', 'true'), true)
  assert.equal(allowLocalSession(local, 'http://[::1]:8000', 'development', 'true'), true)
  assert.equal(allowLocalSession(local, 'http://127.0.0.1:8000', 'production', 'true'), false)
  assert.equal(allowLocalSession(local, 'http://127.0.0.1:8000', 'development', 'false'), false)
  assert.equal(allowLocalSession(local, 'https://theta.example.com', 'development', 'true'), false)
  assert.equal(allowLocalSession('https://theta.example.com', 'http://127.0.0.1:8000', 'development', 'true'), false)
  assert.equal(allowLocalSession(local, 'http://127.0.0.1.evil.example', 'development', 'true'), false)
  assert.equal(allowLocalSession(local, 'file:///tmp/backend', 'development', 'true'), false)
})

test('本地来源使用浏览器 Host 校验，拒绝跨站和 DNS 重绑定', () => {
  assert.equal(isLocalRequestOrigin('http:', '127.0.0.1:4320', 'http://127.0.0.1:4320', 'same-origin'), true)
  assert.equal(isLocalRequestOrigin('http:', 'localhost:4320', 'http://localhost:4320', 'same-origin'), true)
  assert.equal(isLocalRequestOrigin('http:', '127.0.0.1:4320', null, null), true)
  assert.equal(isLocalRequestOrigin('http:', '127.0.0.1:4320', 'http://evil.example', 'cross-site'), false)
  assert.equal(isLocalRequestOrigin('http:', 'evil.example:4320', null, null), false)
  assert.equal(isLocalRequestOrigin('http:', '127.0.0.1:4320', 'http://127.0.0.1:9000', 'same-site'), false)
  assert.equal(isLocalRequestOrigin('http:', 'user@localhost:4320', null, null), false)
})

test('开源手动工作台开放数据接口，账号、管理和路径穿越保持禁用', () => {
  for (const route of ['config', 'health', 'api/projects', 'api/upload', 'api/train/jobs', 'api/results/data/catalog', 'api/runtime/config', 'api/stopwords/default']) {
    assert.equal(isManualWorkspacePath(route.split('/')), true, route)
  }
  for (const route of ['api/auth/me', 'api/auth/login', 'api/admin/users', 'api/oss/sts-token', 'api/projects/../auth', 'api/projects/%2e%2e', 'api//projects']) {
    assert.equal(isManualWorkspacePath(route.split('/')), false, route)
  }
  assert.equal(isManualWorkspacePath(['api', 'projects', '../auth']), false)
  assert.equal(isManualWorkspacePath(['api', 'projects', '..\\auth']), false)
})
