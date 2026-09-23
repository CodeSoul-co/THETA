import assert from 'node:assert/strict'
import test from 'node:test'
import { isUploadedFileId, uploadManualFiles } from './manual-upload.ts'

const files = Array.from({ length: 4 }, (_, i) => ({ name: `${i}.csv`, size: 10 }))

test('只有有效上传回执可以启动训练', () => {
  for (const id of [null, undefined, '', 'undefined', '12abc', '1e2', '0x10', '4.0', 0, -1, 1.2, Number.MAX_SAFE_INTEGER + 1]) assert.equal(isUploadedFileId(id), false)
  for (const id of [1, '42']) assert.equal(isUploadedFileId(id), true)
})

test('成功上传保留每个文件回执，进度不倒退，最终为 100%', async () => {
  const progress: number[] = []
  const result = await uploadManualFiles(files, async (file, report) => {
    report(100); report(50)
    return { file_id: Number(file.name[0]) + 1 }
  }, value => progress.push(value))
  assert.deepEqual(result.map(r => r.fileId), ['1', '2', '3', '4'])
  assert.equal(progress.at(-1), 100)
  assert.ok(progress.every((value, index) => index === 0 || value >= progress[index - 1]))
})

test('登录失败不会开启下一批上传，当前批全部结束后才返回错误', async () => {
  const calls: string[] = []
  let completed = 0
  await assert.rejects(uploadManualFiles(files, async file => {
    calls.push(file.name)
    if (file.name === '0.csv') throw new Error('登录已过期')
    await new Promise(resolve => setTimeout(resolve, 10))
    completed++
    return { file_id: 1 }
  }, () => {}), /登录已过期/)
  assert.equal(calls.length, 3)
  assert.equal(completed, 2)
})

test('失败或无效文件编号不会产生完成回执，随后可以重新上传', async () => {
  await assert.rejects(uploadManualFiles([files[0]], async () => ({ file_id: 0 }), () => {}), /有效文件编号/)
  const retry = await uploadManualFiles([files[0]], async () => ({ file_id: 9 }), () => {})
  assert.equal(retry[0].fileId, '9')
  await assert.rejects(uploadManualFiles([], async () => ({ file_id: 9 }), () => {}), /选择数据/)
})
