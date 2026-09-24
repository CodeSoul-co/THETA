import test from 'node:test'
import assert from 'node:assert/strict'
import { DATASET_ACCEPT, datasetFilesError } from './dataset-files.ts'

test('picker and drop accept the same Unicode filenames and supported formats', () => {
  for (const suffix of DATASET_ACCEPT.split(',')) assert.equal(datasetFilesError([{ name: `中文 空格 ${suffix.toUpperCase()}`, size: 12 }]), undefined)
})
test('invalid replacement batches do not silently discard files', () => {
  assert.match(datasetFilesError([{ name: 'a.csv', size: 10 }, { name: 'b.exe', size: 10 }])!, /不支持/)
  assert.match(datasetFilesError([{ name: 'a.csv', size: 10 }, { name: 'b.csv', size: 10 }], 'zh-CN', true)!, /一个/)
  assert.match(datasetFilesError([{ name: 'empty.docx', size: 0 }])!, /空文件/)
  assert.match(datasetFilesError([{ name: 'large.xlsx', size: 200 * 1024 * 1024 + 1 }])!, /200 MiB/)
  assert.equal(datasetFilesError([{ name: 'max.xlsx', size: 200 * 1024 * 1024 }]), undefined)
  assert.match(datasetFilesError([])!, /文件夹/)
  assert.match(datasetFilesError([{ name: 'bad.exe', size: 3 }], 'en')!, /not supported/)
})

test('folder drops traverse nested directories and every directory-reader batch', async () => {
  const { droppedFiles, documentCollectionError } = await import('./dataset-files.ts')
  const fileEntry = (name: string, fullPath: string) => ({ name, fullPath, isFile: true, isDirectory: false, file: (ok: (file: File) => void) => ok(new File(['实际正文'], name)) })
  const nested = { name: 'nested', isFile: false, isDirectory: true, createReader: () => {
    let count = 0
    return { readEntries: (ok: (entries: unknown[]) => void) => ok(count++ === 0 ? [fileEntry('two.md', '/folder/nested/two.md')] : []) }
  } }
  const directory = { name: 'folder', isFile: false, isDirectory: true, createReader: () => {
    const batches = [[fileEntry('one.txt', '/folder/one.txt')], [nested], []]
    return { readEntries: (ok: (entries: unknown[]) => void) => ok(batches.shift() ?? []) }
  } }
  const files = await droppedFiles({ items: [{ kind: 'file', webkitGetAsEntry: () => directory }], files: [] } as unknown as DataTransfer)
  assert.deepEqual(files.map(file => file.webkitRelativePath), ['folder/one.txt', 'folder/nested/two.md'])
  assert.equal(documentCollectionError(files), undefined)
  assert.match(documentCollectionError([{ name: 'table.xlsx', size: 5 }])!, /表格请单独上传/)
})
