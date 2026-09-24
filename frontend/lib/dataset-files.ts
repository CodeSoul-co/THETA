export const DATASET_ACCEPT = '.csv,.tsv,.txt,.md,.xlsx,.xls,.json,.jsonl,.ndjson,.parquet,.pdf,.docx'
const extensions = new Set(DATASET_ACCEPT.split(','))

export function datasetFilesError(files: readonly { name: string; size: number }[], locale = 'zh-CN', single = false): string | undefined {
  const zh = locale === 'zh-CN'
  if (!files.length) return zh ? '请选择数据文件。文件夹请通过“选择文件夹”上传。' : 'Choose a data file. Use “Choose folder” to upload a folder.'
  if (single && files.length > 1) return zh ? '每次请选择一个数据文件。' : 'Choose one dataset file at a time.'
  for (const file of files) {
    if (!extensions.has(file.name.slice(file.name.lastIndexOf('.')).toLowerCase())) return zh ? `不支持“${file.name}”。请选择 CSV、Excel、JSON、Parquet、PDF、DOCX 或文本文件。` : `“${file.name}” is not supported. Choose CSV, Excel, JSON, Parquet, PDF, DOCX or text.`
    if (file.size > 200 * 1024 * 1024) return zh ? `“${file.name}”超过单文件 200 MiB 限制。` : `“${file.name}” exceeds the 200 MiB per-file limit.`
    if (!file.size) return zh ? `“${file.name}”是空文件，请选择包含内容的数据文件。` : `“${file.name}” is empty. Choose a file with data.`
  }
}

export const isDocumentCollection = (files: readonly File[]) => files.length > 1 || files.some(file => !!file.webkitRelativePath)
export function documentCollectionError(files: readonly { name: string; size: number }[], locale = 'zh-CN') {
  const problem = datasetFilesError(files, locale)
  if (problem) return problem
  const zh = locale === 'zh-CN'
  if (files.length > 500 || files.reduce((size, file) => size + file.size, 0) > 200 * 1024 * 1024) return zh ? '文件夹最多 500 个文件，总大小不超过 200 MiB，请分批上传。' : 'Upload up to 500 files and 200 MiB in total per collection.'
  if (files.some(file => !/\.(txt|md|pdf|docx)$/iu.test(file.name))) return zh ? '文件夹或多文档合集支持 TXT、Markdown、PDF、DOCX。表格请单独上传并选择正文列。' : 'Document collections support TXT, Markdown, PDF and DOCX. Upload tables separately to select their text columns.'
}

/** Read all directory-reader batches; Chromium may return only 100 entries per call. */
export async function droppedFiles(transfer: DataTransfer): Promise<File[]> {
  const entries = Array.from(transfer.items ?? []).filter(item => item.kind === 'file').map(item => item.webkitGetAsEntry?.()).filter((entry): entry is FileSystemEntry => !!entry)
  if (!entries.some(entry => entry.isDirectory)) return Array.from(transfer.files)
  const files: File[] = []
  let entriesSeen = 0
  let total = 0
  async function visit(entry: FileSystemEntry, depth = 0) {
    if (++entriesSeen > 2000 || depth > 20) throw new Error('文件夹层级或文件数量过多，请分批上传。')
    if (entry.name.startsWith('.')) return
    if (entry.isFile) {
      const file = await new Promise<File>((resolve, reject) => (entry as FileSystemFileEntry).file(resolve, reject))
      total += file.size
      if (files.length >= 500 || total > 200 * 1024 * 1024) throw new Error('文件夹最多 500 个文件，总大小不超过 200 MiB。')
      Object.defineProperty(file, 'webkitRelativePath', { value: entry.fullPath.replace(/^\//, '') })
      files.push(file)
    } else if (entry.isDirectory) {
      const reader = (entry as FileSystemDirectoryEntry).createReader()
      while (true) {
        const batch = await new Promise<FileSystemEntry[]>((resolve, reject) => reader.readEntries(resolve, reject))
        if (!batch.length) break
        for (const child of batch) await visit(child, depth + 1)
      }
    }
  }
  for (const entry of entries) await visit(entry)
  return files
}
