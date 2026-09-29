export const DATASET_ACCEPT = '.csv,.tsv,.txt,.md,.xlsx,.xls,.json,.jsonl,.ndjson,.parquet,.pdf,.docx'
const extensions = new Set(DATASET_ACCEPT.split(','))

export function datasetFilesError(files: readonly { name: string; size: number }[], locale = 'zh-CN', single = false): string | undefined {
  const zh = locale === 'zh-CN'
  if (!files.length) return zh ? '请选择数据文件。文件夹请通过“选择文件夹”上传。' : 'Choose a data file. Use “Choose folder” to upload a folder.'
  if (single && files.length > 1) return zh ? '每次请选择一个数据文件。' : 'Choose one dataset file at a time.'
  for (const file of files) {
    if (!extensions.has(file.name.slice(file.name.lastIndexOf('.')).toLowerCase())) return zh ? `不支持“${file.name}”。请选择 CSV、Excel、JSON、Parquet、PDF、DOCX 或文本文件。` : `“${file.name}” is not supported. Choose CSV, Excel, JSON, Parquet, PDF, DOCX or text.`
    if (!file.size) return zh ? `“${file.name}”是空文件，请选择包含内容的数据文件。` : `“${file.name}” is empty. Choose a file with data.`
  }
}

export const isDocumentCollection = (files: readonly File[]) => files.length > 1 || files.some(file => !!file.webkitRelativePath)
export function documentCollectionError(files: readonly { name: string; size: number }[], locale = 'zh-CN') {
  const problem = datasetFilesError(files, locale)
  if (problem) return problem

}

/** Read all directory-reader batches; Chromium may return only 100 entries per call. */
export async function droppedFiles(transfer: DataTransfer): Promise<File[]> {
  const entries = Array.from(transfer.items ?? []).filter(item => item.kind === 'file').map(item => item.webkitGetAsEntry?.()).filter((entry): entry is FileSystemEntry => !!entry)
  if (!entries.some(entry => entry.isDirectory)) return Array.from(transfer.files)
  const files: File[] = []
  async function visit(entry: FileSystemEntry) {
    if (entry.name === '.DS_Store' || entry.name === '__MACOSX' || entry.name.startsWith('._')) return
    if (entry.isFile) {
      const file = await new Promise<File>((resolve, reject) => (entry as FileSystemFileEntry).file(resolve, reject))
      Object.defineProperty(file, 'webkitRelativePath', { value: entry.fullPath.replace(/^\//, '') })
      files.push(file)
    } else if (entry.isDirectory) {
      const reader = (entry as FileSystemDirectoryEntry).createReader()
      while (true) {
        const batch = await new Promise<FileSystemEntry[]>((resolve, reject) => reader.readEntries(resolve, reject))
        if (!batch.length) break
        for (const child of batch) await visit(child)
      }
    }
  }
  for (const entry of entries) await visit(entry)
  return files
}

/** Warnings never reject a batch. Unsupported folder entries are reported separately. */
export function datasetFilesWarning(files: readonly { size: number }[], locale = 'zh-CN') {
  if (files.length <= 100 && files.reduce((sum, file) => sum + file.size, 0) <= 20 * 1024 * 1024) return undefined
  return locale === 'zh-CN' ? `本次共 ${files.length} 个文件，读取、合并和分析可能耗时较长，请保持页面打开。文件大小和数量不设固定上限。` : `${files.length} files: reading, merging and analysis may take longer. Keep this page open. No fixed file size or count limit.`
}
export function folderDatasetFiles(files: File[]) {
  const supported = files.filter(file => extensions.has(file.name.slice(file.name.lastIndexOf('.')).toLowerCase()) && file.size > 0)
  const accepted = new Set(supported)
  return { supported, skipped: files.filter(file => !accepted.has(file)) }
}
