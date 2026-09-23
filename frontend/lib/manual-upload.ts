export function isUploadedFileId(value: unknown): boolean {
  return (typeof value === 'number' || typeof value === 'string')
    && /^\d+$/.test(String(value).trim()) && Number.isSafeInteger(Number(value)) && Number(value) > 0
}

/** 每批上传全部落定后才继续；失败不会留下后台请求干扰重试。 */
export async function uploadManualFiles<T extends { name: string; size: number }>(
  files: T[],
  upload: (file: T, progress: (percent: number) => void) => Promise<{ file_id: number }>,
  onProgress: (percent: number) => void,
) {
  if (!files.length) throw new Error('请先选择数据文件')
  const progress = files.map(() => 0)
  const receipts: { name: string; fileId: string; size: number }[] = []
  for (let offset = 0; offset < files.length; offset += 3) {
    const batch = await Promise.allSettled(files.slice(offset, offset + 3).map(async (file, index) => {
      const result = await upload(file, percent => {
        progress[offset + index] = Math.max(progress[offset + index], Math.min(100, Math.max(0, percent)))
        onProgress(Math.round(progress.reduce((sum, value) => sum + value, 0) / files.length * .95))
      })
      if (!isUploadedFileId(result.file_id)) throw new Error('上传服务未返回有效文件编号，请重新上传。')
      return { name: file.name, fileId: String(result.file_id), size: file.size }
    }))
    const failed = batch.find(result => result.status === 'rejected')
    if (failed?.status === 'rejected') throw failed.reason
    for (const result of batch) if (result.status === 'fulfilled') receipts.push(result.value)
  }
  onProgress(100)
  return receipts
}
