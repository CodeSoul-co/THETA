'use client'

import { useEffect, useState } from 'react'
import { Button } from '@/components/ui/button'
import { ETMAgentAPI } from '@/lib/api/etm-agent'
import { isTextDocument } from '@/lib/dataset-input'
import { datasetFilesWarning } from '@/lib/dataset-files'

export type UploadedDatasetFile = { name: string; fileId: string; size: number; inputKind?: string; sourceFileIds?: number[] }

export function UploadedDatasetFiles({ files, dataset, disabled, onCombine, onDelete }: {
  files: UploadedDatasetFile[]; dataset: string; disabled: boolean
  onCombine: (files: UploadedDatasetFile[], textColumns: Record<string, string>) => Promise<void>
  onDelete: (file: UploadedDatasetFile) => Promise<void>
}) {
  const originals = files.filter(file => !file.sourceFileIds?.length)
  const [selected, setSelected] = useState<string[]>(originals.map(file => file.fileId))
  const [known, setKnown] = useState<string[]>(originals.map(file => file.fileId))
  const [columns, setColumns] = useState<Record<string, string[]>>({})
  const [textColumns, setTextColumns] = useState<Record<string, string>>({})
  const [working, setWorking] = useState(false)
  const [error, setError] = useState('')
  const [page, setPage] = useState(0)
  useEffect(() => {
    const ids = files.filter(file => !file.sourceFileIds?.length).map(file => file.fileId)
    setSelected(previous => [...previous.filter(id => ids.includes(id)), ...ids.filter(id => !known.includes(id))])
    setKnown(ids)
    setPage(previous => Math.min(previous, Math.max(0, Math.ceil(files.length / 50) - 1)))
  }, [files])
  const chosen = originals.filter(file => selected.includes(file.fileId))
  const warning = datasetFilesWarning(chosen)
  async function combine() {
    setWorking(true); setError('')
    try {
      const mapping = { ...textColumns }
      // Read sequentially so a folder with many tables doesn't launch an unbounded number of workers.
      let missing = false
      for (const file of chosen) {
        if (file.inputKind === 'text' || isTextDocument(file.name)) continue
        if (!columns[file.fileId]) {
          const preview = await ETMAgentAPI.getDatasetPreview(dataset, file.fileId)
          const names = preview.columns
          setColumns(previous => ({ ...previous, [file.fileId]: names }))
          const candidates = names.filter(name => /^(text|content|body|正文|文本|内容)$/i.test(name))
          mapping[file.fileId] ||= candidates.length === 1 ? candidates[0] : names.length === 1 ? names[0] : ''
        }
        if (!mapping[file.fileId]) missing = true
      }
      setTextColumns(mapping)
      if (missing) throw new Error('请在下方为表格选择正文列，再点击合并。每个文件可以使用不同的列名。')
      await onCombine(chosen, mapping)
    } catch (cause) { setError(cause instanceof Error ? cause.message : '合并失败，请重试') }
    finally { setWorking(false) }
  }
  return <div className="space-y-3 rounded-xl border p-4">
    <h3 className="text-sm font-semibold">已上传文件 · {originals.length} 个源文件</h3>
    <p className="text-xs text-slate-500">选择多个文件合并分析，也可在“本次分析数据”中选择单个文件。删除源文件会移除包含它的合并数据，已有训练记录与结果保留。</p>
    <div className="flex flex-wrap items-center gap-2">
      <Button variant="outline" disabled={disabled || working} onClick={() => setSelected(originals.map(file => file.fileId))}>全选</Button>
      <Button variant="outline" disabled={disabled || working} onClick={() => setSelected([])}>取消全选</Button>
      <Button disabled={disabled || working || !chosen.length} onClick={() => void combine()}>{working ? '正在处理…' : `合并所选 ${chosen.length} 个文件分析`}</Button>
    </div>
    {warning && <p role="status" className="rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900">{warning}</p>}
    {error && <p role="alert" className="text-sm text-red-700">{error}</p>}
    <div className="max-h-80 space-y-2 overflow-y-auto">{files.slice(page * 50, (page + 1) * 50).map(file => <div key={file.fileId} className="rounded-lg bg-slate-50 p-3 text-sm">
      <div className="flex items-center gap-2"><label className="flex min-w-0 flex-1 items-center gap-2">{!file.sourceFileIds?.length && <input type="checkbox" aria-label={`纳入分析：${file.name}`} checked={selected.includes(file.fileId)} disabled={disabled || working} onChange={e => setSelected(previous => e.target.checked ? [...previous, file.fileId] : previous.filter(id => id !== file.fileId))} />}<span className="break-all">{file.name}{file.sourceFileIds?.length ? `（合并 ${file.sourceFileIds.length} 个文件）` : ''} · {(file.size / 1024 / 1024).toFixed(1)} MB</span></label>
      <Button variant="outline" size="sm" disabled={disabled || working} aria-label={`删除文件：${file.name}`} onClick={async () => { setWorking(true); setError(''); try { await onDelete(file) } catch (cause) { setError(cause instanceof Error ? cause.message : '删除失败') } finally { setWorking(false) } }}>删除</Button></div>
      {columns[file.fileId] && <label className="mt-2 block">正文列<select aria-label={`${file.name} 的正文列`} className="ml-2 rounded border p-1" value={textColumns[file.fileId] || ''} onChange={e => setTextColumns(previous => ({ ...previous, [file.fileId]: e.target.value }))}><option value="">请选择正文列</option>{columns[file.fileId].map(name => <option key={name} value={name}>{name}</option>)}</select></label>}
    </div>)}</div>
    {files.length > 50 && <div className="flex items-center gap-3 text-sm"><Button variant="outline" disabled={page === 0} onClick={() => setPage(page - 1)}>上一页</Button><span>{page + 1} / {Math.ceil(files.length / 50)}</span><Button variant="outline" disabled={(page + 1) * 50 >= files.length} onClick={() => setPage(page + 1)}>下一页</Button></div>}
  </div>
}
