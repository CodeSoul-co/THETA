"use client"

import { useRef, useState } from 'react'
import { toast } from 'sonner'
import { SimpleETMAPI } from '@/lib/api/backend'
import { ETMAgentAPI } from '@/lib/api/etm-agent'
import { DATASET_ACCEPT, datasetFilesError } from '@/lib/dataset-files'
import { splitRoles, splitError, type DataSplit, type SplitRole, type SplitSource } from '@/lib/data-split'

const labels = { train: '训练集', validation: '验证集', test: '测试集' }
export function DataSplitCard({ value, onChange, dataset, files, onUploaded, onBusy }: {
  value: DataSplit; onChange: (value: DataSplit) => void; dataset: string;
  files: { fileId: string; name: string; size: number }[];
  onUploaded: (file: { fileId: string; name: string; size: number }) => void;
  onBusy: (busy: boolean) => void;
}) {
  const latest = useRef(value); latest.current = value
  const [busy, setBusy] = useState<SplitRole | null>(null)
  const [columns, setColumns] = useState<Partial<Record<SplitRole, string[]>>>({})
  const [error, setError] = useState<string>()
  const [previews, setPreviews] = useState<Partial<Record<SplitRole, string>>>({})
  const updateSource = (role: SplitRole, source: SplitSource) => onChange({ ...latest.current, sources: { ...latest.current.sources, [role]: source } })
  const select = async (role: SplitRole, id: string, file?: File) => {
    setBusy(role); onBusy(true); setError(undefined)
    try {
      if (file) {
        const problem = datasetFilesError([file]); if (problem) throw new Error(problem)
        const receipt = await SimpleETMAPI.uploadDataset(file, dataset)
        id = String(receipt.file_id)
        onUploaded({ fileId: id, name: file.name, size: file.size })
      }
      const preview = await ETMAgentAPI.getDatasetPreview(dataset, id)
      const names = preview.columns
      const textColumn = preview.textColumn || names.find(name => /^(text|content|正文|文本)$/i.test(name)) || names[0] || ''
      setColumns(prev => ({ ...prev, [role]: names }))
      setPreviews(prev => ({ ...prev, [role]: `${preview.totalRecords ?? preview.rows.length} 条记录 · ${String(preview.rows[0]?.[names.indexOf(textColumn)] ?? '').slice(0, 160)}` }))
      updateSource(role, { fileId: id, textColumn, covariates: [] })
      if (file) toast.success(`${labels[role]}上传成功`)
    } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)) }
    finally { setBusy(null); onBusy(false) }
  }
  return <fieldset disabled={!!busy} className="space-y-4 rounded-xl border border-indigo-100 bg-indigo-50/40 p-4">
    <label className="flex items-center gap-3 font-medium"><input type="checkbox" checked={value.enabled} onChange={e => onChange({ ...value, enabled: e.target.checked })} />自定义训练 / 验证 / 测试集划分</label>
    {!value.enabled ? <p className="text-sm text-slate-600">默认随机按 70% / 30% 划分训练集和验证集，再使用全量数据测试。全量测试包含训练数据，不代表独立测试成绩。</p> : <>
      <label className="block text-sm">划分方式<select className="ml-3 rounded-lg border bg-white p-2" value={value.mode} onChange={e => onChange({ ...value, mode: e.target.value as DataSplit['mode'] })}><option value="ratio">按比例划分</option><option value="upload">分别上传三份数据</option></select></label>
      {value.mode === 'ratio' ? <>
        <div className="grid gap-3 sm:grid-cols-3">{splitRoles.map((role, i) => <label key={role} className="space-y-1 text-sm">{labels[role]}（%）<input type="number" min="0" max="100" step="1" className="block w-full rounded-lg border bg-white p-2" value={Math.round(value.ratios[i] * 10000) / 100} onChange={e => onChange({ ...value, ratios: value.ratios.map((v, index) => index === i ? Number(e.target.value) / 100 : v) })} /></label>)}</div>
        <label className="block text-sm">记录顺序<select className="ml-3 rounded-lg border bg-white p-2" value={value.method} onChange={e => onChange({ ...value, method: e.target.value as DataSplit['method'] })}><option value="random">随机划分</option><option value="sequential">按原始顺序划分</option></select></label>
        {value.method === 'random' && <label className="block text-sm">随机种子<input type="number" min="0" max="4294967295" className="ml-3 w-36 rounded-lg border bg-white p-2" value={value.seed} onChange={e => onChange({ ...value, seed: Number(e.target.value) })} /></label>}
        <p className="text-xs text-slate-500">顺序划分依次取训练、验证、测试记录；随机划分使用固定种子，同一份数据和设置可复现。</p>
      </> : <div className="space-y-3">{splitRoles.map(role => {
        const source = value.sources[role]
        const names = columns[role] ?? (source ? [...new Set([source.textColumn, source.timeColumn, source.labelColumn, ...(source.covariates ?? [])].filter((v): v is string => !!v))] : [])
        return <div key={role} className="space-y-3 rounded-xl border bg-white p-4"><h4 className="text-sm font-semibold">{labels[role]}</h4>
          <div className="flex flex-wrap gap-3"><select aria-label={`${labels[role]}文件`} className="min-w-0 flex-1 rounded-lg border p-2 text-sm" value={source?.fileId ?? ''} onChange={e => { if (e.target.value) void select(role, e.target.value) }}><option value="">选择已上传文件</option>{files.map(file => <option key={file.fileId} value={file.fileId}>{file.name}</option>)}</select><label className="cursor-pointer rounded-lg border px-3 py-2 text-sm text-indigo-700">{busy === role ? '读取中…' : '上传文件'}<input className="sr-only" type="file" accept={DATASET_ACCEPT} onChange={e => { const file = e.target.files?.[0]; if (file) void select(role, '', file); e.target.value = '' }} /></label></div>
          {source && <><div className="grid gap-3 sm:grid-cols-3">{(['textColumn', 'timeColumn', 'labelColumn'] as const).map(field => <label key={field} className="text-xs">{{ textColumn: '正文列', timeColumn: '时间列（可选）', labelColumn: '标签列（可选）' }[field]}<select className="mt-1 block w-full rounded-lg border p-2 text-sm" value={source[field] ?? ''} onChange={e => updateSource(role, { ...source, [field]: e.target.value || undefined })}>{field !== 'textColumn' && <option value="">不使用</option>}{names.map(name => <option key={name}>{name}</option>)}</select></label>)}</div><label className="block text-xs">元数据列（STM，多选）<select multiple className="mt-1 block w-full rounded-lg border p-2" value={source.covariates ?? []} onChange={e => updateSource(role, { ...source, covariates: Array.from(e.target.selectedOptions, option => option.value) })}>{names.filter(name => name !== source.textColumn).map(name => <option key={name}>{name}</option>)}</select></label><p className="break-words text-xs text-slate-500">{previews[role] ?? '已保存文件与列选择，可重新选择文件刷新预览。'}</p></>}
        </div>
      })}<p className="text-xs text-slate-500">支持与主数据相同的表格和文档格式。文本、PDF、Word 按正文分段；全量结果为这三份数据的合并结果。</p></div>}
    </>}
    {(error || splitError(value)) && <p role="alert" className="text-sm text-red-700">{error || splitError(value)}</p>}
  </fieldset>
}
