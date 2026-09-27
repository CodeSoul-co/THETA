"use client"
import { useEffect, useState } from 'react'
import { fieldLabel } from '@/lib/presentation'

type Group = { count: number; metrics: Record<string, unknown>; topicProportions: number[] }
type Report = { testUsesAllData: boolean; method: string; mode: string; seed: number; groups: Record<string, Group> }
const labels: Record<string, string> = { all: '全量数据', train: '训练集', validation: '验证集', test: '测试集' }
function numericMetrics(value: Record<string, unknown>, prefix = ''): [string, number][] {
  return Object.entries(value).flatMap(([key, item]): [string, number][] => typeof item === 'number' && Number.isFinite(item) ? [[prefix + key, item]] : item && typeof item === 'object' && !Array.isArray(item) ? numericMetrics(item as Record<string, unknown>, prefix + key + '.') : [])
}
export function SplitResults({ files }: { files: { name: string; url: string }[] }) {
  const url = files.find(file => /(?:^|\/)split_results\.json$/.test(file.name))?.url
  const [report, setReport] = useState<Report>()
  const [error, setError] = useState('')
  const [role, setRole] = useState('all')
  useEffect(() => {
    const controller = new AbortController()
    setReport(undefined); setError(''); setRole('all')
    if (url) void fetch(url, { signal: controller.signal }).then(async response => {
      if (!response.ok) throw new Error('无法读取数据集结果，请刷新重试')
      const data = await response.json()
      if (!data.groups?.all) throw new Error('数据集结果格式不完整')
      setReport(data)
    }).catch(cause => { if (!controller.signal.aborted) setError(String(cause.message ?? cause)) })
    return () => controller.abort()
  }, [url])
  if (!url) return <p className="mt-4 rounded-xl border bg-white p-5 text-sm text-slate-500">这批历史结果未记录数据集划分。重新运行分析后，可分别查看全量、训练、验证和测试结果。</p>
  if (error) return <p role="alert" className="mt-4 text-red-700">{error}</p>
  if (!report) return <p role="status" className="mt-4 text-sm">正在读取各数据集结果…</p>
  const group = report.groups[role]
  const downloads = files.filter(file => file.name.includes(`/splits/${role}/`) || file.name.startsWith(`splits/${role}/`))
  return <section className="mt-4 space-y-4 rounded-2xl border bg-white p-5">
    <div className="flex flex-wrap gap-2" role="tablist" aria-label="结果数据范围">{Object.entries(labels).map(([key, label]) => <button key={key} role="tab" aria-selected={role === key} className={`rounded-lg px-4 py-2 text-sm ${role === key ? 'bg-indigo-600 text-white' : 'bg-slate-100 text-slate-700'}`} onClick={() => setRole(key)}>{label} · {report.groups[key]?.count ?? 0}</button>)}</div>
    {report.testUsesAllData && <p className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">本次采用默认 7:3 训练/验证划分，测试使用全量数据，包含训练集与验证集。因此测试结果与全量结果口径相同，不是独立留出测试。</p>}
    <p className="text-sm text-slate-600">{labels[role]}：{group.count} 条有效正文。{report.mode === 'upload' ? '使用独立上传的数据。' : report.method === 'sequential' ? '按原始顺序划分。' : `随机划分，种子 ${report.seed}。`}所有分组使用同一个已训练模型。</p>
    <div className="grid gap-5 lg:grid-cols-2"><div><h3 className="mb-3 font-medium">评估指标</h3><div className="max-h-96 overflow-auto"><table className="w-full text-sm"><thead><tr className="text-left"><th className="p-2">指标</th><th className="p-2">数值</th></tr></thead><tbody>{numericMetrics(group.metrics).map(([key, value]) => <tr key={key} className="border-t"><td className="p-2">{fieldLabel(key)}</td><td className="p-2 tabular-nums">{value.toLocaleString(undefined, { maximumFractionDigits: 5 })}</td></tr>)}</tbody></table></div></div><div><h3 className="mb-3 font-medium">平均主题权重</h3><div className="max-h-96 space-y-2 overflow-auto">{group.topicProportions.map((value, i) => <div key={i} className="flex items-center gap-3 text-sm"><span className="w-16 shrink-0">主题 {i}</span><div className="h-2 flex-1 rounded bg-slate-100"><div className="h-full rounded bg-indigo-500" style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }} /></div><span className="w-20 text-right tabular-nums">{value.toFixed(4)}</span></div>)}</div></div></div>
    {group.metrics.unavailable && typeof group.metrics.unavailable === 'object' && Object.keys(group.metrics.unavailable).length > 0 ? <details className="rounded-lg bg-slate-50 p-3 text-sm"><summary className="cursor-pointer">部分指标不可用，查看原因</summary>{Object.entries(group.metrics.unavailable as Record<string, unknown>).map(([key, reason]) => <p key={key} className="mt-2">{fieldLabel(key)}：{String(reason)}</p>)}</details> : null}
    <div className="flex flex-wrap gap-3">{downloads.filter(file => /documents\.csv$|theta\.npy$|metrics.*\.json$/.test(file.name)).map(file => <a key={file.name} href={file.url} download className="rounded-lg border px-3 py-2 text-sm text-indigo-700">下载{file.name.endsWith('.csv') ? '逐条文档结果' : file.name.endsWith('.npy') ? '主题矩阵' : '评估指标'}</a>)}</div>
  </section>
}
