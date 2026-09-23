"use client"

import { useEffect, useMemo, useRef, useState } from "react"
import { artifactLabel } from "@/lib/presentation"
import { resolveChartDataFiles } from "@/lib/chart-data"
import { apiFetch, API_BASE } from "@/lib/api/config"
import { ETMAgentAPI, type AgentChartDataPayload, type AgentChartDataSource } from "@/lib/api/etm-agent"
import { MarkdownRenderer } from "@/components/markdown-renderer"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog"
import { useProjectDraft } from "@/lib/use-project-draft"

interface VisualizationTabProps { dataset: string; mode: string; modelName?: string; shouldLoad: boolean; selectedModel?: string }
interface VisualizationFile {
  name: string
  path: string
  url: string
  size: number
  type: string
}

interface VisualizationData {
  dataset: string
  model: string
  global_files: VisualizationFile[]
  topic_files: Record<string, VisualizationFile[]>
}


function boundedChartData(content: string): { content: string; truncated: boolean; totalCharacters: number } {
  const maximum = 480_000
  if (content.length <= maximum) return { content, truncated: false, totalCharacters: content.length }
  const half = Math.floor((maximum - 80) / 2)
  return {
    content: `${content.slice(0, half)}\n# ... middle rows omitted by frontend ...\n${content.slice(-half)}`,
    truncated: true,
    totalCharacters: content.length,
  }
}

async function buildChartDataPayload(data: VisualizationData, chart: VisualizationFile): Promise<AgentChartDataPayload> {
  const allFiles = [...data.global_files, ...Object.values(data.topic_files).flat()]
  const sourceFiles = resolveChartDataFiles(chart, allFiles)
  if (sourceFiles.length === 0) throw new Error(`图表 ${chart.name} 没有可验证的绘图数据文件`)
  const sources: AgentChartDataSource[] = await Promise.all(sourceFiles.map(async (source) => {
    const response = await fetch(
      `${API_BASE}/api/results/${encodeURIComponent(data.dataset)}/visualizations/file?model=${encodeURIComponent(data.model)}&path=${encodeURIComponent(source.path)}`,
      { credentials: "same-origin" },
    )
    if (!response.ok) throw new Error(`无法读取绘图数据 ${source.name}`)
    const bounded = boundedChartData(await response.text())
    return {
      name: source.name,
      path: source.path,
      format: source.name.toLocaleLowerCase().endsWith(".json") ? "json" : "csv",
      ...bounded,
    }
  }))
  return { chartName: artifactLabel(chart.name, "figure"), chartPath: chart.path, dataset: data.dataset, model: data.model, sources }
}

const imageSuffix = /\.(png|jpe?g|webp|svg)$/iu
const familyOf = (file: VisualizationFile) => file.path.replace(/\.(png|jpe?g|webp|svg|pdf)$/iu, "")
const labelOf = (file: VisualizationFile) => artifactLabel(file.name, /\.(csv|json)$/iu.test(file.name) ? "table" : "figure")
type Interpretation = { status: "loading" | "done" | "error"; text: string }

/** Shared native artifacts; one preview per figure, explicit evidence-backed interpretation. */
export function VisualizationTab({ dataset, shouldLoad, selectedModel = "theta" }: VisualizationTabProps) {
  const scope = dataset + ":" + selectedModel
  const [data, setData] = useState<VisualizationData | null>(null)
  const [error, setError] = useState("")
  const [loading, setLoading] = useState(false)
  const [query, setQuery] = useProjectDraft(`figures:${scope}:query`, "")
  const [page, setPage] = useProjectDraft(`figures:${scope}:page`, 0)
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [sending, setSending] = useState(false)
  const [preview, setPreview] = useState<VisualizationFile | null>(null)
  const [zoom, setZoom] = useState(1)
  const [interpretations, setInterpretations] = useProjectDraft<Record<string, Interpretation>>(`figures:${scope}:interpretations`, {})
  const [loadingInterpretations, setLoadingInterpretations] = useState<Record<string, boolean>>({})
  const pending = useRef(new Set<string>())

  useEffect(() => {
    if (!shouldLoad) return
    let cancelled = false
    setLoading(true); setError(""); setData(null); setSelected(new Set())
    void apiFetch<VisualizationData>(API_BASE, `/api/results/${encodeURIComponent(dataset)}/visualizations?model=${encodeURIComponent(selectedModel)}`)
      .then(value => { if (!cancelled) setData({ ...value, model: value.model || selectedModel }) })
      .catch(e => { if (!cancelled) setError(e instanceof Error ? e.message : "结果加载失败") })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [dataset, selectedModel, shouldLoad])

  const allFiles = useMemo(() => data ? [...data.global_files, ...Object.values(data.topic_files).flat()] : [], [data])
  const figures = useMemo(() => {
    const families = new Map<string, VisualizationFile>()
    for (const file of allFiles.filter(file => imageSuffix.test(file.name))) {
      const key = familyOf(file)
      if (!families.has(key) || file.name.endsWith(".png")) families.set(key, file)
    }
    return [...families.values()]
  }, [allFiles])
  const filtered = figures.filter(file => (labelOf(file) + file.path).toLowerCase().includes(query.toLowerCase()))
  const pages = Math.max(1, Math.ceil(filtered.length / 6))
  const visible = filtered.slice(Math.min(page, pages - 1) * 6, (Math.min(page, pages - 1) + 1) * 6)
  const interpret = async (file: VisualizationFile) => {
    const requestKey = scope + ":" + file.path
    if (!data || pending.current.has(requestKey)) return
    pending.current.add(requestKey)
    // Only finished responses are durable; refreshing must not restore a stuck spinner.
    setLoadingInterpretations(prev => ({ ...prev, [requestKey]: true }))
    try {
      const chart = await buildChartDataPayload(data, file)
      const result = await ETMAgentAPI.advisory(
        "请仅依据附件绘图数据，用中文简短解释这张图，引用具体来源文件名及主题编号。不要臆测趋势、显著性或因果；没有基线时不把指标绝对值评级为好坏。iRBO 是反向主题词排序重叠多样性，不是训练稳定性。结论仅限本次数据，不外推到未提供的总体。",
        { app_state: "manual_chart_interpretation", dataset: data.dataset, model: data.model }, [chart])
      setInterpretations(prev => ({ ...prev, [file.path]: { status: "done", text: result.message } }))
    } catch (e) {
      setInterpretations(prev => ({ ...prev, [file.path]: { status: "error", text: e instanceof Error ? e.message : "解读失败" } }))
    } finally {
      pending.current.delete(requestKey)
      setLoadingInterpretations(prev => { const next = { ...prev }; delete next[requestKey]; return next })
    }
  }
  const cite = async (files: VisualizationFile[]) => {
    if (!data || sending) return
    setSending(true); setError("")
    try {
      const payloads = await Promise.all(files.map(file => buildChartDataPayload(data, file)))
      window.dispatchEvent(new CustomEvent("theta:chart-data-to-chat", { detail: payloads }))
      setSelected(new Set())
    } catch (e) { setError(e instanceof Error ? e.message : "引用失败") }
    finally { setSending(false) }
  }
  if (loading) return <p role="status" className="p-8 text-sm text-slate-500">正在读取实际图表…</p>
  if (!data) return <p role="alert" className="p-5 text-sm text-red-600">{error || "暂无图表"}</p>
  return <div className="space-y-5">
    <div className="flex flex-wrap items-center justify-between gap-3">
      <div><h3 className="font-semibold">图表与绘图数据</h3><p className="mt-1 text-xs text-slate-500">共 {figures.length} 张图；同一图的 PNG、SVG、PDF 合并展示。解读仅在点击后生成。</p></div>
      <input aria-label="筛选图表" placeholder="搜索中文图名或主题…" className="rounded-lg border px-3 py-2 text-sm" value={query} onChange={e => { setQuery(e.target.value); setPage(0) }} />
    </div>
    {error && <p role="alert" className="text-sm text-red-600">{error}</p>}
    {selected.size > 0 && <Button disabled={sending} onClick={() => void cite(figures.filter(file => selected.has(file.path)))}>引用所选图表（{selected.size}）</Button>}
    <div className="grid grid-cols-1 gap-5 xl:grid-cols-2">
      {visible.map(file => {
        const interpretation: Interpretation | undefined = loadingInterpretations[scope + ":" + file.path]
          ? { status: "loading", text: "" } : interpretations[file.path]
        const variants = allFiles.filter(item => familyOf(item) === familyOf(file) && /\.(png|jpe?g|svg|webp|pdf)$/iu.test(item.name))
        return <article key={file.path} aria-label={labelOf(file)} className="overflow-hidden rounded-2xl border border-slate-200 bg-white">
          <header className="flex items-center gap-3 border-b bg-slate-50/60 px-4 py-3"><input type="checkbox" aria-label={`选择 ${labelOf(file)}`} checked={selected.has(file.path)} onChange={e => setSelected(prev => { const next = new Set(prev); if (e.target.checked && next.size < 8) next.add(file.path); else next.delete(file.path); return next })} /><h4 className="min-w-0 flex-1 text-sm font-medium">{labelOf(file)}</h4></header>
          <button className="block w-full cursor-zoom-in bg-white p-3" aria-label={`放大 ${labelOf(file)}`} onClick={() => { setPreview(file); setZoom(1) }}><img src={file.url} alt={labelOf(file)} loading="lazy" className="h-64 w-full object-contain" /></button>
          <div className="flex flex-wrap items-center gap-2 border-t px-4 py-3"><Button size="sm" variant="outline" disabled={sending} onClick={() => void cite([file])}>引用到咨询</Button><Button size="sm" variant="outline" disabled={interpretation?.status === "loading"} onClick={() => void interpret(file)}>{interpretation?.status === "loading" ? "正在解读…" : interpretation ? "重新解读" : "生成解读"}</Button><span className="ml-auto flex gap-2">{variants.map(item => <a key={item.path} href={item.url} download={item.name} className="text-xs text-blue-700">{item.type.toUpperCase()}</a>)}</span></div>
          {interpretation && <details open className="border-t bg-slate-50/60 p-4"><summary className="cursor-pointer text-sm font-medium">图表解读</summary><div className="mt-3 max-h-72 overflow-y-auto text-sm">{interpretation.status === "done" ? <MarkdownRenderer content={interpretation.text} /> : <p role="status" className={interpretation.status === "error" ? "text-red-600" : "text-slate-500"}>{interpretation.text || "正在读取绘图数据并生成解读…"}</p>}</div></details>}
        </article>
      })}
    </div>
    <div className="flex items-center justify-center gap-4"><Button variant="outline" disabled={page === 0} onClick={() => setPage(p => p - 1)}>上一页</Button><span className="text-sm">{Math.min(page, pages - 1) + 1} / {pages}</span><Button variant="outline" disabled={page >= pages - 1} onClick={() => setPage(p => p + 1)}>下一页</Button></div>
    <details className="rounded-xl border p-4"><summary className="cursor-pointer text-sm font-medium">绘图数据、交互图与其他文件</summary><p className="mt-2 text-xs text-slate-500">原始数据表与配置不作为图片预览。全部文件及模型、代码可在“导出”页下载。</p><ul className="mt-3 grid max-h-72 grid-cols-1 gap-2 overflow-y-auto md:grid-cols-2">{allFiles.filter(file => !/\.(png|jpe?g|webp|svg|pdf)$/iu.test(file.name)).map(file => <li key={file.path}><a className="text-sm text-blue-700" href={file.url} download={file.name}>{labelOf(file)} · {file.type.toUpperCase()}</a></li>)}</ul></details>
    <Dialog open={!!preview} onOpenChange={open => { if (!open) setPreview(null) }}><DialogContent className="flex max-h-[94dvh] flex-col sm:max-w-[94vw]"><DialogTitle>{preview ? labelOf(preview) : "图表预览"}</DialogTitle><div className="flex gap-2"><Button size="sm" variant="outline" onClick={() => setZoom(z => Math.max(.5, z - .25))}>缩小</Button><Button size="sm" variant="outline" onClick={() => setZoom(z => Math.min(4, z + .25))}>放大</Button><Button size="sm" variant="outline" onClick={() => setZoom(1)}>复原</Button><span className="self-center text-xs">{Math.round(zoom * 100)}%</span></div><div className="max-h-[74dvh] overflow-auto bg-slate-100 p-3">{preview && <img src={preview.url} alt={labelOf(preview)} style={{ width: `${zoom * 100}%`, maxWidth: "none" }} />}</div></DialogContent></Dialog>
  </div>
}
