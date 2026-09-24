"use client"

import { ExecutionLog } from "@/components/project/execution-log"

import { useProjectDraft } from "@/lib/use-project-draft"

import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { shouldRefreshResultCatalog, resultReadinessNotice } from "@/lib/result-readiness"
import { artifactLabel, fieldLabel, phaseLabel, statusLabel, systemText, advisoryErrorMessage } from "@/lib/presentation"
import { resolveChartDataFiles } from "@/lib/chart-data"
import { MarkdownRenderer } from "@/components/markdown-renderer"
import { ETMAgentAPI, type AgentChartDataPayload } from "@/lib/api/etm-agent"
import { listDatasets, type WebResultCatalog } from "@/components/theta-workbench/api/client"
import { loadResultCatalog, resultScope, resultProjectName, resultDatasetName, type ResultSource } from "./result-source"
import { BarChart3, CheckCircle2, ChevronRight, Clock3, Database, Download, FileText, ImageIcon, Layers3, Loader2, MessageSquare, Package, Quote, RefreshCw, Sparkles } from "lucide-react"

type ResultFile = { name: string; kind: string; url: string }
type ResultItem = WebResultCatalog["results"][number]

const FIGURE_SUFFIX = /\.(png|jpe?g|webp|gif|svg|pdf)$/iu

/** 按“这张图是干什么用的”分组，避免一次铺开上百张图。 */
const CATEGORIES: Array<{ key: string; label: string; test: RegExp }> = [
  { key: "time", label: "时间演化", test: /by year|over time|evolution|high-frequency|year_topic|演化|演变|年度|高频词/iu },
  { key: "words", label: "主题词", test: /word ?cloud|wordcloud|word distribution|salient|topic \d+|word_counts|词云|词汇|词分布|词语分布/iu },
  { key: "proportion", label: "主题占比", test: /proportion|strength|占比|强度/iu },
  { key: "relation", label: "主题关系", test: /network|correlation|similarity|dendrogram|heatmap|distance|sankey|网络|相关性|相似度|热力图|距离|桑基/iu },
  { key: "document", label: "文档投影", test: /cluster|umap|projection|outlier|文档|投影|离群/iu },
  { key: "quality", label: "质量指标", test: /coherence|exclusivity|metric|evaluation|loss|kl|一致性|排他性|指标|损失/iu },
]
const categoryOf = (family: string) => CATEGORIES.find((category) => category.test.test(basename(family)))?.key ?? "other"

const basename = (name: string) => name.split("/").pop() ?? name
const familyOf = (file: ResultFile) => file.name.replace(FIGURE_SUFFIX, "")
const FILE_LABEL: Record<string, string> = { figure: "图形", table: "数据表", matrix: "矩阵", report: "报告", artifact: "文件" }

const titleOf = (family: string) => artifactLabel(family, "figure")

const scopeOf = (name: string): { kind: "global" | "topic"; topic?: string } => {
  const match = /(?:^|\/)topic\/topic_(\d+)\//iu.exec(name)
  if (match) return { kind: "topic", topic: "topic_" + match[1] }
  return { kind: "global" }
}

/** 每个图形家族只保留一份预览：优先 PNG，其次 SVG；PDF 只作为下载件。 */
function previewFor(family: string, files: ResultFile[]): string | undefined {
  const variants = files.filter((file) => familyOf(file) === family)
  const pick = (suffix: string) => variants.find((file) => file.name.toLowerCase().endsWith(suffix))?.url
  return pick(".png") ?? pick(".svg")
}

function parseCsv(text: string): string[][] {
  const rows: string[][] = []
  let row: string[] = []
  let field = ""
  let quoted = false
  const source = text.replace(/^\uFEFF/u, "")
  for (let index = 0; index < source.length; index += 1) {
    const char = source[index]
    if (quoted) {
      if (char === '"') {
        if (source[index + 1] === '"') { field += '"'; index += 1 } else quoted = false
      } else field += char
    } else if (char === '"') quoted = true
    else if (char === ",") { row.push(field); field = "" }
    else if (char === "\n") { row.push(field); rows.push(row); row = []; field = "" }
    else if (char !== "\r") field += char
  }
  if (field || row.length) { row.push(field); rows.push(row) }
  return rows.filter((line) => line.some((cell) => cell.trim() !== ""))
}

const escapeCell = (value: string) => value.replace(/\|/gu, "\\|").replace(/\n/gu, " ").trim()

function toMarkdown(rows: string[][], limit = 40): { markdown: string; shown: number; total: number } {
  if (!rows.length) return { markdown: "", shown: 0, total: 0 }
  const [header, ...body] = rows
  const shownRows = body.slice(0, limit)
  const lines = [
    "| " + header.map(cell => escapeCell(fieldLabel(cell))).join(" | ") + " |",
    "| " + header.map((_, index) => index === 0 ? "---" : "---:").join(" | ") + " |",
    ...shownRows.map((line) => "| " + header.map((_, index) => escapeCell(/^(metric|指标)$/i.test(header[index]) ? fieldLabel(line[index] ?? "") : line[index] ?? "")).join(" | ") + " |"),
  ]
  return { markdown: lines.join("\n"), shown: shownRows.length, total: body.length }
}

const downloadFile = (url: string, name: string) => {
  const anchor = document.createElement("a")
  anchor.href = url
  anchor.download = name
  anchor.rel = "noreferrer"
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
}

export function ResearchResultView({ source, initialDestination, onOpenAssistant, onSelection, onContinueResearch, continuingResearch }: {
  source: ResultSource
  initialDestination?: "results" | "interpretation"
  onOpenAssistant: () => void
  onContinueResearch: () => void
  continuingResearch: boolean
  onSelection: (selection: { model: string; jobId: string } | null) => void
}) {
  const runId = resultScope(source)
  const projectName = resultProjectName(source)
  const datasetName = resultDatasetName(source)
  const projectId = source.kind === "run" ? source.projectId : undefined
  const initialCatalog = source.kind === "run" ? source.catalog : undefined
  const [catalog, setCatalog] = useState<ResultItem[]>(initialCatalog?.results ?? [])
  const [refreshing, setRefreshing] = useState(false)
  const refreshInFlight = useRef(false)
  const [loadError, setLoadError] = useState<string>()
  const [selectedJobId, setSelectedJobId] = useProjectDraft<string | undefined>(`results:${runId}:job`, initialCatalog?.results.find((item) => item.status === "completed")?.jobId ?? initialCatalog?.results[0]?.jobId)
  const [activeTab, setActiveTab] = useProjectDraft(`results:${runId}:tab`, initialDestination === "interpretation" ? "visualizations" : "overview")
  const [scope, setScope] = useProjectDraft<"global" | "topic">(`results:${runId}:scope`, "global")
  const [category, setCategory] = useProjectDraft(`results:${runId}:category`, "all")
  const [topicKey, setTopicKey] = useProjectDraft<string>(`results:${runId}:topic`, "")
  const [figurePage, setFigurePage] = useState(0)
  const [analyses, setAnalyses] = useProjectDraft<Record<string, string>>(`results:${runId}:${selectedJobId}:analyses`, {})
  const [busy, setBusy] = useState<Record<string, boolean>>({})
  const [quoted, setQuoted] = useState<string[]>([])
  const referenceId = (family: string) => JSON.stringify([runId, selectedJobId, family])
  useEffect(() => {
    const sync = (event: Event) => setQuoted((event as CustomEvent<string[]>).detail ?? [])
    window.addEventListener("theta:chart-references-changed", sync)
    window.dispatchEvent(new CustomEvent("theta:request-chart-references"))
    return () => { window.removeEventListener("theta:chart-references-changed", sync) }
  }, [])
  useEffect(() => { setBusy({}); setTables({}); setTableErrors({}) }, [selectedJobId])
  const [tableErrors, setTableErrors] = useState<Record<string, string>>({})
  const [tables, setTables] = useState<Record<string, { markdown: string; shown: number; total: number }>>({})
  const [datasetLabel, setDatasetLabel] = useState(() => datasetName && !datasetName.startsWith("dataset-") ? datasetName : undefined)

  useEffect(() => {
    if (!projectId) return
    void listDatasets(projectId).then(({ datasets }) => { if (datasets[0]?.name) setDatasetLabel(datasets[0].name) }).catch(() => undefined)
  }, [projectId])

  const refresh = useCallback(async () => {
    if (refreshInFlight.current) return
    refreshInFlight.current = true
    setRefreshing(true)
    setLoadError(undefined)
    setTableErrors({})
    try {
      const next = await loadResultCatalog(source)
      setCatalog(next.results)
      setSelectedJobId((current) => current && next.results.some((item) => item.jobId === current) ? current : next.results.find((item) => item.status === "completed")?.jobId ?? next.results[0]?.jobId)
    } catch (error) {
      setLoadError(systemText(error instanceof Error ? error.message : String(error), "读取结果失败，请重试"))
    } finally { refreshInFlight.current = false; setRefreshing(false) }
  }, [runId])

  useEffect(() => {
    void refresh()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    // Report publication can happen after training has already reached 100%.
    if (!shouldRefreshResultCatalog(catalog)) return
    const timer = window.setInterval(() => { void refresh() }, 4000)
    return () => window.clearInterval(timer)
  }, [catalog, refresh])

  useEffect(() => {
    if (initialCatalog) setCatalog(initialCatalog.results)
  }, [initialCatalog])

  useEffect(() => {
    const onFocus = () => { void refresh() }
    window.addEventListener('focus', onFocus)
    return () => window.removeEventListener('focus', onFocus)
  }, [refresh])

  const selected = catalog.find((item) => item.jobId === selectedJobId) ?? catalog[0]
  const readinessNotice = selected ? resultReadinessNotice(selected, source.kind === 'run') : undefined
  useEffect(() => { onSelection(selected ? { model: selected.modelId, jobId: selected.jobId } : null) }, [selected?.jobId, selected?.modelId, onSelection])
  const completed = catalog.filter((item) => item.status === "completed")
  const files: ResultFile[] = selected?.artifacts?.files ?? []
  const figures = useMemo(() => files.filter((file) => file.kind === "figure"), [files])
  const tableFiles = useMemo(() => files.filter((file) => file.kind === "table"), [files])
  const matrices = files.filter((file) => file.kind === "matrix")
  const archiveUrl = (selected?.artifacts as { archiveUrl?: string } | undefined)?.archiveUrl

  const figureFamilies = useMemo(() => {
    const families = new Map<string, ResultFile[]>()
    for (const file of figures) {
      const family = familyOf(file)
      // 表格图片不再展示（改用 markdown 表格），核心指标总览不再产出也不再展示。
      if (/topic_table|主题汇总表/iu.test(family) || /7 Core Metrics|7项核心指标/iu.test(family)) continue
      families.set(family, [...(families.get(family) ?? []), file])
    }
    return [...families.keys()]
  }, [figures])

  const topicKeys = useMemo(() => [...new Set(figures.map((file) => scopeOf(file.name)).filter((item) => item.kind === "topic").map((item) => item.topic!))].sort((left, right) => Number(left.split("_")[1]) - Number(right.split("_")[1])), [figures])
  useEffect(() => { if (!topicKeys.includes(topicKey) && topicKeys.length) setTopicKey(topicKeys[0]) }, [topicKey, topicKeys])

  const scopedFamilies = useMemo(() => figureFamilies.filter((family) => scopeOf(family).kind === scope), [figureFamilies, scope])
  const categoryOptions = useMemo(() => {
    const counts = new Map<string, number>()
    for (const family of scopedFamilies) counts.set(categoryOf(family), (counts.get(categoryOf(family)) ?? 0) + 1)
    return [{ key: "all", label: "全部", count: scopedFamilies.length }, ...CATEGORIES.filter((item) => counts.has(item.key)).map((item) => ({ key: item.key, label: item.label, count: counts.get(item.key) ?? 0 })), ...(counts.has("other") ? [{ key: "other", label: "其他", count: counts.get("other") ?? 0 }] : [])]
  }, [scopedFamilies])
  const visibleFamilies = useMemo(() => {
    const topics = scope === "topic" ? scopedFamilies.filter((family) => scopeOf(family).topic === topicKey) : scopedFamilies
    return category === "all" ? topics : topics.filter((family) => categoryOf(family) === category)
  }, [category, scope, scopedFamilies, topicKey])
  useEffect(() => { setFigurePage(0) }, [category, scope, topicKey, selectedJobId])
  const figurePages = Math.max(1, Math.ceil(visibleFamilies.length / 6))
  const pageFamilies = visibleFamilies.slice(Math.min(figurePage, figurePages - 1) * 6, (Math.min(figurePage, figurePages - 1) + 1) * 6)

  const findTable = useCallback((matcher: RegExp) => tableFiles.find((file) => matcher.test(file.name)), [tableFiles])

  const loadTable = useCallback(async (file: ResultFile, limit = 40) => {
    if (tables[file.url] || tableErrors[file.url]) return tables[file.url]
    try {
      const response = await fetch(file.url)
      if (!response.ok) throw new Error("读取表格失败，请刷新后重试")
      const parsed = toMarkdown(parseCsv(await response.text()), limit)
      setTables((prev) => ({ ...prev, [file.url]: parsed }))
      return parsed
    } catch (error) { setTableErrors(prev => ({ ...prev, [file.url]: error instanceof Error ? error.message : "读取表格失败" })) }
  }, [tables, tableErrors])

  const CORE_TABLES = useMemo(() => [
    { key: "主题 × 关键词", file: findTable(/tables\/topic_keywords\.csv$/u) ?? findTable(/topic_word_weights\.csv$/u) ?? findTable(/(?:topic_table|主题表)\.csv$/u), note: "模型实际导出的主题关键词与权重；完整内容可下载 CSV。" },
    { key: "文档 × 主题", file: findTable(/tables\/document_topic\.csv$/u), note: "每篇文档在各主题上的权重（theta 原值）与主导主题。" },
  ].filter((item) => item.file), [findTable])

  useEffect(() => { for (const item of CORE_TABLES) void loadTable(item.file!) }, [CORE_TABLES, loadTable, selectedJobId])

  const sourcesFor = useCallback((family: string): ResultFile[] => {
    return resolveChartDataFiles({ name: basename(family), path: family }, files.map(file => ({ ...file, path: file.name })))
  }, [files])

  const buildPayload = useCallback(async (chartName: string, filesToUse: ResultFile[]): Promise<AgentChartDataPayload> => {
    if (!filesToUse.length) throw new Error('当前图表没有可验证的绘图数据，不能生成解读')
    const sources = await Promise.all(filesToUse.slice(0, 4).map(async (file) => {
      const response = await fetch(file.url)
      if (!response.ok) throw new Error("读取绘图数据失败，请重试")
      const full = await response.text()
      const content = full.length > 512_000 ? full.slice(0, 512_000) : full
      return { name: basename(file.name), path: file.name, format: "csv" as const, content, truncated: content.length < full.length, totalCharacters: full.length }
    }))
    return { chartName, chartPath: chartName, dataset: datasetLabel ?? projectName, model: (selected?.modelId ?? "model").toUpperCase(), sources }
  }, [datasetLabel, projectName, selected?.modelId])

  const analyze = useCallback(async (family: string) => {
    const label = titleOf(family)
    setBusy((prev) => ({ ...prev, [family]: true }))
    try {
      const payload = await buildPayload(label, sourcesFor(family))
      const response = await ETMAgentAPI.advisory(
        "请根据这份图表的原始数据，写一段可以直接放进研究报告的结果分析。",
        { app_state: "result_chart_analysis", model: (selected?.modelId ?? "").toUpperCase(), dataset: datasetLabel ?? projectName, figure: label },
        [payload],
        "chart-analysis",
      )
      setAnalyses((prev) => ({ ...prev, [family]: response.message }))
      window.dispatchEvent(new CustomEvent("theta:chart-analysis", { detail: { chartName: label, analysis: response.message } }))
    } catch (error) {
      setAnalyses((prev) => ({ ...prev, [family]: "解读失败：" + advisoryErrorMessage(error) }))
    } finally {
      setBusy((prev) => ({ ...prev, [family]: false }))
    }
  }, [buildPayload, datasetLabel, projectName, selected?.modelId, sourcesFor])

  const toggleQuote = useCallback(async (family: string) => {
    const label = titleOf(family)
    setBusy(prev => ({ ...prev, [family]: true }))
    try {
      const payload = await buildPayload(label, sourcesFor(family))
      window.dispatchEvent(new CustomEvent("theta:chart-data-to-chat", { detail: [{ ...payload, referenceId: referenceId(family) }] }))
      onOpenAssistant()
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : "引用失败，请重试")
    } finally { setBusy(prev => ({ ...prev, [family]: false })) }
  }, [buildPayload, runId, selectedJobId, sourcesFor, onOpenAssistant])

  const downloadSources = useCallback(async (family: string) => {
    for (const file of sourcesFor(family)) {
      downloadFile(file.url, basename(file.name))
      await new Promise((resolve) => setTimeout(resolve, 250))
    }
  }, [sourcesFor])

  const statusTone = (status?: string) => status === "completed" ? "bg-emerald-50 text-emerald-700" : status === "failed" ? "bg-rose-50 text-rose-700" : "bg-blue-50 text-blue-700"

  const metricRows = useMemo(() => {
    const file = findTable(/evaluation_metrics\.csv$/u)
    const parsed = file ? tables[file.url] : undefined
    if (!parsed) return [] as Array<[string, string]>
    return parseCsvFromMarkdown(parsed.markdown).slice(1).map((line) => [line[0], line[1]] as [string, string])
  }, [findTable, tables])

  const curated = useCallback((patterns: RegExp[]) => figureFamilies.filter((family) => patterns.some((pattern) => pattern.test(basename(family)))), [figureFamilies])
  const topicFigures = useMemo(() => curated([/topic_wordcloud_grid|词云总览/iu, /Topic Proportion Distribution|主题占比/iu, /Intertopic Distance Map|主题间距离/iu, /Topic Correlation Network|主题相关性网络/iu, /Representative Topic Evolution|代表性主题/iu]), [curated])
  const metricFigures = useMemo(() => curated([/Topic Coherence|主题一致性/iu, /Topic Exclusivity|主题排他性/iu, /Topic Distribution Similarity Evolution|主题分布相似度演化/iu]), [curated])

  useEffect(() => {
    const file = findTable(/evaluation_metrics\.csv$/u)
    if (file) void loadTable(file, 20)
  }, [findTable, loadTable, selectedJobId])

  if (!selected) {
    return <div className="min-h-full bg-[#f5f8fc] p-8"><div className="mx-auto max-w-3xl rounded-2xl border border-dashed border-slate-200 bg-white p-10 text-center text-sm text-slate-500"><p role={loadError ? 'alert' : 'status'}>{loadError ?? (refreshing ? '正在读取训练结果…' : '当前项目还没有可查看的训练结果。')}</p><button type="button" disabled={refreshing} onClick={() => void refresh()} className="mt-4 rounded-lg border px-3 py-2">重新加载</button></div></div>
  }

  const tableCard = (title: string, file: ResultFile | undefined, note: string) => {
    if (!file) return null
    const parsed = tables[file.url]
    return (
      <section key={title} className="rounded-[22px] border border-slate-200 bg-white p-5 shadow-sm">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div><h3 className="text-base font-semibold text-slate-900">{title}</h3><p className="mt-1 text-xs leading-5 text-slate-500">{note}</p></div>
          <button type="button" onClick={() => downloadFile(file.url, basename(file.name))} className="inline-flex items-center gap-1.5 rounded-xl border border-slate-200 px-3 py-2 text-xs font-medium text-slate-600 transition hover:border-blue-200 hover:text-blue-600"><Download className="h-3.5 w-3.5" />下载{artifactLabel(file.name, "table")}</button>
        </div>
        <div className="mt-4 max-h-[420px] overflow-auto rounded-xl border border-slate-100">
          {tableErrors[file.url] ? <p role="alert" className="p-3 text-xs text-rose-600">{tableErrors[file.url]}</p> : parsed ? <MarkdownRenderer content={parsed.markdown} className="p-3 text-xs" /> : <div className="flex items-center gap-2 p-4 text-xs text-slate-400"><Loader2 className="h-3.5 w-3.5 animate-spin" />正在读取表格…</div>}
        </div>
        {parsed && <p className="mt-2 text-[11px] text-slate-400">表内显示前 {parsed.shown} 行，共 {parsed.total} 行；完整数据请下载 CSV。</p>}
      </section>
    )
  }

  const figureCard = (family: string) => {
    const url = previewFor(family, files)
    const variants = files.filter((file) => familyOf(file) === family)
    const download = variants.find((file) => file.name.toLowerCase().endsWith(".png")) ?? variants[0]
    const sources = sourcesFor(family)
    const title = titleOf(family)
    const analysis = analyses[family]
    return (
      <article key={family} className="overflow-hidden rounded-2xl border border-slate-200 bg-white">
        <div className="flex min-h-[260px] items-center justify-center bg-slate-50">
          {url ? <img src={url} alt={title} className="max-h-[300px] w-full object-contain" loading="lazy" /> : <div className="flex flex-col items-center gap-2 text-slate-400"><ImageIcon className="h-8 w-8" /><span className="text-xs">该图只有矢量/下载文件</span></div>}
        </div>
        <div className="border-t border-slate-200 px-4 py-3">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0"><p className="break-words text-sm font-medium text-slate-800">{title}</p><p className="mt-0.5 truncate text-[11px] text-slate-400">{sources.length ? `${sources.length} 份配套数据` : "研究图表"}</p></div>
            {download && <button type="button" onClick={() => downloadFile(download.url, basename(download.name))} className="shrink-0 rounded-lg border border-slate-200 px-2.5 py-1.5 text-[11px] text-slate-600 transition hover:border-blue-200 hover:text-blue-600" title={artifactLabel(download.name, "figure")}>下载图</button>}
          </div>
          <div className="mt-2.5 flex flex-wrap items-center gap-2">
            <button type="button" disabled={busy[family]} onClick={() => void toggleQuote(family)} className={"inline-flex items-center gap-1 rounded-lg px-2.5 py-1.5 text-[11px] transition " + (quoted.includes(referenceId(family)) ? "bg-blue-600 text-white" : "border border-slate-200 text-slate-600 hover:border-blue-200 hover:text-blue-600")}>
              <Quote className="h-3 w-3" />{quoted.includes(referenceId(family)) ? "已引用到对话" : "引用"}
            </button>
            <button type="button" disabled={busy[family]} onClick={() => void analyze(family)} className="inline-flex items-center gap-1 rounded-lg bg-slate-900 px-2.5 py-1.5 text-[11px] text-white transition hover:bg-blue-600 disabled:opacity-60">
              {busy[family] ? <Loader2 className="h-3 w-3 animate-spin" /> : <Sparkles className="h-3 w-3" />}猫咪科学家解读
            </button>
            {sources.length > 0 && <button type="button" onClick={() => void downloadSources(family)} className="inline-flex items-center gap-1 rounded-lg border border-slate-200 px-2.5 py-1.5 text-[11px] text-slate-600 transition hover:border-blue-200 hover:text-blue-600" title={"绘图原始数据：" + sources.map((file) => artifactLabel(file.name, "table")).join("、")}>
              <Download className="h-3 w-3" />下载绘图数据{sources.length > 1 ? "（" + sources.length + "）" : ""}
            </button>}
          </div>
          {analysis && <details open className="mt-3 rounded-xl border border-blue-100 bg-blue-50/60 p-3 text-xs leading-6 text-slate-700"><summary className="cursor-pointer font-medium">结果解读（点击收起或展开）</summary><MarkdownRenderer content={analysis} className="mt-2" /></details>}
        </div>
      </article>
    )
  }

  return (
    <div className="min-h-full min-w-0 bg-[#f5f8fc] p-4 [overflow-wrap:anywhere] sm:p-6 lg:p-8">
      <div className="mx-auto max-w-[1380px]">
        {loadError && <p role="alert" className="mb-4 rounded-lg bg-rose-50 p-3 text-sm text-rose-700">{loadError}</p>}
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap items-center gap-2">
            <button type="button" onClick={onContinueResearch} disabled={continuingResearch} aria-busy={continuingResearch} title={source.kind === 'dataset' ? '复制当前项目的已完成结果到对话模式，保留原手动项目' : '回到当前项目的对话，继续研究'} className="inline-flex items-center gap-2 rounded-xl bg-blue-600 px-3 py-2 text-sm font-medium text-white shadow-sm transition hover:bg-blue-700 disabled:opacity-60">{continuingResearch ? <Loader2 className="h-4 w-4 animate-spin" /> : <MessageSquare className="h-4 w-4" />}{continuingResearch ? '正在复制项目结果…' : '咨询结果助手'}</button>
            {archiveUrl && <button type="button" onClick={() => downloadFile(archiveUrl, "theta-" + (selected.modelId ?? "result") + "-figures-and-data.zip")} className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-600 shadow-sm transition hover:border-blue-200 hover:text-blue-600"><Package className="h-4 w-4" />下载当前模型完整包</button>}
            <button type="button" onClick={() => void refresh()} className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-600 shadow-sm transition hover:border-blue-200 hover:text-blue-600"><RefreshCw className={"h-4 w-4 " + (refreshing ? "animate-spin" : "")} />刷新结果</button>
          </div>
        </div>

        <header className="overflow-hidden rounded-[24px] border border-slate-200/80 bg-white shadow-[0_12px_34px_rgba(49,72,118,0.07)]">
          <div className="border-b border-slate-100 px-5 py-5 sm:px-7">
            <div className="flex flex-wrap items-start justify-between gap-5">
              <div className="min-w-0">
                <div className="mb-2 flex items-center gap-2 text-xs font-medium text-blue-600"><Sparkles className="h-3.5 w-3.5" />研究成果</div>
                <h1 className="break-words text-2xl font-semibold tracking-[-0.02em] text-slate-950">{projectName}</h1>
                <p className="mt-1.5 flex items-center gap-2 text-sm text-slate-500"><Database className="h-4 w-4" />{datasetLabel || "已绑定研究数据"}</p>
              </div>
              <div className="flex gap-2">
                <div className="rounded-2xl bg-slate-50 px-4 py-3 text-center"><strong className="block text-xl text-slate-900">{catalog.length}</strong><span className="text-xs text-slate-500">训练任务</span></div>
                <div className="rounded-2xl bg-blue-50/70 px-4 py-3 text-center"><strong className="block text-xl text-blue-700">{completed.length}</strong><span className="text-xs text-blue-600">已完成</span></div>
              </div>
            </div>
          </div>
          <div className="px-5 py-4 sm:px-7">
            <div className="mb-3 flex items-center justify-between"><p className="text-xs font-medium uppercase tracking-[0.14em] text-slate-400">选择模型结果</p><p className="text-xs text-slate-400">每个模型保留独立参数、图表与导出文件</p></div>
            <div className="grid grid-cols-[repeat(auto-fill,minmax(min(100%,180px),1fr))] gap-2.5">
              {catalog.map((item) => {
                const active = item.jobId === selected?.jobId
                return <button key={item.jobId} type="button" aria-pressed={active} onClick={() => setSelectedJobId(item.jobId)} className={"group rounded-2xl border p-3 text-left transition " + (active ? "border-blue-400 bg-blue-50/60 shadow-[0_8px_18px_rgba(59,130,246,0.12)]" : "border-slate-200 bg-white hover:border-blue-200 hover:bg-slate-50/70")}>
                  <div className="flex flex-wrap items-start justify-between gap-2"><strong className="whitespace-nowrap text-sm text-slate-900">{item.modelId.toUpperCase()}</strong><span className={"shrink-0 whitespace-nowrap rounded-full px-2 py-0.5 text-[10px] font-medium " + statusTone(item.status)}>{statusLabel(item.status)}</span></div>
                  <div className="mt-3 flex flex-wrap items-center justify-between gap-2 text-[11px] text-slate-500"><span>{item.percent ?? (item.status === "completed" ? 100 : 0)}%</span><span>{item.artifacts?.figureCount ?? 0} 图 · {item.artifacts?.tableCount ?? 0} 表</span></div>
                </button>
              })}
            </div>
          </div>
        </header>

        {readinessNotice && <p role="status" className="mt-4 rounded-xl border border-blue-100 bg-blue-50 p-4 text-sm leading-6 text-blue-800">{readinessNotice}</p>}

        <div className="mt-5 flex flex-wrap gap-1.5 rounded-2xl border border-slate-200 bg-white p-1.5 shadow-sm">
          {[["overview", "结果概览"], ["topics", "主题结果"], ["metrics", "评估指标"], ["visualizations", "可视化"], ["files", "导出文件"], ...(source.kind === "dataset" ? [["logs", "执行日志"]] : [])].map(([key, label]) => (
            <button key={key} type="button" aria-pressed={activeTab === key} onClick={() => setActiveTab(key)} className={"rounded-xl px-4 py-2.5 text-sm transition " + (activeTab === key ? "bg-slate-900 text-white" : "text-slate-600 hover:bg-slate-50")}>{label}</button>
          ))}
        </div>

        {activeTab === "logs" && <div className="mt-4"><ExecutionLog states={selected.execution ? [selected.execution] : []} workers={[{ id: selected.jobId, model: selected.modelId }]} logs={[]} running={selected.status === "running" || selected.status === "queued"} /></div>}

        {activeTab === "overview" && (
          <div className="mt-4 grid gap-4 lg:grid-cols-[minmax(0,1fr)_320px]">
            <section className="rounded-[22px] border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
              <div className="flex flex-wrap items-center justify-between gap-3"><div><p className="text-xs font-medium text-blue-600">当前模型</p><h2 className="mt-1 text-xl font-semibold text-slate-900">{selected.modelId.toUpperCase()}</h2></div><div className="flex gap-2">{selected.artifacts?.reportUrl && <a href={selected.artifacts.reportUrl} target="_blank" rel="noreferrer" className="inline-flex items-center gap-2 rounded-xl bg-slate-900 px-3.5 py-2 text-sm font-medium text-white transition hover:bg-blue-600"><FileText className="h-4 w-4" />完整原生报告</a>}</div></div>
              <div className="mt-5 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                {[{ label: "状态", value: statusLabel(selected.status), icon: CheckCircle2 }, { label: "阶段", value: phaseLabel(selected.phase), icon: Clock3 }, { label: "图表", value: String(selected.artifacts?.figureCount ?? 0), icon: ImageIcon }, { label: "数据文件", value: String((selected.artifacts?.tableCount ?? 0) + (selected.artifacts?.matrixCount ?? 0)), icon: Layers3 }].map((item) => <article key={item.label} className="rounded-2xl border border-slate-100 bg-slate-50/70 p-4"><item.icon className="h-4 w-4 text-blue-600" /><p className="mt-3 text-xs text-slate-500">{item.label}</p><strong className="mt-0.5 block truncate text-base text-slate-900">{item.value}</strong></article>)}
              </div>
              <div className="mt-5 space-y-4">{CORE_TABLES.map((item) => tableCard(item.key, item.file, item.note))}</div>
            </section>
            <aside className="space-y-4">
              <section className="rounded-[22px] border border-slate-200 bg-white p-5 shadow-sm"><div className="flex items-center gap-2 text-sm font-semibold text-slate-900"><BarChart3 className="h-4 w-4 text-blue-600" />产物清单</div><div className="mt-4 space-y-2.5">{[{ label: "可视化", value: figureFamilies.length }, { label: "数据表", value: tableFiles.length }, { label: "矩阵数据", value: matrices.length }, { label: "全部文件", value: files.length }].map((item) => <div key={item.label} className="flex items-center justify-between rounded-xl bg-slate-50 px-3 py-2.5 text-sm"><span className="text-slate-500">{item.label}</span><strong className="text-slate-900">{item.value}</strong></div>)}</div></section>
              <section className="rounded-[22px] border border-blue-100 bg-gradient-to-br from-blue-50 to-white p-5"><p className="text-sm font-semibold text-slate-900">带着结果，继续研究</p><p className="mt-2 text-xs leading-5 text-slate-600">{source.kind === 'dataset' ? '将当前项目的已完成结果复制到独立的对话项目，继续提问、比较模型或修改图表。原手动项目保持不变。' : '回到当前项目的对话，继续提问、比较模型或修改图表。'}单张图表也可以引用到右侧助手进行只读咨询。</p><button type="button" onClick={onContinueResearch} disabled={continuingResearch} className="mt-4 inline-flex items-center gap-1 text-sm font-medium text-blue-700 disabled:opacity-60">{continuingResearch ? '正在复制项目结果…' : '咨询结果助手'} <ChevronRight className="h-4 w-4" /></button></section>
            </aside>
          </div>
        )}

        {activeTab === "topics" && (
          <div className="mt-4 space-y-4">
            <div className="grid gap-4 md:grid-cols-2">{topicFigures.map((family) => figureCard(family))}</div>
            {tableCard("主题 × 关键词", CORE_TABLES.find((item) => item.key.startsWith("主题"))?.file, "模型实际导出的主题关键词与权重；完整内容可下载 CSV。")}
          </div>
        )}

        {activeTab === "metrics" && (
          <div className="mt-4 space-y-4">
            {metricRows.length > 0 && <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">{metricRows.map(([key, value]) => <article key={key} className="rounded-2xl border border-slate-100 bg-white p-4 shadow-sm"><p className="truncate text-xs text-slate-500">{key}</p><strong className="mt-2 block text-lg text-slate-900">{Number.isFinite(Number(value)) ? Number(value).toFixed(4) : value}</strong></article>)}</div>}
            <div className="grid gap-4 md:grid-cols-2">{metricFigures.map((family) => figureCard(family))}</div>
            {tableCard("评估指标明细", findTable(/evaluation_metrics\.csv$/u), "一行一个指标；口径以原生报告为准，数字之间不构成模型优劣排名。")}
          </div>
        )}

        {activeTab === "files" && <section className="mt-4 rounded-[22px] border border-slate-200 bg-white p-5">
          <h2 className="text-base font-semibold">导出文件</h2>
          <p className="mt-2 text-sm text-slate-500">完整包包含当前模型的图表、绘图数据、训练产物、源代码与使用方法。下方可单独下载文件。</p>
          <div className="mt-4 max-h-[560px] space-y-2 overflow-auto">
            {files.map(file => <a key={file.url} href={file.url} download={basename(file.name)} className="flex items-center justify-between gap-3 rounded-xl border border-slate-100 p-3 text-sm hover:bg-slate-50"><span className="min-w-0 break-all">{artifactLabel(file.name, file.kind)}</span><span className="shrink-0 text-xs text-slate-500">{FILE_LABEL[file.kind] ?? "文件"} · {file.name.split(".").pop()?.toUpperCase()}</span></a>)}
          </div>
        </section>}

        {activeTab === "visualizations" && (
          <div className="mt-4 space-y-4">
            <section className="rounded-[22px] border border-slate-200 bg-white p-5 shadow-sm">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div><h3 className="text-base font-semibold text-slate-900">可视化</h3><p className="mt-1 text-xs text-slate-500">按报告分层与用途筛选；每张卡都能下载图、下载对应的绘图原始数据、引用到对话，或直接让猫咪科学家写一段结果分析。</p></div>
                <div className="flex flex-wrap items-center gap-2">
                  <div className="flex rounded-xl border border-slate-200 p-1">
                    <button type="button" onClick={() => setScope("global")} className={"rounded-lg px-3 py-1.5 text-xs " + (scope === "global" ? "bg-slate-900 text-white" : "text-slate-600")}>全局</button>
                    <button type="button" onClick={() => setScope("topic")} className={"rounded-lg px-3 py-1.5 text-xs " + (scope === "topic" ? "bg-slate-900 text-white" : "text-slate-600")}>分主题</button>
                  </div>
                  {scope === "topic" && <select value={topicKey} onChange={(event) => setTopicKey(event.target.value)} className="rounded-xl border border-slate-200 px-3 py-2 text-xs text-slate-700">{topicKeys.map((key) => <option key={key} value={key}>主题 {key.split("_")[1]}</option>)}</select>}
                </div>
              </div>
              <div className="mt-3 flex flex-wrap gap-1.5">
                {categoryOptions.map((item) => <button key={item.key} type="button" onClick={() => setCategory(item.key)} className={"rounded-full border px-3 py-1.5 text-xs transition " + (category === item.key ? "border-blue-300 bg-blue-50 text-blue-700" : "border-slate-200 text-slate-600 hover:border-blue-200")}>{item.label}<span className="ml-1 text-[10px] text-slate-400">{item.count}</span></button>)}
              </div>
              {visibleFamilies.length ? <div className="mt-4 grid gap-4 md:grid-cols-2">{pageFamilies.map((family) => figureCard(family))}</div> : <div className="mt-4 flex min-h-40 items-center justify-center rounded-2xl border border-dashed border-slate-200 bg-slate-50/60 text-sm text-slate-500">该筛选下暂无图形产物。</div>}
              {figurePages > 1 && <div className="mt-4 flex items-center justify-end gap-3 text-sm"><button disabled={figurePage === 0} onClick={() => setFigurePage(page => Math.max(0, page - 1))} className="rounded-lg border px-3 py-2 disabled:opacity-40">上一页</button><span>第 {Math.min(figurePage + 1, figurePages)} / {figurePages} 页</span><button disabled={figurePage >= figurePages - 1} onClick={() => setFigurePage(page => page + 1)} className="rounded-lg border px-3 py-2 disabled:opacity-40">下一页</button></div>}
            </section>
          </div>
        )}
      </div>
    </div>
  )
}

function parseCsvFromMarkdown(markdown: string): string[][] {
  return markdown.split("\n").filter((line) => line.trim() && !/^\|\s*-+/u.test(line)).map((line) => line.replace(/^\||\|$/gu, "").split(" | ").map((cell) => cell.replace(/\\\|/gu, "|")))
}
