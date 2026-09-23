"use client"

import { useState } from "react"
import { artifactLabel } from "@/lib/presentation"
import JSZip from "jszip"
import { apiFetch, API_BASE } from "@/lib/api/config"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { Download, Package, Loader2, FileJson, FileText, Image, Table } from "lucide-react"

interface ExportTabProps {
  dataset: string
  mode: string
  selectedModel?: string
}

interface TopicWordsData {
  dataset: string
  model: string
  topics: Record<string, [string, number | null][]>
}

interface MetricsData {
  dataset: string
  model: string
  metrics: Record<string, number | number[]>
}

interface VisualizationData {
  dataset: string
  model: string
  global_files: { name: string; path: string; url: string; size: number; type: string }[]
  topic_files: Record<string, { name: string; path: string; url: string; size: number; type: string }[]>
}

// 导出项定义
const EXPORT_ITEMS = [
  { id: "topic_words", label: "主题词", description: "各主题的 top-K 关键词与权重", icon: Table, formats: ["csv", "json"] },
  { id: "metrics", label: "评估指标", description: "多维度评估指标数值", icon: FileJson, formats: ["csv", "json"] },
  { id: "visualizations", label: "可视化图表", description: "全局可视化和主题可视化所有图片", icon: Image, formats: ["zip"] },
]

export function ExportTab({ dataset, mode, selectedModel = "theta" }: ExportTabProps) {
  const [selected, setSelected] = useState<Set<string>>(
    new Set(EXPORT_ITEMS.map((o) => o.id))
  )
  const [selectedFormats, setSelectedFormats] = useState<Record<string, string>>({
    topic_words: "csv",
    metrics: "csv",
  })
  const [exporting, setExporting] = useState(false)
  const [progress, setProgress] = useState("")
  const [progressPercent, setProgressPercent] = useState(0)

  const toggle = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const setFormat = (id: string, format: string) => {
    setSelectedFormats((prev) => ({ ...prev, [id]: format }))
  }

  // 下载最终 ZIP 文件
  const downloadZip = async (zip: JSZip, filename: string) => {
    const content = await zip.generateAsync({ type: "blob" })
      const url = URL.createObjectURL(content)
      const a = document.createElement("a")
      a.href = url
      a.download = filename
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
  }

  // 获取图片 blob - 通过后端代理避免 CORS 跨域
  const fetchImageBlob = async (
    url: string,
    dataset: string,
    model: string,
    path: string
  ): Promise<Blob> => {
    // 使用后端代理，避免 OSS 跨域
    const proxyUrl = url.startsWith('/api/') ? url : `${API_BASE}/api/results/${encodeURIComponent(dataset)}/visualizations/image?path=${encodeURIComponent(path)}&model=${encodeURIComponent(model)}`
    // 添加认证头
    const token = typeof window !== 'undefined' ? localStorage.getItem('access_token') : null
    const headers: Record<string, string> = {}
    if (token) {
      headers['Authorization'] = `Bearer ${token}`
    }
    const response = await fetch(proxyUrl, { headers })
    if (!response.ok) {
      throw new Error(`图表下载失败： ${response.status} ${response.statusText}`)
    }
    return await response.blob()
  }

  // 生成主题词内容
  const generateTopicWordsContent = (data: TopicWordsData, format: string): string => {
    if (format === "csv") {
      // CSV 格式：主题, 排名, 词语, 权重
      const rows = [["主题", "排名", "词语", "权重"]]
      for (const [topicId, words] of Object.entries(data.topics)) {
        const wordsList = words
        wordsList.forEach(([word, weight], idx) => {
          const topicNumber = Number(topicId.replace(/^topic_/, ''))
          rows.push([Number.isFinite(topicNumber) ? `主题${topicNumber + 1}` : topicId, String(idx + 1), word, typeof weight === 'number' ? weight.toFixed(6) : ''])
        })
      }
      return '\uFEFF' + rows.map(r => r.map(cell => `"${cell.replaceAll('"', '""')}"`).join(',')).join('\n')
    } else {
      // JSON 格式
      const formatted: Record<string, { words: string[]; weights: (number | null)[] }> = {}
      for (const [topicId, words] of Object.entries(data.topics)) {
        const wordsList = words
        formatted[topicId] = {
          words: wordsList.map(([w]) => w),
          weights: wordsList.map(([, v]) => v),
        }
      }
      return JSON.stringify(formatted, null, 2)
    }
  }

  // 生成评估指标内容
  const generateMetricsContent = (data: MetricsData, format: string): string => {
    const validKeys = ["TD", "iRBO", "NPMI", "C_V", "UMass", "Exclusivity", "PPL"]

    const metricLabels: Record<string, string> = {
      TD: "主题多样性",
      iRBO: "主题差异度",
      NPMI: "主题连贯性",
      C_V: "主题一致性",
      UMass: "UMass 连贯性",
      Exclusivity: "主题互斥性",
      PPL: "困惑度",
    }

    if (format === "csv") {
      // CSV 格式
      const rows = [["指标名称", "指标英文名", "数值"]]
      for (const key of validKeys) {
        if (key in data.metrics) {
          const value = data.metrics[key]
          rows.push([
            metricLabels[key] || key,
            key,
            typeof value === "number" ? value.toFixed(6) : String(value),
          ])
        }
      }
      return rows.map((r) => r.join(",")).join("\n")
    } else {
      // JSON 格式
      const filteredMetrics: Record<string, number> = {}
      for (const key of validKeys) {
        if (key in data.metrics) {
          filteredMetrics[key] = data.metrics[key] as number
        }
      }
      return JSON.stringify(filteredMetrics, null, 2)
    }
  }

  // 完整打包导出
  const handleExportAll = async () => {
    if (selected.size === 0) {
      alert("请至少选择一项导出内容")
      return
    }

    setExporting(true)
    setProgress("初始化...")
    setProgressPercent(0)

    try {
      const zip = new JSZip()
      const zipFilename = `${dataset}_${selectedModel}_export.zip`

      let processed = 0
      const total = selected.size

      // 1. 导出主题词
      if (selected.has("topic_words")) {
        setProgress("获取主题词数据...")
        const data = await apiFetch<TopicWordsData>(
          API_BASE,
          `/api/results/${encodeURIComponent(dataset)}/topic-words?model=${encodeURIComponent(selectedModel)}`
        )

        const format = selectedFormats.topic_words || "csv"
        const content = generateTopicWordsContent(data, format)
        const ext = format === "csv" ? "csv" : "json"

        zip.file(`topic_words/topic_words.${ext}`, content)
        processed++
        setProgressPercent(Math.round((processed / total) * 100))
      }

      // 2. 导出评估指标
      if (selected.has("metrics")) {
        setProgress("获取评估指标...")
        const data = await apiFetch<MetricsData>(
          API_BASE,
          `/api/results/${encodeURIComponent(dataset)}/metrics?model=${encodeURIComponent(selectedModel)}`
        )

        const format = selectedFormats.metrics || "csv"
        const content = generateMetricsContent(data, format)
        const ext = format === "csv" ? "csv" : "json"

        zip.file(`metrics/metrics.${ext}`, content)
        processed++
        setProgressPercent(Math.round((processed / total) * 100))
      }

      // 3. 导出可视化图表（所有图片分类放入文件夹）
      if (selected.has("visualizations")) {
        setProgress("获取可视化列表...")
        const vizData = await apiFetch<VisualizationData>(
          API_BASE,
          `/api/results/${encodeURIComponent(dataset)}/visualizations?model=${encodeURIComponent(selectedModel)}`
        )

        // 创建文件夹结构:
        // visualizations/
        //   global/
        //     *.png
        //   topics/
        //     topic_01/
        //       *.png

        const vizFolder = zip.folder("visualizations")
        if (!vizFolder) throw new Error("无法创建图表目录")

        // 全局可视化
        let imageCount = 0
        const totalImages =
          (vizData.global_files?.length || 0) +
          Object.values(vizData.topic_files || {}).reduce(
            (sum, files) => sum + files.length,
            0
          )

        // 处理全局图片
        if (vizData.global_files && vizData.global_files.length > 0) {
          const globalFolder = vizFolder.folder("global")
          if (!globalFolder) throw new Error("无法创建全局结果目录")

          for (const file of vizData.global_files) {
            setProgress(`下载图片: ${artifactLabel(file.name, "figure")} (${imageCount + 1}/${totalImages})`)
            // file.path 格式: "global/filename.png"
            const blob = await fetchImageBlob(file.url, dataset, selectedModel, file.path)
            globalFolder.file(file.name, blob)
            imageCount++
            // 正确计算进度：避免超过 100%
            const itemProgress = (imageCount / totalImages) * (100 / total)
            const totalProgress = (processed * 100 / total) + itemProgress
            setProgressPercent(Math.round(Math.min(totalProgress, 100)))
          }
        }

        // 处理主题图片
        if (vizData.topic_files) {
          for (const [topicId, files] of Object.entries(vizData.topic_files)) {
            const topicFolder = vizFolder.folder(`topics/topic_${topicId}`)
            if (!topicFolder) continue

            for (const file of files) {
              setProgress(`下载图片: ${artifactLabel(file.name, "figure")} (${imageCount + 1}/${totalImages})`)
              // file.path 格式: "topic/topic_N/filename.png"
              const blob = await fetchImageBlob(file.url, dataset, selectedModel, file.path)
              topicFolder.file(file.name, blob)
              imageCount++
              // 正确计算进度：避免超过 100%
              const itemProgress = (imageCount / totalImages) * (100 / total)
              const totalProgress = (processed * 100 / total) + itemProgress
              setProgressPercent(Math.round(Math.min(totalProgress, 100)))
            }
          }
        }

        processed++
        setProgressPercent(Math.round((processed / total) * 100))
      }

      // 生成并下载 ZIP
      setProgress("打包 ZIP...")
      await downloadZip(zip, zipFilename)

      setProgress("")
      setProgressPercent(100)

      // eslint-disable-next-line no-console
      console.log(`✅ Export completed: ${zipFilename} (${selected.size} items exported)`)
    } catch (e) {
      console.error("Export failed:", e)
      alert(`导出失败: ${e instanceof Error ? e.message : String(e)}`)
    } finally {
      setExporting(false)
    }
  }

  return (
    <div className="max-w-2xl space-y-6">
      <div className="space-y-3 rounded-xl border border-blue-200 bg-blue-50 p-5">
        <h3 className="text-sm font-semibold text-blue-900">完整研究交付包</h3>
        <p className="text-xs leading-5 text-slate-600">下载当前模型的全部图表、绘图数据、训练模型、源代码和中文使用方法。新训练任务附带全部模型统一的加载入口；历史任务可能没有完整模型文件。</p>
        <Button asChild variant="outline"><a href={`${API_BASE}/api/results/${encodeURIComponent(dataset)}/archive?model=${encodeURIComponent(selectedModel)}`} download><Package className="mr-2 size-4" />下载模型、数据与代码</a></Button>
      </div>
      {/* 选择导出内容 */}
      <div className="rounded-xl border border-slate-200 bg-white p-5 space-y-4">
        <h3 className="text-sm font-semibold text-slate-700">选择导出内容</h3>
        {EXPORT_ITEMS.map((item) => {
          const Icon = item.icon
          const isSelected = selected.has(item.id)
          return (
            <div key={item.id} className="space-y-2">
              <label className="flex items-start gap-3 cursor-pointer group">
                <Checkbox
                  checked={isSelected}
                  onCheckedChange={() => toggle(item.id)}
                  className="mt-0.5 border-slate-300 data-[state=checked]:bg-blue-600"
                />
                <div className="flex items-center gap-2 flex-1">
                  <Icon className="w-4 h-4 text-slate-400" />
                  <div>
                    <p className="text-sm font-medium text-slate-800 group-hover:text-slate-900">
                      {item.label}
                    </p>
                    <p className="text-xs text-slate-500 mt-0.5">{item.description}</p>
                  </div>
                </div>
              </label>

              {/* 格式选择 */}
              {isSelected && item.formats.length > 1 && (
                <div className="ml-10 flex items-center gap-2">
                  <span className="text-xs text-slate-500">格式:</span>
                  <div className="flex gap-1">
                    {item.formats.map((format) => (
                      <button
                        key={format}
                        onClick={() => setFormat(item.id, format)}
                        className={`text-xs px-2 py-1 rounded border transition-colors ${
                          selectedFormats[item.id] === format
                            ? "bg-blue-100 border-blue-300 text-blue-700"
                            : "bg-white border-slate-200 text-slate-600 hover:border-slate-300"
                        }`}
                      >
                        {format.toUpperCase()}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )
        })}
      </div>

      {/* 导出按钮 */}
      <div className="flex flex-col gap-3">
        <Button
          onClick={handleExportAll}
          disabled={selected.size === 0 || exporting}
          className="bg-blue-600 hover:bg-blue-700"
        >
          {exporting ? (
            <>
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              {progress || "打包中..."} {progressPercent > 0 ? ` ${progressPercent}%` : ""}
            </>
          ) : (
            <>
              <Package className="w-4 h-4 mr-2" />
              打包并下载 ZIP ({selected.size} 项)
            </>
          )}
        </Button>
      </div>

      {/* ZIP 结构说明 */}
      <div className="rounded-lg bg-slate-50 border border-slate-200 px-4 py-3 text-xs text-slate-500 flex items-start gap-2">
        <Download className="w-3.5 h-3.5 mt-0.5 shrink-0" />
        <div className="space-y-1">
          <p>• 所有选中内容打包为单个 ZIP 文件，命名格式: <code className="bg-slate-200 px-1 rounded">{dataset}_{selectedModel}_export.zip</code></p>
          <p>• ZIP 内部按目录分类:</p>
          <pre className="bg-slate-100 p-2 rounded mt-1 overflow-x-auto">
{`${dataset}_${selectedModel}_export.zip
├── topic_words/
│   └── topic_words.[csv|json]
├── metrics/
│   └── metrics.[csv|json]
└── visualizations/
    ├── global/
    │   └── *.png (全局可视化)
    └── topics/
        └── topic_N/
            └── *.png (每个主题的可视化)`}
          </pre>
        </div>
      </div>
    </div>
  )
}
