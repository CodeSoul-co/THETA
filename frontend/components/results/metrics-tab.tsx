"use client"

import { useEffect, useState } from "react"
import { apiFetch, API_BASE } from "@/lib/api/config"
import { Loader2, AlertCircle } from "lucide-react"
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  Radar,
  ResponsiveContainer,
  Tooltip,
} from "recharts"

interface MetricsTabProps {
  dataset: string
  mode: string
  shouldLoad: boolean
  selectedModel?: string
}

interface MetricsData {
  dataset: string
  model: string
  metrics: Record<string, number | number[]>
}

// 固定的 7 个评估维度
const METRICS_CONFIG = [
  { key: "TD", label: "主题多样性",满分: 1 },
  { key: "iRBO", label: "主题差异度",满分: 1 },
  { key: "NPMI", label: "主题连贯性",满分: 1 },
  { key: "C_V", label: "主题一致性",满分: 1 },
  { key: "UMass", label: "语义连贯性",满分: -1, 反转: true },
  { key: "Exclusivity", label: "主题互斥性",满分: 1 },
  { key: "PPL", label: "困惑度",满分: 500, 反转: true },
]

// 将指标值归一化到 0-1
function normalize(value: number, max: number,反转: boolean = false): number {
  let normalized = Math.max(0, Math.min(1, value / max))
  if (反转) {
    normalized = 1 - normalized
  }
  return normalized
}

export function MetricsTab({ dataset, mode, shouldLoad, selectedModel = "theta" }: MetricsTabProps) {
  const [metricsData, setMetricsData] = useState<MetricsData | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!shouldLoad) return
    setLoading(true)
    setError(null)

    apiFetch<MetricsData>(API_BASE, `/api/results/${encodeURIComponent(dataset)}/metrics?model=${encodeURIComponent(selectedModel)}`)
      .then((data) => {
        setMetricsData(data)
      })
      .catch((e) => {
        console.error("Failed to load metrics:", e)
        setError(e.message ?? "加载失败")
        setMetricsData(null)
      })
      .finally(() => setLoading(false))
  }, [dataset, mode, shouldLoad, selectedModel])

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="w-6 h-6 animate-spin text-slate-400 mr-2" />
        <span className="text-slate-500">加载指标...</span>
      </div>
    )
  }

  if (error || !metricsData) {
    return (
      <div className="flex flex-col items-center justify-center py-20 gap-2 text-slate-500">
        <AlertCircle className="w-8 h-8 text-amber-400" />
        <p className="text-sm">{error ?? "暂无评估指标数据"}</p>
      </div>
    )
  }

  const { metrics } = metricsData

  // 构建雷达图数据
  const radarData = METRICS_CONFIG.map((config) => {
    const raw = metrics[config.key]
    // 取数组均值或直接使用标量值
    const rawValue = Array.isArray(raw) ? (raw as number[]).filter((v) => typeof v === "number").reduce((a, b) => a + b, 0) / (raw as number[]).filter((v) => typeof v === "number").length : (raw as number)
    const normalizedValue = normalize(rawValue, config.满分, config.反转)
    return {
      metric: config.label,
      value: normalizedValue,
      raw: rawValue,
    }
  })

  return (
    <div className="space-y-8">
      {/* 雷达图 */}
      <div className="rounded-xl border border-slate-200 bg-white p-6">
        <h3 className="text-sm font-semibold text-slate-700 mb-4">综合评估雷达图</h3>
        <ResponsiveContainer width="100%" height={320}>
          <RadarChart data={radarData} margin={{ top: 16, right: 32, bottom: 16, left: 32 }}>
            <PolarGrid stroke="#e2e8f0" />
            <PolarAngleAxis
              dataKey="metric"
              tick={{ fontSize: 11, fill: "#64748b" }}
            />
            <Tooltip
              formatter={(value: number, _name: string, entry: { payload?: { raw?: number } }) => [
                entry?.payload?.raw != null ? entry.payload.raw.toFixed(4) : value.toFixed(4),
                "值",
              ]}
            />
            <Radar
              dataKey="value"
              stroke="#3b82f6"
              fill="#3b82f6"
              fillOpacity={0.18}
              strokeWidth={2}
            />
          </RadarChart>
        </ResponsiveContainer>
      </div>

      {/* 指标表格 */}
      <div className="rounded-xl border border-slate-200 bg-white overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-100 bg-slate-50">
              <th className="text-left px-4 py-2.5 font-medium text-slate-600 text-xs">指标</th>
              <th className="text-right px-4 py-2.5 font-medium text-slate-600 text-xs">数值</th>
            </tr>
          </thead>
          <tbody>
            {METRICS_CONFIG.map((config, idx) => {
              const raw = metrics[config.key]
              const rawValue = Array.isArray(raw) ? (raw as number[]).filter((v) => typeof v === "number").reduce((a, b) => a + b, 0) / (raw as number[]).filter((v) => typeof v === "number").length : (raw as number)
              return (
                <tr
                  key={config.key}
                  className={`border-b border-slate-50 ${idx % 2 === 0 ? "bg-white" : "bg-slate-50/50"}`}
                >
                  <td className="px-4 py-2.5 text-slate-700">
                    {config.label}
                  </td>
                  <td className="px-4 py-2.5 text-right font-mono text-slate-900">
                    {typeof rawValue === "number" ? rawValue.toFixed(4) : "-"}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
