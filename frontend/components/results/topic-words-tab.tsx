"use client"

import { useEffect, useState } from "react"
import { Badge } from "@/components/ui/badge"
import { ChevronDown, ChevronUp, Loader2, AlertCircle } from "lucide-react"
import { apiFetch, API_BASE } from "@/lib/api/config"

interface TopicWordsTabProps {
  dataset: string
  mode: string
  shouldLoad: boolean
  selectedModel?: string
}

interface TopicData {
  dataset: string
  model: string
  topics: Record<string, [string, number | null][]>
}

export function TopicWordsTab({ dataset, mode, shouldLoad, selectedModel = "theta" }: TopicWordsTabProps) {
  const [topicData, setTopicData] = useState<TopicData | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [expandedTopics, setExpandedTopics] = useState<Set<string>>(new Set())

  useEffect(() => {
    if (!shouldLoad) return
    setLoading(true)
    setError(null)

    apiFetch<TopicData>(API_BASE, `/api/results/${encodeURIComponent(dataset)}/topic-words?model=${encodeURIComponent(selectedModel)}`)
      .then((data) => {
        setTopicData(data)
        setExpandedTopics(new Set())
      })
      .catch((e) => {
        console.error("Failed to load topic words:", e)
        setError(e.message ?? "加载失败")
        setTopicData(null)
      })
      .finally(() => setLoading(false))
  }, [dataset, shouldLoad, selectedModel])

  const toggleExpand = (topic: string) => {
    setExpandedTopics((prev) => {
      const next = new Set(prev)
      if (next.has(topic)) next.delete(topic)
      else next.add(topic)
      return next
    })
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="w-6 h-6 animate-spin text-slate-400 mr-2" />
        <span className="text-slate-500">加载主题词...</span>
      </div>
    )
  }

  if (error || !topicData) {
    return (
      <div className="flex flex-col items-center justify-center py-20 gap-2 text-slate-500">
        <AlertCircle className="w-8 h-8 text-amber-400" />
        <p className="text-sm">{error ?? "暂无主题词数据"}</p>
      </div>
    )
  }

  const { topics } = topicData
  const topicEntries = Object.entries(topics).sort((a, b) => a[0].localeCompare(b[0], "zh-CN", { numeric: true }))

  return (
    <div className="space-y-4">
      {/* 顶部栏：主题数量 */}
      <div className="flex items-center justify-end">
        {topicData && (
          <p className="text-sm text-slate-500">
            {topicData.model.toUpperCase()} · 共 {topicEntries.length} 个主题
          </p>
        )}
      </div>

      {/* 主题列表 */}
      <div className="space-y-3">
        {topicEntries.map(([topicKey, words], topicIndex) => {
          const isExpanded = expandedTopics.has(topicKey)
          const identifier = topicKey.match(/\d+/)?.[0]
          const topicNum = identifier === undefined ? topicIndex + 1 : Number(identifier) + 1
          const displayWords = isExpanded ? words : words.slice(0, 10)
          // 取前5个词作为预览
          const previewWords = words.slice(0, 5).map((w) => w[0]).join(" · ")

          return (
            <div key={topicKey} className="rounded-xl border border-slate-200 bg-white overflow-hidden">
              <button
                onClick={() => toggleExpand(topicKey)}
                className="w-full flex items-center justify-between px-4 py-3 hover:bg-slate-50 transition-colors text-left"
              >
                <div className="flex items-center gap-2 min-w-0">
                  <Badge variant="outline" className="text-xs font-mono shrink-0">
                    主题 {topicNum}
                  </Badge>
                  <span className="text-sm text-slate-600 truncate min-w-0">
                    {previewWords}
                  </span>
                </div>
                {isExpanded ? (
                  <ChevronUp className="w-4 h-4 text-slate-400 shrink-0" />
                ) : (
                  <ChevronDown className="w-4 h-4 text-slate-400 shrink-0" />
                )}
              </button>

              {isExpanded && (
                <div className="px-4 pb-4 pt-2 border-t border-slate-100">
                  <div className="flex flex-wrap gap-2">
                    {displayWords.map(([word, weight], wIdx) => (
                      <span
                        key={word + wIdx}
                        className="inline-flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full bg-blue-50 border border-blue-100 text-blue-700 font-medium"
                        style={{ opacity: Math.max(0.4, 1 - wIdx * 0.03) }}
                        title={weight === null ? "该结果仅提供主题词顺序" : `权重: ${weight.toFixed(4)}`}
                      >
                        {word}
                      </span>
                    ))}
                  </div>
                  {!isExpanded && words.length > 10 && (
                    <p className="text-xs text-slate-400 mt-2">+{words.length - 10} 更多</p>
                  )}
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
