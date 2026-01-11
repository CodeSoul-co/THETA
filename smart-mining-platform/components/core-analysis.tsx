"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Layers, Play, BarChart3, Cloud, ScanText as Scatter } from "lucide-react"

interface TopicCluster {
  id: string
  name: string
  keywords: string[]
  density: number
  color: string
}

interface UmapPoint {
  x: number
  y: number
  cluster: number
  label: string
}

export function CoreAnalysis() {
  const [selectedLora, setSelectedLora] = useState("")
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [showResults, setShowResults] = useState(false)
  const [umapPoints, setUmapPoints] = useState<UmapPoint[]>([])

  const loraAdapters = [
    { id: "medical", name: "医疗健康", size: "98MB", accuracy: "94.2%" },
    { id: "finance", name: "金融分析", size: "102MB", accuracy: "92.8%" },
    { id: "public-health", name: "公共卫生", size: "95MB", accuracy: "93.5%" },
    { id: "legal", name: "法律法规", size: "88MB", accuracy: "91.7%" },
    { id: "education", name: "教育研究", size: "92MB", accuracy: "90.4%" },
    { id: "social", name: "社会科学", size: "97MB", accuracy: "89.9%" },
  ]

  const topicClusters: TopicCluster[] = [
    {
      id: "1",
      name: "病理诊断",
      keywords: ["肿瘤", "细胞", "分化", "免疫组化", "病理切片"],
      density: 0.85,
      color: "bg-primary",
    },
    {
      id: "2",
      name: "临床表现",
      keywords: ["症状", "体征", "预后", "转移", "复发"],
      density: 0.72,
      color: "bg-accent",
    },
    {
      id: "3",
      name: "分子检测",
      keywords: ["基因", "突变", "测序", "标志物", "靶向"],
      density: 0.68,
      color: "bg-chart-3",
    },
    {
      id: "4",
      name: "治疗方案",
      keywords: ["手术", "化疗", "放疗", "免疫治疗", "靶向药"],
      density: 0.61,
      color: "bg-chart-4",
    },
  ]

  const topKeywords = [
    { word: "肿瘤分类", weight: 100 },
    { word: "免疫组化", weight: 92 },
    { word: "病理诊断", weight: 88 },
    { word: "分子标志物", weight: 82 },
    { word: "细胞分化", weight: 78 },
    { word: "基因突变", weight: 72 },
    { word: "预后评估", weight: 68 },
    { word: "临床分期", weight: 62 },
    { word: "靶向治疗", weight: 58 },
    { word: "影像学", weight: 52 },
  ]

  useEffect(() => {
    // Generate random UMAP points
    const points: UmapPoint[] = []
    const clusterCenters = [
      { x: 25, y: 30 },
      { x: 70, y: 25 },
      { x: 30, y: 70 },
      { x: 75, y: 75 },
    ]

    clusterCenters.forEach((center, clusterIdx) => {
      for (let i = 0; i < 15; i++) {
        points.push({
          x: center.x + (Math.random() - 0.5) * 25,
          y: center.y + (Math.random() - 0.5) * 25,
          cluster: clusterIdx,
          label: `Doc ${points.length + 1}`,
        })
      }
    })
    setUmapPoints(points)
  }, [])

  const handleAnalysis = () => {
    setIsAnalyzing(true)
    setTimeout(() => {
      setIsAnalyzing(false)
      setShowResults(true)
    }, 2000)
  }

  const clusterColors = ["fill-primary", "fill-accent", "fill-chart-3", "fill-chart-4"]

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-foreground">核心技术分析模块</h2>
          <p className="text-sm text-muted-foreground mt-1">Core Analysis Layer · 主题建模与聚类可视化</p>
        </div>
        <Badge variant="outline" className="bg-chart-4/10 text-chart-4 border-chart-4/30">
          Analysis Engine
        </Badge>
      </div>

      {/* LoRA Adapter Selector */}
      <Card className="bg-card border-border">
        <CardHeader className="pb-3">
          <CardTitle className="text-base font-medium flex items-center gap-2">
            <Layers className="w-4 h-4 text-primary" />
            LoRA 适配器选择器
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-4 items-end">
            <div className="flex-1">
              <label className="text-sm text-muted-foreground mb-2 block">选择领域专家适配器</label>
              <Select value={selectedLora} onValueChange={setSelectedLora}>
                <SelectTrigger className="bg-input border-border">
                  <SelectValue placeholder="选择 LoRA 适配器..." />
                </SelectTrigger>
                <SelectContent>
                  {loraAdapters.map((adapter) => (
                    <SelectItem key={adapter.id} value={adapter.id}>
                      <div className="flex items-center justify-between w-full">
                        <span>{adapter.name}</span>
                        <span className="text-xs text-muted-foreground ml-4">
                          {adapter.size} · {adapter.accuracy}
                        </span>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <Button onClick={handleAnalysis} disabled={!selectedLora || isAnalyzing} className="px-6">
              <Play className="w-4 h-4 mr-2" />
              {isAnalyzing ? "分析中..." : "生成分析"}
            </Button>
          </div>

          {selectedLora && (
            <div className="flex gap-2 flex-wrap">
              {loraAdapters.find((a) => a.id === selectedLora)?.name && (
                <Badge className="bg-primary/20 text-primary border-0">
                  已加载: {loraAdapters.find((a) => a.id === selectedLora)?.name}专家模型
                </Badge>
              )}
              <Badge variant="outline">约 100MB 模型参数</Badge>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Analysis Results */}
      {showResults && (
        <div className="grid lg:grid-cols-2 gap-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
          {/* UMAP Visualization */}
          <Card className="bg-card border-border">
            <CardHeader className="pb-3">
              <CardTitle className="text-base font-medium flex items-center gap-2">
                <Scatter className="w-4 h-4 text-primary" />
                UMAP 降维散点图
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="relative w-full h-64 bg-muted/20 rounded-lg border border-border overflow-hidden">
                <svg viewBox="0 0 100 100" className="w-full h-full">
                  {/* Grid lines */}
                  {[20, 40, 60, 80].map((pos) => (
                    <g key={pos}>
                      <line
                        x1={pos}
                        y1="0"
                        x2={pos}
                        y2="100"
                        stroke="currentColor"
                        strokeOpacity="0.1"
                        strokeWidth="0.2"
                      />
                      <line
                        x1="0"
                        y1={pos}
                        x2="100"
                        y2={pos}
                        stroke="currentColor"
                        strokeOpacity="0.1"
                        strokeWidth="0.2"
                      />
                    </g>
                  ))}
                  {/* Data points */}
                  {umapPoints.map((point, idx) => (
                    <circle
                      key={idx}
                      cx={point.x}
                      cy={point.y}
                      r="2"
                      className={`${clusterColors[point.cluster]} opacity-70 hover:opacity-100 transition-opacity cursor-pointer`}
                    />
                  ))}
                </svg>
                {/* Legend */}
                <div className="absolute bottom-2 right-2 flex gap-2">
                  {topicClusters.slice(0, 4).map((cluster, idx) => (
                    <div key={cluster.id} className="flex items-center gap-1">
                      <div className={`w-2 h-2 rounded-full ${cluster.color}`} />
                      <span className="text-xs text-muted-foreground">{cluster.name}</span>
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>

          {/* HDBSCAN Clusters */}
          <Card className="bg-card border-border">
            <CardHeader className="pb-3">
              <CardTitle className="text-base font-medium flex items-center gap-2">
                <BarChart3 className="w-4 h-4 text-primary" />
                HDBSCAN 密度聚类结果
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {topicClusters.map((cluster) => (
                  <div key={cluster.id} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <div className={`w-3 h-3 rounded-full ${cluster.color}`} />
                        <span className="text-sm font-medium text-foreground">{cluster.name}</span>
                      </div>
                      <span className="text-xs text-muted-foreground">密度: {(cluster.density * 100).toFixed(0)}%</span>
                    </div>
                    <div className="w-full h-2 bg-muted/30 rounded-full overflow-hidden">
                      <div
                        className={`h-full ${cluster.color} transition-all duration-500`}
                        style={{ width: `${cluster.density * 100}%` }}
                      />
                    </div>
                    <div className="flex gap-1 flex-wrap">
                      {cluster.keywords.slice(0, 3).map((kw, idx) => (
                        <Badge key={idx} variant="outline" className="text-xs">
                          {kw}
                        </Badge>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Keyword Cloud */}
          <Card className="lg:col-span-2 bg-card border-border">
            <CardHeader className="pb-3">
              <CardTitle className="text-base font-medium flex items-center gap-2">
                <Cloud className="w-4 h-4 text-primary" />
                c-TF-IDF Top-10 关键词标签云
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-2 justify-center py-4">
                {topKeywords.map((kw, idx) => (
                  <Badge
                    key={idx}
                    variant="outline"
                    className="transition-all hover:scale-105 cursor-pointer"
                    style={{
                      fontSize: `${Math.max(12, kw.weight / 8)}px`,
                      padding: `${4 + kw.weight / 20}px ${8 + kw.weight / 10}px`,
                      opacity: 0.5 + kw.weight / 200,
                    }}
                  >
                    {kw.word}
                  </Badge>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  )
}
