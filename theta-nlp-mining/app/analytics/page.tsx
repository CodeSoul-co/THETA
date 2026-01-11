"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import {
  LineChart,
  Line,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts"
import { TrendingUp, Network, GitBranch } from "lucide-react"
import { AISidebar } from "@/components/ai-sidebar"

// Sample data
const timeSeriesData = [
  { month: "2019-01", count: 45 },
  { month: "2019-02", count: 52 },
  { month: "2019-03", count: 85 },
  { month: "2019-04", count: 73 },
  { month: "2019-05", count: 68 },
  { month: "2019-06", count: 95 },
]

const umapData = Array.from({ length: 50 }, (_, i) => ({
  x: Math.random() * 100,
  y: Math.random() * 100,
  cluster: i % 5,
}))

const topKeywords = [
  { word: "政策改革", weight: 95 },
  { word: "立法框架", weight: 88 },
  { word: "社会治理", weight: 82 },
  { word: "数字化转型", weight: 76 },
  { word: "公共服务", weight: 71 },
  { word: "创新驱动", weight: 68 },
  { word: "可持续发展", weight: 65 },
  { word: "民生保障", weight: 62 },
  { word: "区域协调", weight: 58 },
  { word: "风险防控", weight: 55 },
]

const clusterColors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"]

export default function AnalyticsPage() {
  const [selectedPoint, setSelectedPoint] = useState<any>(null)

  const aiMessages = [
    { type: "metric", text: "F1-Score: 0.87" },
    { type: "metric", text: "Cv Score: 0.92" },
    { type: "insight", text: "检测到 2019-03 政策激增点，相关主题数量增加 63%" },
    {
      type: "recommendation",
      text: selectedPoint
        ? `Cluster ${selectedPoint.cluster}: 包含 ${Math.floor(Math.random() * 50) + 10} 个文档，主题聚焦于"政策执行"`
        : "点击左侧数据点查看详细分析",
    },
  ]

  return (
    <div className="flex h-screen bg-background">
      {/* Left Main Workspace - 75% */}
      <div className="flex-1 overflow-y-auto p-8">
        <div className="max-w-6xl mx-auto space-y-6">
          {/* Header */}
          <div>
            <h1 className="text-4xl font-bold tracking-tight text-foreground">BERTopic 深度挖掘仪表盘</h1>
            <p className="text-muted-foreground mt-2">Analytics Dashboard</p>
          </div>

          {/* Visualization Tabs */}
          <Tabs defaultValue="timeline" className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="timeline" className="flex items-center gap-2">
                <TrendingUp className="w-4 h-4" />
                主题演化
              </TabsTrigger>
              <TabsTrigger value="space" className="flex items-center gap-2">
                <Network className="w-4 h-4" />
                语义空间
              </TabsTrigger>
              <TabsTrigger value="hierarchy" className="flex items-center gap-2">
                <GitBranch className="w-4 h-4" />
                主题树
              </TabsTrigger>
            </TabsList>

            <TabsContent value="timeline" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle>时序演化分析</CardTitle>
                  <CardDescription>主题在时间维度上的分布变化（2019年政策激增点标注）</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={400}>
                    <LineChart data={timeSeriesData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                      <XAxis dataKey="month" stroke="hsl(var(--muted-foreground))" />
                      <YAxis stroke="hsl(var(--muted-foreground))" />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "hsl(var(--popover))",
                          border: "1px solid hsl(var(--border))",
                          borderRadius: "8px",
                        }}
                      />
                      <Line
                        type="monotone"
                        dataKey="count"
                        stroke="hsl(var(--primary))"
                        strokeWidth={2}
                        dot={{ r: 4 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="space" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle>UMAP 语义降维空间</CardTitle>
                  <CardDescription>高维语义向量降维至二维平面，颜色代表聚类簇</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={400}>
                    <ScatterChart>
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                      <XAxis type="number" dataKey="x" stroke="hsl(var(--muted-foreground))" />
                      <YAxis type="number" dataKey="y" stroke="hsl(var(--muted-foreground))" />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "hsl(var(--popover))",
                          border: "1px solid hsl(var(--border))",
                          borderRadius: "8px",
                        }}
                        cursor={{ strokeDasharray: "3 3" }}
                      />
                      <Scatter data={umapData} onClick={(data) => setSelectedPoint(data)}>
                        {umapData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={clusterColors[entry.cluster]} />
                        ))}
                      </Scatter>
                    </ScatterChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="hierarchy" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle>层次聚类树</CardTitle>
                  <CardDescription>展示主题之间的层次关系与相似度</CardDescription>
                </CardHeader>
                <CardContent className="h-[400px] flex items-center justify-center">
                  <div className="text-center text-muted-foreground">
                    <GitBranch className="w-16 h-16 mx-auto mb-4 opacity-50" />
                    <p>层次聚类树可视化组件</p>
                    <p className="text-xs mt-2">（使用 D3.js 或类似库实现树状图）</p>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>

          {/* Keywords Cloud */}
          <Card>
            <CardHeader>
              <CardTitle>c-TF-IDF 关键词提取</CardTitle>
              <CardDescription>Top-10 高权重术语标签</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-3">
                {topKeywords.map((kw) => (
                  <Badge
                    key={kw.word}
                    variant="secondary"
                    className="px-4 py-2 text-base"
                    style={{
                      fontSize: `${0.8 + (kw.weight / 100) * 0.6}rem`,
                      opacity: 0.6 + (kw.weight / 100) * 0.4,
                    }}
                  >
                    {kw.word}
                  </Badge>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Right AI Sidebar - 25% */}
      <AISidebar title="建模分析师 Agent" subtitle="Modeling Analyst" messages={aiMessages} />
    </div>
  )
}
