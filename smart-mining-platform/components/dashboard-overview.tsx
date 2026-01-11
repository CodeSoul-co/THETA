"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Database, FileSearch, Brain, Activity, TrendingUp, Users, FileText, Clock } from "lucide-react"

export function DashboardOverview() {
  const stats = [
    {
      title: "已处理文档",
      value: "12,847",
      change: "+12.5%",
      icon: FileText,
      color: "text-primary",
    },
    {
      title: "知识库条目",
      value: "45,392",
      change: "+8.2%",
      icon: Database,
      color: "text-accent",
    },
    {
      title: "RAG检索次数",
      value: "89,124",
      change: "+23.1%",
      icon: FileSearch,
      color: "text-chart-3",
    },
    {
      title: "活跃用户",
      value: "1,284",
      change: "+5.7%",
      icon: Users,
      color: "text-chart-4",
    },
  ]

  const recentActivities = [
    { action: "数据预处理完成", file: "临床病例数据集.csv", time: "2分钟前" },
    { action: "RAG检索", query: "三线细胞癌诊断标准", time: "5分钟前" },
    { action: "主题建模分析", model: "医疗健康LoRA", time: "12分钟前" },
    { action: "文档上传", file: "WHO肿瘤分类第5版.pdf", time: "1小时前" },
  ]

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-foreground">系统概览</h2>
          <p className="text-sm text-muted-foreground mt-1">智研多源平台运行状态</p>
        </div>
        <Badge variant="outline" className="bg-primary/10 text-primary border-primary/30">
          <Activity className="w-3 h-3 mr-1" />
          系统正常
        </Badge>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {stats.map((stat) => (
          <Card key={stat.title} className="bg-card border-border">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <stat.icon className={`w-5 h-5 ${stat.color}`} />
                <Badge variant="secondary" className="text-xs">
                  <TrendingUp className="w-3 h-3 mr-1" />
                  {stat.change}
                </Badge>
              </div>
              <div className="mt-3">
                <p className="text-2xl font-bold text-foreground">{stat.value}</p>
                <p className="text-xs text-muted-foreground mt-1">{stat.title}</p>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Recent Activities */}
      <Card className="bg-card border-border">
        <CardHeader className="pb-3">
          <CardTitle className="text-base font-medium flex items-center gap-2">
            <Clock className="w-4 h-4 text-primary" />
            最近活动
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {recentActivities.map((activity, idx) => (
              <div key={idx} className="flex items-center justify-between p-3 rounded-lg bg-muted/30">
                <div>
                  <p className="text-sm font-medium text-foreground">{activity.action}</p>
                  <p className="text-xs text-muted-foreground mt-0.5">
                    {activity.file || activity.query || activity.model}
                  </p>
                </div>
                <span className="text-xs text-muted-foreground">{activity.time}</span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* System Status */}
      <div className="grid md:grid-cols-3 gap-4">
        <Card className="bg-card border-border">
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-primary/10">
                <Brain className="w-5 h-5 text-primary" />
              </div>
              <div>
                <p className="text-sm font-medium text-foreground">AI 模型状态</p>
                <p className="text-xs text-primary">6 个LoRA适配器就绪</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card border-border">
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-accent/10">
                <Database className="w-5 h-5 text-accent" />
              </div>
              <div>
                <p className="text-sm font-medium text-foreground">向量数据库</p>
                <p className="text-xs text-accent">45K 条目已索引</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card border-border">
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-chart-3/10">
                <FileSearch className="w-5 h-5 text-chart-3" />
              </div>
              <div>
                <p className="text-sm font-medium text-foreground">RAG 引擎</p>
                <p className="text-xs text-chart-3">平均响应 0.8s</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
