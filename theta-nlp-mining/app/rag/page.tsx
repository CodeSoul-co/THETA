"use client"

import { Suspense, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Search, BookOpen, ExternalLink, FileImage, Database } from "lucide-react"
import { AISidebar } from "@/components/ai-sidebar"

function RAGPageContent() {
  const [searchTerm, setSearchTerm] = useState("")
  const [showSource, setShowSource] = useState(false)

  const sampleContent = `
    三线细胞癌（Three-lined cell carcinoma）是一种罕见的恶性肿瘤，
    最早由阿克曼医生在 1954 年的病理学研究中描述⁽¹⁾。该疾病的典型特征包括：
    
    1. 细胞形态学特征：肿瘤细胞呈三条平行排列的线状结构⁽²⁾
    2. 免疫组化标记：CK7(+), CK20(-), CDX2(-)⁽³⁾
    3. 发病机制：目前认为与 TP53 基因突变相关⁽⁴⁾
    
    诊断依据主要参考《病理诊断标准手册》（第8版）第342页的描述⁽⁵⁾。
  `

  const citations = [
    { id: 1, source: "Ackerman LV. (1954) Archives of Pathology", type: "paper" },
    { id: 2, source: "肿瘤细胞形态学图谱.pdf - 第67页", type: "pdf" },
    { id: 3, source: "WHO 肿瘤分类标准 2021版", type: "database" },
    { id: 4, source: "Nature Medicine 2023;29(3):445-458", type: "paper" },
    { id: 5, source: "病理诊断标准手册（第8版）- 第342页", type: "pdf" },
  ]

  const aiMessages = [
    { type: "framework", text: "诊断框架建议：建议采用三阶段诊断法 - 形态学筛查 → 免疫组化确认 → 分子检测验证" },
    { type: "evidence", text: "科研论据：当前引用知识库包含 3 篇 SCI 文献、2 本专业教材、1 个临床数据库" },
    { type: "source", text: "溯源状态：已链接 5 个引用来源，可点击角标查看原文" },
  ]

  const knowledgeBases = [
    { name: "肿瘤病理学教材", icon: BookOpen, count: 2 },
    { name: "临床影像资料", icon: FileImage, count: 15 },
    { name: "PubMed 检索库", icon: Database, count: 3 },
  ]

  return (
    <div className="flex h-screen bg-background">
      {/* Left Main Workspace - 75% */}
      <div className="flex-1 overflow-y-auto p-8">
        <div className="max-w-5xl mx-auto space-y-6">
          {/* Header */}
          <div>
            <h1 className="text-4xl font-bold tracking-tight text-foreground">RAG 专业医疗科研工作台</h1>
            <p className="text-muted-foreground mt-2">RAG Specialist Workspace</p>
          </div>

          {/* Search Bar */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Search className="w-5 h-5" />
                术语检索
              </CardTitle>
              <CardDescription>支持专业医学词汇、疾病名称、药物名称精确检索</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex gap-2">
                <Input
                  placeholder="例如：三线细胞癌、阿克曼综合征..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="text-base"
                />
                <Button>搜索</Button>
              </div>
            </CardContent>
          </Card>

          {/* Enhanced Reader */}
          <Card>
            <CardHeader>
              <CardTitle>增强阅读器 (PDF/OCR 混合模式)</CardTitle>
              <CardDescription>溯源系统：点击文中数字角标查看原始来源</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="prose prose-sm max-w-none dark:prose-invert">
                <div className="bg-muted/30 p-6 rounded-lg leading-relaxed">
                  {sampleContent.split("⁽").map((part, idx) => {
                    if (idx === 0) return <span key={idx}>{part}</span>
                    const citationNum = part.match(/^(\d+)⁾/)?.[1]
                    const restText = part.replace(/^\d+⁾/, "")
                    return (
                      <span key={idx}>
                        <button
                          onClick={() => setShowSource(!showSource)}
                          className="inline-flex items-center justify-center w-5 h-5 text-xs font-medium text-primary-foreground bg-primary rounded-full hover:bg-primary/80 transition-colors mx-0.5"
                        >
                          {citationNum}
                        </button>
                        {restText}
                      </span>
                    )
                  })}
                </div>
              </div>

              {/* Citation Preview */}
              {showSource && (
                <div className="mt-4 p-4 border border-accent rounded-lg bg-accent/5">
                  <h4 className="text-sm font-semibold mb-3">引用来源列表：</h4>
                  <div className="space-y-2">
                    {citations.map((cite) => (
                      <div key={cite.id} className="flex items-start gap-3 text-sm">
                        <Badge variant="outline" className="shrink-0">
                          {cite.id}
                        </Badge>
                        <span className="flex-1">{cite.source}</span>
                        <Button size="sm" variant="ghost" className="h-6 px-2">
                          <ExternalLink className="w-3 h-3" />
                        </Button>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Knowledge Base Preview */}
          <Card>
            <CardHeader>
              <CardTitle>当前引用的知识库</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-3 gap-4">
                {knowledgeBases.map((kb) => {
                  const Icon = kb.icon
                  return (
                    <div key={kb.name} className="flex items-center gap-3 p-4 border rounded-lg">
                      <Icon className="w-8 h-8 text-primary" />
                      <div>
                        <p className="text-sm font-medium">{kb.name}</p>
                        <p className="text-xs text-muted-foreground">{kb.count} 项资源</p>
                      </div>
                    </div>
                  )
                })}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Right AI Sidebar - 25% */}
      <AISidebar title="领域专家 Agent" subtitle="Domain Expert" messages={aiMessages} />
    </div>
  )
}

export default function RAGPage() {
  return (
    <Suspense fallback={null}>
      <RAGPageContent />
    </Suspense>
  )
}
