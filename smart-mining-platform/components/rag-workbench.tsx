"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Search, BookOpen, FileText, ExternalLink, Info, Microscope, Stethoscope } from "lucide-react"

interface SearchResult {
  id: string
  term: string
  source: string
  chapter: string
  page: number
  relevance: number
}

interface DiagnosticCase {
  id: string
  title: string
  symptoms: string[]
  knowledgeBase: string[]
}

export function RagWorkbench() {
  const [searchQuery, setSearchQuery] = useState("")
  const [searchResults, setSearchResults] = useState<SearchResult[]>([])
  const [isSearching, setIsSearching] = useState(false)

  const mockSearchResults: SearchResult[] = [
    {
      id: "1",
      term: "三线细胞癌",
      source: "《病理学诊断指南》",
      chapter: "第12章 肿瘤病理学",
      page: 456,
      relevance: 98,
    },
    {
      id: "2",
      term: "阿克曼外科病理学",
      source: "Ackerman's Surgical Pathology",
      chapter: "Chapter 8: Soft Tissue Tumors",
      page: 1234,
      relevance: 95,
    },
    {
      id: "3",
      term: "细胞分化特征",
      source: "《WHO肿瘤分类》第5版",
      chapter: "软组织肿瘤分类",
      page: 89,
      relevance: 87,
    },
  ]

  const diagnosticCases: DiagnosticCase[] = [
    {
      id: "1",
      title: "疑似软组织肉瘤病例",
      symptoms: ["深部软组织肿块", "生长迅速", "边界不清"],
      knowledgeBase: ["影像学特征", "病理切片分析", "分子标记物"],
    },
    {
      id: "2",
      title: "淋巴结转移评估",
      symptoms: ["淋巴结肿大", "质地坚硬", "活动度差"],
      knowledgeBase: ["TNM分期", "免疫组化", "基因检测"],
    },
  ]

  const handleSearch = () => {
    if (!searchQuery.trim()) return
    setIsSearching(true)
    setTimeout(() => {
      setSearchResults(mockSearchResults)
      setIsSearching(false)
    }, 800)
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-foreground">RAG 专业医疗/科研工作台</h2>
          <p className="text-sm text-muted-foreground mt-1">Research & RAG Section · 智能检索增强</p>
        </div>
        <Badge variant="outline" className="bg-accent/10 text-accent border-accent/30">
          Knowledge Retrieval
        </Badge>
      </div>

      {/* Terminology Search */}
      <Card className="bg-card border-border">
        <CardHeader className="pb-3">
          <CardTitle className="text-base font-medium flex items-center gap-2">
            <Search className="w-4 h-4 text-primary" />
            术语检索区
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-2">
            <Input
              placeholder="输入医学术语，如：三线细胞癌、阿克曼..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSearch()}
              className="flex-1 bg-input border-border"
            />
            <Button onClick={handleSearch} disabled={isSearching}>
              {isSearching ? "检索中..." : "快速定位"}
            </Button>
          </div>

          <div className="flex gap-2 flex-wrap">
            <Badge
              variant="secondary"
              className="cursor-pointer hover:bg-primary/20"
              onClick={() => setSearchQuery("三线细胞癌")}
            >
              三线细胞癌
            </Badge>
            <Badge
              variant="secondary"
              className="cursor-pointer hover:bg-primary/20"
              onClick={() => setSearchQuery("阿克曼")}
            >
              阿克曼
            </Badge>
            <Badge
              variant="secondary"
              className="cursor-pointer hover:bg-primary/20"
              onClick={() => setSearchQuery("免疫组化")}
            >
              免疫组化
            </Badge>
            <Badge
              variant="secondary"
              className="cursor-pointer hover:bg-primary/20"
              onClick={() => setSearchQuery("分子病理")}
            >
              分子病理
            </Badge>
          </div>

          {/* Search Results */}
          {searchResults.length > 0 && (
            <ScrollArea className="h-64 rounded-lg border border-border">
              <div className="p-4 space-y-3">
                {searchResults.map((result) => (
                  <div
                    key={result.id}
                    className="p-3 rounded-lg bg-muted/30 hover:bg-muted/50 transition-colors cursor-pointer"
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex items-center gap-2">
                        <BookOpen className="w-4 h-4 text-primary" />
                        <span className="font-medium text-foreground">{result.term}</span>
                      </div>
                      <Badge variant="outline" className="text-xs">
                        相关度 {result.relevance}%
                      </Badge>
                    </div>
                    <div className="mt-2 text-sm text-muted-foreground">
                      <p>{result.source}</p>
                      <p className="flex items-center gap-2 mt-1">
                        <FileText className="w-3 h-3" />
                        {result.chapter} · 第 {result.page} 页
                      </p>
                    </div>
                    <Button variant="ghost" size="sm" className="mt-2 h-7 text-xs">
                      <ExternalLink className="w-3 h-3 mr-1" />
                      跳转原文
                    </Button>
                  </div>
                ))}
              </div>
            </ScrollArea>
          )}
        </CardContent>
      </Card>

      {/* Diagnostic Framework */}
      <Card className="bg-card border-border">
        <CardHeader className="pb-3">
          <CardTitle className="text-base font-medium flex items-center gap-2">
            <Stethoscope className="w-4 h-4 text-primary" />
            科研诊断框架
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4">
            {diagnosticCases.map((caseItem) => (
              <div
                key={caseItem.id}
                className="p-4 rounded-lg border border-border bg-muted/20 hover:bg-muted/30 transition-colors"
              >
                <div className="flex items-center gap-2 mb-3">
                  <Microscope className="w-4 h-4 text-accent" />
                  <span className="font-medium text-foreground">{caseItem.title}</span>
                </div>

                <div className="grid md:grid-cols-2 gap-4">
                  <div>
                    <p className="text-xs text-muted-foreground mb-2">临床表现</p>
                    <div className="flex flex-wrap gap-1">
                      {caseItem.symptoms.map((symptom, idx) => (
                        <Badge key={idx} variant="outline" className="text-xs">
                          {symptom}
                        </Badge>
                      ))}
                    </div>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground mb-2">知识库关联</p>
                    <div className="flex flex-wrap gap-1">
                      {caseItem.knowledgeBase.map((kb, idx) => (
                        <Badge key={idx} className="text-xs bg-primary/20 text-primary border-0">
                          {kb}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </div>

                <Button variant="outline" size="sm" className="mt-3 bg-transparent">
                  <Info className="w-3 h-3 mr-1" />
                  开始辅助诊断
                </Button>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
