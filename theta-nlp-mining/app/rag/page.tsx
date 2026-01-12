"use client"

import { Suspense, useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Search, BookOpen, ExternalLink, FileImage, Database, Loader2, AlertCircle } from "lucide-react"
import { AISidebar } from "@/components/ai-sidebar"
import { ragService } from "@/lib/api/services"
import { Alert, AlertDescription } from "@/components/ui/alert"
import type { Citation } from "@/lib/api/services"

function RAGPageContent() {
  const [searchTerm, setSearchTerm] = useState("")
  const [showSource, setShowSource] = useState(false)
  const [searching, setSearching] = useState(false)
  const [searchError, setSearchError] = useState<string | null>(null)
  const [searchResults, setSearchResults] = useState<{
    content: string
    citations: Citation[]
  } | null>(null)
  const [knowledgeBases, setKnowledgeBases] = useState<Array<{
    name: string
    icon: typeof BookOpen
    count: number
  }>>([])

  // 加载知识库列表
  useEffect(() => {
    const loadKnowledgeBases = async () => {
      try {
        const response = await ragService.getKnowledgeBases()
        if (response.success && response.data) {
          const kbIcons = [BookOpen, FileImage, Database]
          setKnowledgeBases(
            response.data.map((kb, idx) => ({
              name: kb.name,
              icon: kbIcons[idx % kbIcons.length],
              count: kb.count,
            }))
          )
        }
      } catch (err) {
        // 如果 API 失败，使用默认数据
        setKnowledgeBases([
          { name: "肿瘤病理学教材", icon: BookOpen, count: 2 },
          { name: "临床影像资料", icon: FileImage, count: 15 },
          { name: "PubMed 检索库", icon: Database, count: 3 },
        ])
      }
    }
    loadKnowledgeBases()
  }, [])

  const handleSearch = async () => {
    if (!searchTerm.trim()) {
      setSearchError("请输入搜索关键词")
      return
    }

    setSearching(true)
    setSearchError(null)
    setSearchResults(null)

    try {
      const response = await ragService.search(searchTerm.trim())
      
      if (response.success && response.data && response.data.results.length > 0) {
        const firstResult = response.data.results[0]
        setSearchResults({
          content: firstResult.content,
          citations: firstResult.citations,
        })
      } else {
        setSearchError(response.error || "未找到相关结果")
      }
    } catch (err) {
      setSearchError(err instanceof Error ? err.message : "搜索失败")
    } finally {
      setSearching(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !searching) {
      handleSearch()
    }
  }

  // 格式化内容，添加引用标注
  const formatContentWithCitations = (content: string, citations: Citation[]) => {
    let formattedContent = content
    citations.forEach((cite, idx) => {
      const citationNum = idx + 1
      // 在内容中查找引用标记并替换为可点击的角标
      formattedContent = formattedContent.replace(
        new RegExp(`\\[${citationNum}\\]|⁽${citationNum}⁾`, 'g'),
        `⁽${citationNum}⁾`
      )
    })
    return formattedContent
  }

  const displayContent = searchResults
    ? formatContentWithCitations(searchResults.content, searchResults.citations)
    : `请输入搜索关键词开始检索...`

  const displayCitations = searchResults?.citations || []

  const aiMessages = [
    { 
      type: "framework", 
      text: searchResults 
        ? "诊断框架建议：建议采用三阶段诊断法 - 形态学筛查 → 免疫组化确认 → 分子检测验证"
        : "请输入搜索关键词获取诊断建议"
    },
    { 
      type: "evidence", 
      text: searchResults
        ? `科研论据：当前引用知识库包含 ${displayCitations.filter(c => c.type === 'paper').length} 篇 SCI 文献、${displayCitations.filter(c => c.type === 'pdf').length} 本专业教材、${displayCitations.filter(c => c.type === 'database').length} 个临床数据库`
        : `当前引用知识库包含 ${knowledgeBases.reduce((sum, kb) => sum + kb.count, 0)} 项资源`
    },
    { 
      type: "source", 
      text: searchResults
        ? `溯源状态：已链接 ${displayCitations.length} 个引用来源，可点击角标查看原文`
        : "点击文中数字角标查看原始来源"
    },
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
                  onKeyPress={handleKeyPress}
                  className="text-base"
                  disabled={searching}
                />
                <Button onClick={handleSearch} disabled={searching || !searchTerm.trim()}>
                  {searching ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      搜索中...
                    </>
                  ) : (
                    <>
                      <Search className="w-4 h-4 mr-2" />
                      搜索
                    </>
                  )}
                </Button>
              </div>
              {searchError && (
                <Alert variant="destructive" className="mt-4">
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>{searchError}</AlertDescription>
                </Alert>
              )}
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
                  {displayContent.split("⁽").map((part, idx) => {
                    if (idx === 0) return <span key={idx}>{part}</span>
                    const citationNum = part.match(/^(\d+)⁾/)?.[1]
                    const restText = part.replace(/^\d+⁾/, "")
                    const citation = displayCitations.find((c, i) => i + 1 === Number(citationNum))
                    return (
                      <span key={idx}>
                        <button
                          onClick={() => setShowSource(!showSource)}
                          className="inline-flex items-center justify-center w-5 h-5 text-xs font-medium text-primary-foreground bg-primary rounded-full hover:bg-primary/80 transition-colors mx-0.5"
                          title={citation?.source}
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
              {showSource && displayCitations.length > 0 && (
                <div className="mt-4 p-4 border border-accent rounded-lg bg-accent/5">
                  <h4 className="text-sm font-semibold mb-3">引用来源列表：</h4>
                  <div className="space-y-2">
                    {displayCitations.map((cite, idx) => (
                      <div key={idx} className="flex items-start gap-3 text-sm">
                        <Badge variant="outline" className="shrink-0">
                          {idx + 1}
                        </Badge>
                        <span className="flex-1">{cite.source}</span>
                        {cite.url && (
                          <Button 
                            size="sm" 
                            variant="ghost" 
                            className="h-6 px-2"
                            onClick={() => window.open(cite.url, '_blank')}
                          >
                            <ExternalLink className="w-3 h-3" />
                          </Button>
                        )}
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
