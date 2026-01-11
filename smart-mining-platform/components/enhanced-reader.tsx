"use client"

import type React from "react"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { BookOpen, ImageIcon, FileText, ChevronLeft, ChevronRight, ZoomIn, ZoomOut } from "lucide-react"

interface Citation {
  id: string
  text: string
  source: string
  type: "book" | "image" | "ocr"
  page?: number
  imageUrl?: string
}

interface Paragraph {
  id: string
  text: string
  citations: Citation[]
}

export function EnhancedReader() {
  const [hoveredCitation, setHoveredCitation] = useState<Citation | null>(null)
  const [currentPage, setCurrentPage] = useState(1)

  const paragraphs: Paragraph[] = [
    {
      id: "1",
      text: "软组织肉瘤是一组起源于间叶组织的恶性肿瘤，具有高度异质性。",
      citations: [
        {
          id: "c1",
          text: "软组织肉瘤是一组起源于间叶组织的恶性肿瘤",
          source: "《WHO肿瘤分类》第5版",
          type: "book",
          page: 12,
        },
      ],
    },
    {
      id: "2",
      text: "根据2020年WHO分类，软组织肿瘤被分为多个亚型，包括脂肪细胞肿瘤、成纤维细胞/肌成纤维细胞肿瘤等。免疫组织化学标记物在鉴别诊断中起关键作用。",
      citations: [
        {
          id: "c2",
          text: "根据2020年WHO分类",
          source: "WHO Classification of Tumours, 5th Edition",
          type: "book",
          page: 45,
        },
        {
          id: "c3",
          text: "免疫组织化学标记物在鉴别诊断中起关键作用",
          source: "病理切片 H&E染色图像",
          type: "image",
          imageUrl: "/pathology-slide-microscope-image.jpg",
        },
      ],
    },
    {
      id: "3",
      text: "分子检测技术的进步使得肿瘤的精准分类成为可能。FISH检测和NGS测序已成为常规诊断手段。",
      citations: [
        {
          id: "c4",
          text: "FISH检测和NGS测序已成为常规诊断手段",
          source: "OCR识别：实验室报告单",
          type: "ocr",
        },
      ],
    },
    {
      id: "4",
      text: "三线细胞癌的诊断需要综合形态学、免疫表型和分子特征。细胞形态呈现多形性，核分裂象增多是重要特征。",
      citations: [
        {
          id: "c5",
          text: "三线细胞癌的诊断需要综合形态学、免疫表型和分子特征",
          source: "《Ackerman外科病理学》第11版",
          type: "book",
          page: 1567,
        },
        {
          id: "c6",
          text: "细胞形态呈现多形性",
          source: "高倍镜下细胞形态图",
          type: "image",
          imageUrl: "/cancer-cell-microscopy-high-magnification.jpg",
        },
      ],
    },
  ]

  const renderTextWithCitations = (paragraph: Paragraph) => {
    const text = paragraph.text
    const elements: React.ReactNode[] = []
    let lastIndex = 0

    paragraph.citations.forEach((citation, citationIndex) => {
      const index = text.indexOf(citation.text, lastIndex)
      if (index !== -1) {
        // Add text before citation
        if (index > lastIndex) {
          elements.push(<span key={`text-${paragraph.id}-${citationIndex}`}>{text.slice(lastIndex, index)}</span>)
        }
        // Add citation with superscript
        elements.push(
          <span
            key={`citation-${citation.id}`}
            className="relative inline"
            onMouseEnter={() => setHoveredCitation(citation)}
            onMouseLeave={() => setHoveredCitation(null)}
          >
            <span className="text-primary underline decoration-dotted cursor-help">{citation.text}</span>
            <sup className="text-xs text-accent font-semibold ml-0.5 cursor-help">[{citationIndex + 1}]</sup>
          </span>,
        )
        lastIndex = index + citation.text.length
      }
    })

    // Add remaining text
    if (lastIndex < text.length) {
      elements.push(<span key={`text-${paragraph.id}-end`}>{text.slice(lastIndex)}</span>)
    }

    return elements.length > 0 ? elements : text
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-foreground">增强式阅读器</h2>
          <p className="text-sm text-muted-foreground mt-1">悬停角标查看来源 · 支持书籍/图片OCR溯源</p>
        </div>
        <Badge variant="outline" className="bg-chart-3/10 text-chart-3 border-chart-3/30">
          Enhanced Reader
        </Badge>
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* Main Reader */}
        <Card className="lg:col-span-2 bg-card border-border">
          <CardHeader className="pb-3 flex flex-row items-center justify-between">
            <CardTitle className="text-base font-medium flex items-center gap-2">
              <BookOpen className="w-4 h-4 text-primary" />
              文献阅读区
            </CardTitle>
            <div className="flex items-center gap-2">
              <Button variant="ghost" size="icon" className="h-8 w-8">
                <ZoomOut className="w-4 h-4" />
              </Button>
              <span className="text-sm text-muted-foreground">100%</span>
              <Button variant="ghost" size="icon" className="h-8 w-8">
                <ZoomIn className="w-4 h-4" />
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-[500px] pr-4">
              <div className="space-y-6 text-foreground leading-relaxed">
                <h3 className="text-lg font-semibold text-primary">第三章：软组织肿瘤的病理诊断</h3>
                {paragraphs.map((paragraph) => (
                  <p key={paragraph.id} className="text-sm">
                    {renderTextWithCitations(paragraph)}
                  </p>
                ))}
              </div>
            </ScrollArea>

            {/* Page Navigation */}
            <div className="flex items-center justify-between mt-4 pt-4 border-t border-border">
              <Button variant="ghost" size="sm" disabled={currentPage === 1}>
                <ChevronLeft className="w-4 h-4 mr-1" />
                上一页
              </Button>
              <span className="text-sm text-muted-foreground">第 {currentPage} / 24 页</span>
              <Button variant="ghost" size="sm" onClick={() => setCurrentPage(currentPage + 1)}>
                下一页
                <ChevronRight className="w-4 h-4 ml-1" />
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Citation Panel */}
        <Card className="bg-card border-border">
          <CardHeader className="pb-3">
            <CardTitle className="text-base font-medium">来源追溯</CardTitle>
          </CardHeader>
          <CardContent>
            {hoveredCitation ? (
              <div className="space-y-4 animate-in fade-in duration-200">
                <div className="flex items-center gap-2">
                  {hoveredCitation.type === "book" && (
                    <Badge className="bg-accent/20 text-accent border-0">
                      <FileText className="w-3 h-3 mr-1" />
                      书籍来源
                    </Badge>
                  )}
                  {hoveredCitation.type === "image" && (
                    <Badge className="bg-primary/20 text-primary border-0">
                      <ImageIcon className="w-3 h-3 mr-1" />
                      图片来源
                    </Badge>
                  )}
                  {hoveredCitation.type === "ocr" && (
                    <Badge className="bg-chart-3/20 text-chart-3 border-0">
                      <FileText className="w-3 h-3 mr-1" />
                      OCR识别
                    </Badge>
                  )}
                </div>

                <div className="p-3 rounded-lg bg-muted/30">
                  <p className="text-sm text-foreground font-medium mb-2">"{hoveredCitation.text}"</p>
                  <p className="text-xs text-muted-foreground">
                    {hoveredCitation.source}
                    {hoveredCitation.page && ` · 第 ${hoveredCitation.page} 页`}
                  </p>
                </div>

                {hoveredCitation.imageUrl && (
                  <div className="rounded-lg overflow-hidden border border-border">
                    <img
                      src={hoveredCitation.imageUrl || "/placeholder.svg"}
                      alt="Citation source"
                      className="w-full h-40 object-cover"
                    />
                    <div className="p-2 bg-muted/30">
                      <p className="text-xs text-muted-foreground">图片来源：{hoveredCitation.source}</p>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="h-48 flex flex-col items-center justify-center text-muted-foreground">
                <BookOpen className="w-8 h-8 mb-2 opacity-50" />
                <p className="text-sm">悬停文中角标查看来源</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
