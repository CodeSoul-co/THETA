"use client"

import React from "react"

import { useState, useRef, useCallback, useEffect } from "react"
import {
  Sparkles,
  Paperclip,
  Send,
  Clock,
  Zap,
  Upload,
  FileText,
  Check,
  Copy,
  MessageSquare,
  BarChart3,
  ExternalLink,
  Trash2,
  Mic,
  PanelRightClose,
  ChevronDown,
  ChevronUp,
  X,
  Expand,
  Quote,
  Plus,
  Pin,
  PinOff,
  RotateCcw,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Avatar, AvatarImage, AvatarFallback } from "@/components/ui/avatar"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Badge } from "@/components/ui/badge"
import { Textarea } from "@/components/ui/textarea"
import { TypingMessage } from "@/components/typing-message"
import { MarkdownRenderer } from "@/components/markdown-renderer"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { cn } from "@/lib/utils"
import { PROMPTS } from "@/lib/config"
import { API_BASE } from "@/lib/api/config"
import type { AgentChartDataPayload } from "@/lib/api/etm-agent"
import { useProjectDraft } from "@/lib/use-project-draft"
import styles from './ai-sidebar.module.css'

// Message Types
export type MessageType = "text" | "chart_widget" | "table_summary" | "file_upload"

export interface ChartCitation {
  ref: string
  type: "chart"
  url: string
  caption: string
  analysis?: string
}

export interface ChatMessage {
  id: string
  role: "user" | "ai"
  content: string
  type: MessageType
  timestamp: string
  isThinking?: boolean
  shouldAnimateTyping?: boolean
  data?: {
    chartData?: { label: string; value: number }[]
    chartId?: string
    file?: { name: string; size: string; parsed: boolean }
    tableSummary?: { rows: number; columns: number; preview: string[] }
    imageAttachments?: { name: string; mimeType: string; size: number; url: string }[]
    fileAttachments?: { name: string; mimeType: string; size: number }[]
    chartDataAttachments?: { name: string; sourceCount: number }[]
  }
  followUpQuestions?: string[]
  /** Agent 回复中的图表引用 */
  citations?: ChartCitation[]
}

export interface SendMessageImage {
  name: string
  mimeType: string
  size: number
  dataUrl: string
}

export interface SendMessageFile {
  name: string
  mimeType: string
  size: number
  dataUrl: string
}

export interface SendMessagePayload {
  content: string
  images?: SendMessageImage[]
  files?: SendMessageFile[]
  charts?: AgentChartDataPayload[]
  /** 已生成的猫咪科学家解读，随引用一起进入对话上下文。 */
  chartAnalyses?: { chartName: string; analysis: string }[]
}

export interface SuggestionCard {
  title: string
  description: string
  onClick: () => void
}

interface AiSidebarProps {
  draftKey: string
  mode?: "full" | "advisory"
  chatHistory: ChatMessage[]
  onSendMessage: (payload: string | SendMessagePayload) => void | boolean | Promise<void | boolean>
  onDataUploaded?: (file: File) => void
  onFocusChart?: (chartId: string) => void
  onClearChat?: () => void
  onCollapse?: () => void
  onNewConversation?: () => void
  onSelectConversation?: (id: string) => void
  onPinConversation?: (id: string, pinned: boolean) => void
  onDeleteConversation?: (id: string) => void
  onRestoreConversation?: (id: string) => void
  consultationBusy?: boolean
  historyStoredInDatabase?: boolean
  sending?: boolean
  inputDisabled?: boolean
  activeConversationId?: string
  conversations?: { id: string; title: string; updatedAt: string; messageCount: number; pinned?: boolean; deletedAt?: string }[]
  /** 动态智能建议（训练完成后由 interpret API 生成） */
  dynamicSuggestions?: SuggestionCard[]
}

interface PendingImageAttachment {
  id: string
  file: File
  previewUrl: string
}

interface PendingFileAttachment {
  id: string
  file: File
}

function isChatAttachmentFile(file: File): boolean {
  return !file.type.startsWith("image/")
}

function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => {
      const result = reader.result
      if (typeof result === "string") {
        resolve(result)
      } else {
        reject(new Error("Failed to convert file to data URL"))
      }
    }
    reader.onerror = reject
    reader.readAsDataURL(file)
  })
}

// Mini Chart Component
function MiniChart({ data, chartId, onFocusChart }: {
  data: { label: string; value: number }[]
  chartId: string
  onFocusChart?: (chartId: string) => void
}) {
  const maxValue = Math.max(...data.map(d => d.value))
  
  return (
    <div className="mt-3 p-3 bg-slate-50 rounded-xl border border-slate-100">
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs font-semibold text-slate-500">数据可视化</span>
        <Button
          variant="ghost"
          size="sm"
          className="h-7 text-xs text-blue-600 hover:text-blue-700 hover:bg-blue-50 gap-1"
          onClick={() => {
            console.log("[v0] Focus chart:", chartId)
            onFocusChart?.(chartId)
          }}
        >
          <ExternalLink className="h-3 w-3" />
          在工作区查看
        </Button>
      </div>
      <div className="flex items-end gap-1.5 h-20">
        {data.map((item, index) => (
          <div key={index} className="flex-1 flex flex-col items-center gap-1">
            <div
              className="w-full bg-gradient-to-t from-blue-500 to-blue-400 rounded-t-sm transition-all hover:from-blue-600 hover:to-blue-500"
              style={{ height: `${(item.value / maxValue) * 100}%` }}
            />
            <span className="text-[10px] text-slate-400 truncate w-full text-center">{item.label}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

// File Card Component
function FileCard({ file }: { file: { name: string; size: string; parsed: boolean } }) {
  return (
    <div className="mt-2 flex items-center gap-3 p-3 bg-slate-50 rounded-xl border border-slate-100">
      <div className="h-10 w-10 rounded-lg bg-blue-50 flex items-center justify-center">
        <FileText className="h-5 w-5 text-blue-600" />
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-slate-800 truncate">{file.name}</p>
        <p className="text-xs text-slate-400">{file.size}</p>
      </div>
      {file.parsed && (
        <div className="h-6 w-6 rounded-full bg-green-100 flex items-center justify-center">
          <Check className="h-3.5 w-3.5 text-green-600" />
        </div>
      )}
    </div>
  )
}

// Table Summary Component
function TableSummary({ data }: { data: { rows: number; columns: number; preview: string[] } }) {
  return (
    <div className="mt-3 p-3 bg-slate-50 rounded-xl border border-slate-100">
      <div className="flex items-center gap-4 mb-2">
        <span className="text-xs text-slate-500">
          <span className="font-semibold text-slate-700">{data.rows.toLocaleString()}</span> 行
        </span>
        <span className="text-xs text-slate-500">
          <span className="font-semibold text-slate-700">{data.columns}</span> 列
        </span>
      </div>
      <div className="flex flex-wrap gap-1.5">
        {data.preview.map((col, i) => (
          <Badge key={i} variant="secondary" className="text-xs bg-white border-slate-200">
            {col}
          </Badge>
        ))}
      </div>
    </div>
  )
}

// Message Bubble Component - AI 文本消息支持打字机效果与 citations
function MessageBubble({ 
  message, 
  isLatestAiMessage,
  onFocusChart,
  onFollowUpClick 
}: { 
  message: ChatMessage
  isLatestAiMessage?: boolean
  onFocusChart?: (chartId: string) => void
  onFollowUpClick?: (question: string) => void
}) {
  const [expandedCitation, setExpandedCitation] = useState<string | null>(null)
  const [copiedId, setCopiedId] = useState<string | null>(null)
  const [citationsCollapsed, setCitationsCollapsed] = useState(false)
  const isUser = message.role === "user"
  const showTyping = !isUser && message.type === "text" && message.shouldAnimateTyping === true && isLatestAiMessage && message.content.length > 0
  const isThinking = !isUser && message.isThinking && message.content.length === 0
  const hasCitations = !isUser && message.citations && message.citations.length > 0
  
  return (
    <div className={`flex gap-2.5 group ${isUser ? "flex-row-reverse" : "flex-row"}`}>
      {!isUser && (
        <img
          src="/ai-avatar.png"
          alt="猫咪科学家"
          className="h-9 w-9 shrink-0 rounded-xl ring-2 ring-white shadow-sm object-contain"
        />
      )}
      
      <div className={`max-w-[85%] ${isUser ? "items-end" : "items-start"}`}>
        <div
          className={cn(
            "px-4 py-2.5 text-sm leading-relaxed",
            isUser
              ? "bg-slate-100 text-slate-700 rounded-2xl rounded-tr-md shadow-sm"
              : "bg-white border border-slate-100 text-white rounded-2xl rounded-tl-md shadow-sm"
          )}
        >
  {isThinking ? (
            <div className="flex items-center gap-1.5 text-slate-500">
              <div className="flex gap-0.5">
                <span className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bounce" style={{animationDelay: '0ms'}} />
                <span className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bounce" style={{animationDelay: '150ms'}} />
                <span className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bounce" style={{animationDelay: '300ms'}} />
              </div>
              <span className="text-xs">{PROMPTS.SIDEBAR.thinking}</span>
            </div>
          ) : showTyping ? (
            <TypingMessage
              content={message.content}
              isLatest={true}
              className="text-slate-700"
              speed={40}
            />
          ) : (
            <MarkdownRenderer content={message.content} className="text-slate-700" />
          )}
          
          {/* Chart Widget */}
          {message.type === "chart_widget" && message.data?.chartData && (
            <MiniChart 
              data={message.data.chartData} 
              chartId={message.data.chartId || "chart-1"}
              onFocusChart={onFocusChart}
            />
          )}
          
          {/* File Upload Card */}
          {message.type === "file_upload" && message.data?.file && (
            <FileCard file={message.data.file} />
          )}
          
          {/* Table Summary */}
          {message.type === "table_summary" && message.data?.tableSummary && (
            <TableSummary data={message.data.tableSummary} />
          )}

          {/* Image attachments */}
          {message.data?.imageAttachments && message.data.imageAttachments.length > 0 && (
            <div className="mt-3 grid grid-cols-2 gap-2">
              {message.data.imageAttachments.map((img, idx) => (
                <div key={`${img.name}-${idx}`} className="overflow-hidden rounded-xl border border-slate-200 bg-slate-50">
                  <img src={img.url} alt={img.name} className="h-28 w-full object-cover" />
                  <div className="px-2 py-1">
                    <p className="truncate text-[11px] text-slate-600">{img.name}</p>
                  </div>
                </div>
              ))}
            </div>
          )}

          {message.data?.fileAttachments && message.data.fileAttachments.length > 0 && (
            <div className="mt-3 space-y-2">
              {message.data.fileAttachments.map((f, idx) => (
                <div key={`${f.name}-${idx}`} className="flex items-center gap-2 rounded-xl border border-slate-200 bg-slate-50 px-2.5 py-2">
                  <FileText className="h-4 w-4 text-slate-500" />
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-xs font-medium text-slate-700">{f.name}</p>
                    <p className="text-[11px] text-slate-500">{Math.ceil(f.size / 1024)} KB</p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
        
        {/* Citation cards - 可折叠 */}
        {hasCitations && (
          <Collapsible open={!citationsCollapsed} onOpenChange={v => setCitationsCollapsed(!v)}>
            <CollapsibleTrigger asChild>
              <button className="flex items-center gap-1 mt-2 text-xs text-slate-500 hover:text-slate-700">
                {citationsCollapsed ? <ChevronDown className="w-3 h-3" /> : <ChevronUp className="w-3 h-3" />}
                图表引用 ({message.citations!.length})
              </button>
            </CollapsibleTrigger>
            <CollapsibleContent>
              <div className="mt-2 space-y-2">
                {message.citations!.map((cit, i) => (
                  <div
                    key={i}
                    className="p-3 bg-slate-50 rounded-xl border border-slate-100 overflow-hidden"
                  >
                    <p className="text-xs font-medium text-slate-700 mb-1">{cit.caption}</p>
                    {cit.analysis && <p className="text-xs text-slate-500 mb-2">{cit.analysis}</p>}
                    <div
                      className="relative cursor-pointer rounded-lg overflow-hidden border border-slate-200"
                      onClick={() => setExpandedCitation(prev => (prev === cit.url ? null : cit.url))}
                    >
                      <img
                        src={cit.url}
                        alt={cit.caption}
                        className="w-full h-auto max-h-40 object-contain"
                      />
                      <span className="absolute top-1 right-1 p-1 bg-white/80 rounded">
                        <Expand className="w-3 h-3" />
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </CollapsibleContent>
          </Collapsible>
        )}
        
        {/* 全屏查看图表 */}
        {expandedCitation && (
          <div
            className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4"
            onClick={() => setExpandedCitation(null)}
          >
            <button className="absolute top-4 right-4 p-2 bg-white/20 rounded-full hover:bg-white/40">
              <X className="w-5 h-5 text-white" />
            </button>
            <img
              src={expandedCitation}
              alt="查看图表"
              className="max-w-full max-h-full object-contain"
              onClick={e => e.stopPropagation()}
            />
          </div>
        )}
        
        {/* Follow-up Questions */}
        {!isUser && message.followUpQuestions && message.followUpQuestions.length > 0 && (
          <div className="flex flex-wrap gap-1.5 mt-2">
            {message.followUpQuestions.map((question, i) => (
              <Badge
                key={i}
                variant="outline"
                className="text-xs cursor-pointer bg-white hover:bg-blue-50 hover:border-blue-200 hover:text-blue-700 transition-all"
                onClick={() => onFollowUpClick?.(question)}
              >
                {question}
              </Badge>
            ))}
          </div>
        )}
        
        <p className={`text-[10px] text-slate-400 mt-1.5 ${isUser ? "text-right" : "text-left"}`}>
          {message.timestamp}
        </p>
      </div>
    </div>
  )
}

export function AiSidebar({
  draftKey,
  mode = "full",
  chatHistory,
  onSendMessage,
  onDataUploaded,
  onFocusChart,
  onClearChat,
  onCollapse,
  dynamicSuggestions,
  onNewConversation,
  onSelectConversation,
  activeConversationId,
  conversations,
  onPinConversation,
  onDeleteConversation,
  onRestoreConversation,
  consultationBusy = false,
  historyStoredInDatabase = false,
  sending = false,
  inputDisabled = false,
}: AiSidebarProps) {
  const advisoryOnly = mode === "advisory"
  const [showHistory, setShowHistory] = useState(false)
  const [showDeleted, setShowDeleted] = useState(false)
  const isResponding = sending || chatHistory.some(message => message.isThinking)
  const visibleConversations = [...(conversations ?? [])].filter(thread => showDeleted ? thread.deletedAt : !thread.deletedAt)
    .sort((a, b) => Number(!!b.pinned) - Number(!!a.pinned) || b.updatedAt.localeCompare(a.updatedAt))
  const [inputValue, setInputValue] = useProjectDraft(`${draftKey}:input`, "")
  const [isDragOver, setIsDragOver] = useState(false)
  const [pendingImages, setPendingImages] = useState<PendingImageAttachment[]>([])
  const [pendingFiles, setPendingFiles] = useState<PendingFileAttachment[]>([])
  // 结果页选中的图表以"引用"形式挂在输入框上，随下一条消息一起发送。
  const [pendingCharts, setPendingCharts] = useProjectDraft<AgentChartDataPayload[]>(`${draftKey}:charts`, [])
  const [chartAnalyses, setChartAnalyses] = useProjectDraft<Record<string, string>>(`${draftKey}:analyses`, {})
  const fileInputRef = useRef<HTMLInputElement>(null)
  const scrollAreaRef = useRef<HTMLDivElement>(null)
  const pendingImagesRef = useRef<PendingImageAttachment[]>([])

  useEffect(() => {
    pendingImagesRef.current = pendingImages
  }, [pendingImages])

  useEffect(() => {
    return () => {
      for (const img of pendingImagesRef.current) {
        URL.revokeObjectURL(img.previewUrl)
      }
    }
  }, [])

  const appendImageAttachments = useCallback((files: File[]) => {
    const imageFiles = files.filter((f) => f.type.startsWith("image/"))
    if (imageFiles.length === 0) return

    const next = imageFiles.map((file) => ({
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      file,
      previewUrl: URL.createObjectURL(file),
    }))

    setPendingImages((prev) => [...prev, ...next])
  }, [])

  // 将图片 URL 转换为 base64 dataUrl（通过后端代理解决 CORS）
  const fetchImageAsDataUrl = useCallback(async (url: string, name: string, dataset?: string, path?: string): Promise<SendMessageImage> => {
    const token = localStorage.getItem("access_token")
    let proxyUrl: string
    if (path && dataset) {
      proxyUrl = `${API_BASE}/api/results/${encodeURIComponent(dataset)}/visualizations/image?path=${encodeURIComponent(path)}`
    } else if (url.startsWith("/")) {
      proxyUrl = `${API_BASE}${url}`
    } else {
      proxyUrl = url
    }
    const response = await fetch(proxyUrl, {
      headers: { Authorization: `Bearer ${token}` },
    })
    if (!response.ok) throw new Error("Failed to fetch image")
    const blob = await response.blob()
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => resolve({
        name,
        mimeType: blob.type || "image/png",
        size: blob.size,
        dataUrl: reader.result as string,
      })
      reader.onerror = reject
      reader.readAsDataURL(blob)
    })
  }, [])

  // 图表卡片只发送生成该图的结构化数据，不发送图片像素。
  // 选中后先变成输入框上方的"引用"，由用户自己决定何时提问、问什么。
  useEffect(() => {
    const handleChartDataToChat = (e: Event) => {
      const charts = (e as CustomEvent<AgentChartDataPayload[]>).detail
      if (!charts?.length) return
      setPendingCharts((prev) => {
        const merged = [...prev]
        for (const chart of charts) {
          const index = merged.findIndex((item) => item.chartName === chart.chartName)
          if (index >= 0) merged[index] = chart
          else merged.push(chart)
        }
        return merged.slice(-4)
      })
    }
    const handleChartAnalysis = (e: Event) => {
      const detail = (e as CustomEvent<{ chartName?: string; analysis?: string }>).detail
      if (!detail?.chartName || !detail?.analysis) return
      setChartAnalyses((prev) => ({ ...prev, [detail.chartName!]: detail.analysis! }))
    }

    window.addEventListener("theta:chart-data-to-chat", handleChartDataToChat)
    window.addEventListener("theta:chart-analysis", handleChartAnalysis)
    return () => {
      window.removeEventListener("theta:chart-data-to-chat", handleChartDataToChat)
      window.removeEventListener("theta:chart-analysis", handleChartAnalysis)
    }
  }, [onSendMessage])

  const removePendingChart = useCallback((chartName: string) => {
    setPendingCharts((prev) => prev.filter((chart) => chart.chartName !== chartName))
  }, [])

  const removePendingImage = useCallback((id: string) => {
    setPendingImages((prev) => {
      const target = prev.find((p) => p.id === id)
      if (target) URL.revokeObjectURL(target.previewUrl)
      return prev.filter((p) => p.id !== id)
    })
  }, [])

  const appendFileAttachments = useCallback((files: File[]) => {
    const attachmentFiles = files.filter(isChatAttachmentFile)
    if (attachmentFiles.length === 0) return

    const next = attachmentFiles.map((file) => ({
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      file,
    }))
    setPendingFiles((prev) => [...prev, ...next])
  }, [])

  const removePendingFile = useCallback((id: string) => {
    setPendingFiles((prev) => prev.filter((p) => p.id !== id))
  }, [])

  // Handle file upload
  const handleFileUpload = useCallback((file: File) => {
    console.log("[v0] File uploaded:", file.name, file.size)
    
    // Trigger callback to main workspace
    onDataUploaded?.(file)
    
    // Send a message about the upload
    onSendMessage(`已上传文件: ${file.name}`)
  }, [onDataUploaded, onSendMessage])

  // Drag and drop handlers
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragOver(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragOver(false)
  }, [])

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragOver(false)

    const files = Array.from(e.dataTransfer.files)
    if (files.length > 0) {
      const imageFiles = files.filter((f) => f.type.startsWith("image/"))
      const attachmentFiles = files.filter((f) => !f.type.startsWith("image/"))

      if (imageFiles.length > 0) {
        appendImageAttachments(imageFiles)
      }

      if (attachmentFiles.length > 0) {
        appendFileAttachments(attachmentFiles)
      }
      return
    }

    // Support dropping chart bundles/text from result visualization cards.
    const customPayload = e.dataTransfer.getData("application/x-theta-ai-drop")
    if (customPayload) {
      try {
        const parsed = JSON.parse(customPayload) as { type?: string; text?: string; url?: string; name?: string; path?: string; dataset?: string }
        if (parsed?.type === "image" && parsed?.url) {
          // 拖拽的是图片，通过后端代理获取 base64 发送给 AI 分析
          try {
            const imageAttachment = await fetchImageAsDataUrl(parsed.url, parsed.name || "image", parsed.dataset, parsed.path)
            onSendMessage({ content: "请分析以下图片：", images: [imageAttachment] })
          } catch {
            // 回退到发送 URL
            const imageMarkdown = `![${parsed.name || "image"}](${parsed.url})`
            onSendMessage(`请分析以下图片：\n${imageMarkdown}`)
          }
          return

        }
        if (parsed?.text) {
          onSendMessage(parsed.text)
          return
        }
      } catch {
        // Fallback to plain text handling below.
      }
    }

    // 尝试处理图片 URL（从外部拖拽图片到聊天框）
    const uriList = e.dataTransfer.getData("text/uri-list")
    if (uriList) {
      const urls = uriList.split("\n").filter((u) => u.trim() && !u.startsWith("#"))
      if (urls.length > 0) {
        const imageUrl = urls[0]
        if (imageUrl.match(/\.(png|jpg|jpeg|gif|webp)$/i) || imageUrl.includes("/visualization/") || imageUrl.includes("/oss")) {
          // 直接发送图片 URL 给 AI 分析
          const imageName = imageUrl.split("/").pop() || "image"
          const imageMarkdown = `![${imageName}](${imageUrl})`
          onSendMessage(`请分析以下图片：\n${imageMarkdown}`)
          return
        }
      }
    }

    const plainText = e.dataTransfer.getData("text/plain")
    if (plainText && plainText.trim()) {
      onSendMessage(plainText.trim())
    }
  }, [appendFileAttachments, appendImageAttachments, onSendMessage, fetchImageAsDataUrl])

  // Handle paperclip click
  const handleAttachClick = useCallback(() => {
    fileInputRef.current?.click()
  }, [])

  const handleFileInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    if (files.length > 0) {
      const imageFiles = files.filter((f) => f.type.startsWith("image/"))
      const attachmentFiles = files.filter((f) => !f.type.startsWith("image/"))

      if (imageFiles.length > 0) {
        appendImageAttachments(imageFiles)
      }

      if (attachmentFiles.length > 0) {
        appendFileAttachments(attachmentFiles)
      }
    }
    // Reset input
    e.target.value = ""
  }, [appendFileAttachments, appendImageAttachments])

  // Handle send message
  const handleSend = useCallback(async () => {
    const trimmed = inputValue.trim()
    if (!trimmed && pendingImages.length === 0 && pendingFiles.length === 0 && pendingCharts.length === 0) {
      return
    }
    const analyses = pendingCharts
      .map((chart) => ({ chartName: chart.chartName, analysis: chartAnalyses[chart.chartName] }))
      .filter((item): item is { chartName: string; analysis: string } => Boolean(item.analysis))

    if (pendingImages.length === 0 && pendingFiles.length === 0) {
      const accepted = await onSendMessage({
        content: trimmed || "请基于我引用的原始绘图数据写一段研究结论。",
        ...(pendingCharts.length ? { charts: pendingCharts } : {}),
        ...(analyses.length ? { chartAnalyses: analyses } : {}),
      })
      if (accepted !== false) { setInputValue(""); setPendingCharts([]) }
      return
    }

    try {
      const images = await Promise.all(
        pendingImages.map(async (img) => ({
          name: img.file.name,
          mimeType: img.file.type || "image/png",
          size: img.file.size,
          dataUrl: await fileToDataUrl(img.file),
        }))
      )
      const files = await Promise.all(
        pendingFiles.map(async (doc) => ({
          name: doc.file.name,
          mimeType: doc.file.type || "application/octet-stream",
          size: doc.file.size,
          dataUrl: await fileToDataUrl(doc.file),
        }))
      )
      const accepted = await onSendMessage({
        content: trimmed,
        images,
        files,
        ...(pendingCharts.length ? { charts: pendingCharts } : {}),
        ...(analyses.length ? { chartAnalyses: analyses } : {}),
      })
      if (accepted === false) return
      setInputValue("")
      setPendingImages((prev) => {
        prev.forEach((p) => URL.revokeObjectURL(p.previewUrl))
        return []
      })
      setPendingFiles([])
      setPendingCharts([])
    } catch (error) {
      console.error("[v0] Failed to send image attachments:", error)
    }
  }, [chartAnalyses, inputValue, onSendMessage, pendingCharts, pendingFiles, pendingImages])

  // Handle follow-up click
  const handleFollowUpClick = useCallback((question: string) => {
    onSendMessage(question)
  }, [onSendMessage])

  // Handle key press
  const handleKeyPress = useCallback((e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing && !isResponding) {
      e.preventDefault()
      handleSend()
    }
  }, [handleSend, isResponding])

  return (
    <aside 
      aria-label="项目咨询助手"
      className={`${styles.sidebar} w-full min-w-0 h-full flex flex-col bg-gradient-to-b from-white to-slate-50/30 border-l border-slate-200/60 shadow-lg shadow-slate-200/30 relative overflow-hidden`}
      style={{ maxWidth: "100%" }}
      onDragOver={advisoryOnly ? undefined : handleDragOver}
      onDragLeave={advisoryOnly ? undefined : handleDragLeave}
      onDrop={advisoryOnly ? undefined : handleDrop}
    >
      {/* Drag Overlay */}
      {!advisoryOnly && isDragOver && (
        <div className="absolute inset-0 z-50 bg-gradient-to-br from-blue-500/15 to-indigo-500/15 backdrop-blur-md border-2 border-dashed border-blue-400 rounded-lg flex flex-col items-center justify-center">
          <div className="h-20 w-20 rounded-2xl bg-gradient-to-br from-blue-100 to-indigo-100 flex items-center justify-center mb-4 shadow-xl shadow-blue-200/50">
            <Upload className="h-10 w-10 text-blue-600" />
          </div>
          <p className="text-lg font-bold text-blue-700">拖放内容到此处</p>
          <p className="text-sm text-blue-500/80 mt-1.5 font-medium">支持文件，或可视化卡片/多选图表</p>
        </div>
      )}

      {/* Hidden File Input */}
      {!advisoryOnly && (
        <input
          ref={fileInputRef}
          type="file"
          className="hidden"
          multiple
          onChange={handleFileInputChange}
        />
      )}

      {/* Header - constrained to sidebar width */}
      <div className="h-14 flex-shrink-0 border-b border-slate-100/80 px-3 sm:px-4 flex items-center justify-between bg-white/80 backdrop-blur-sm min-w-0 w-full">
        <div className="flex items-center gap-2.5 sm:gap-3 min-w-0 overflow-hidden">
          <img
          src="/ai-avatar.png"
          alt="猫咪科学家"
          className="h-9 w-9 sm:h-10 sm:w-10 rounded-xl ring-2 ring-white shadow-sm object-contain"
        />
          <div className="flex items-center gap-2 min-w-0">
            <span className="font-bold text-slate-800 text-sm sm:text-base tracking-tight">猫咪科学家</span>
            <div className="hidden sm:flex items-center gap-1.5 px-2 py-0.5 bg-emerald-50 rounded-full">
              <div className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
              <span className="text-[10px] text-emerald-600 font-semibold">{advisoryOnly ? "咨询与解读" : "在线"}</span>
            </div>
          </div>
        </div>
        <div className="flex items-center gap-1">
          {!advisoryOnly && <Button
            variant="ghost" 
            size="icon" 
            className="h-8 w-8 text-slate-400 hover:text-red-500 hover:bg-red-50 bg-transparent rounded-xl transition-all duration-200"
            onClick={onClearChat}
            title="清除对话"
          >
            <Trash2 className="h-3.5 w-3.5" />
          </Button>}
          {onCollapse && (
            <Button 
              variant="ghost" 
              size="icon" 
              className="h-8 w-8 text-slate-400 hover:text-slate-600 hover:bg-slate-100 bg-transparent rounded-xl transition-all duration-200"
              onClick={onCollapse}
              title="收起边栏"
            >
              <PanelRightClose className="h-3.5 w-3.5" />
            </Button>
          )}
        </div>
      </div>

      {/* These consultations are independent of the main training conversations. */}
      <div className={styles.toolbar}>
        {advisoryOnly ? (
          <>
            <button type="button" disabled={consultationBusy} onClick={onNewConversation} className={styles.newConversation}><Plus size={14} />新建咨询</button>
            <button type="button" onClick={() => setShowHistory(value => !value)} aria-expanded={showHistory} className={styles.historyButton}><Clock size={14} />咨询历史<span>{conversations?.filter(thread => !thread.deletedAt).length ?? 0}</span></button>
          </>
        ) : (
          <button className="flex items-center gap-1.5 text-sm text-slate-700 hover:text-slate-900 transition-colors font-semibold">
            <Clock className="h-3.5 w-3.5" />
            <span>历史</span>
            <span className="ml-0.5 px-2 py-0.5 bg-gradient-to-r from-blue-500 to-indigo-500 text-white text-[10px] rounded-full font-bold shadow-sm">{chatHistory.length}</span>
          </button>
        )}
      </div>
      {showHistory && <section className={styles.history} aria-label="当前项目咨询历史">
        <div className={styles.historyHeading}><p>仅当前项目 · {historyStoredInDatabase ? '已保存到数据库' : '保存在本浏览器'}</p>
          {onRestoreConversation && <button type="button" onClick={() => setShowDeleted(value => !value)} aria-pressed={showDeleted}>{showDeleted ? '返回历史' : `已删除 ${conversations?.filter(t => t.deletedAt).length ?? 0}`}</button>}
        </div>
        {visibleConversations.length === 0 && <p>{showDeleted ? '暂无已删除咨询' : '暂无咨询记录，可点击上方新建'}</p>}
        {visibleConversations.map(thread => <div className={styles.historyRow} key={thread.id} aria-label={thread.title}>
          <button type="button" className={styles.historySelect} disabled={consultationBusy || !!thread.deletedAt}
            aria-current={thread.id === activeConversationId ? 'true' : undefined}
            onClick={() => { onSelectConversation?.(thread.id); setShowHistory(false) }}>
            <strong>{thread.pinned && <Pin size={12} aria-label="已置顶" />}{thread.title}</strong>
            <span>{thread.messageCount} 条提问 · {new Date(thread.updatedAt).toLocaleString('zh-CN', { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' })}</span>
          </button>
          <div className={styles.historyActions}>
            {thread.deletedAt ? <button type="button" disabled={consultationBusy} title="恢复咨询" aria-label={`恢复咨询：${thread.title}`} onClick={() => onRestoreConversation?.(thread.id)}><RotateCcw size={14} /></button> : <>
              {onPinConversation && <button type="button" disabled={consultationBusy} title={thread.pinned ? '取消置顶' : '置顶'} aria-label={`${thread.pinned ? '取消置顶' : '置顶'}：${thread.title}`} onClick={() => onPinConversation(thread.id, !thread.pinned)}>{thread.pinned ? <PinOff size={14} /> : <Pin size={14} />}</button>}
              {onDeleteConversation && <button type="button" disabled={consultationBusy} title="删除咨询（可恢复）" aria-label={`删除咨询：${thread.title}`} onClick={() => onDeleteConversation(thread.id)}><Trash2 size={14} /></button>}
            </>}
          </div>
        </div>)}
      </section>}

      {/* Chat Body - flex-1 min-h-0 so it shrinks */}
      <ScrollArea className="flex-1 min-h-0 min-w-0 overflow-hidden" ref={scrollAreaRef}>
        {chatHistory.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-72 px-8">
            <div className="relative">
              <div className="h-28 w-28 rounded-3xl overflow-hidden mb-6 flex items-center justify-center bg-slate-100/50">
                <img src="/ai-avatar.png" alt="猫咪科学家" className="w-[140%] h-[140%] object-contain scale-75" />
              </div>
              {onNewConversation && <button type="button" disabled={consultationBusy} aria-label="新建咨询对话" title="新建咨询对话" onClick={onNewConversation}
                className="absolute -top-1 -right-1 h-7 w-7 rounded-full bg-emerald-500 text-white flex items-center justify-center shadow-sm hover:bg-emerald-600 focus-visible:outline-2 focus-visible:outline-emerald-600 focus-visible:outline-offset-2">
                <Plus size={16} />
              </button>}
            </div>
            <p className="text-sm text-slate-700 text-center leading-relaxed font-semibold">
              {advisoryOnly ? "咨询当前项目，或解读已选中的结果" : "开始对话，让猫咪科学家帮您分析数据"}
            </p>
            <p className="text-xs text-slate-400 text-center mt-2 font-medium">
              {advisoryOnly ? "引用图表后可继续提问；训练与参数配置请返回工作台" : "您也可以直接拖放文件到此处"}
            </p>
          </div>
        ) : (
          <div className="p-4 space-y-4">
            {(() => {
              return chatHistory.map((message, index) => (
                <MessageBubble
                  key={message.id}
                  message={message}
                  isLatestAiMessage={index === chatHistory.length - 1 && message.role === "ai"}
                  onFocusChart={onFocusChart}
                  onFollowUpClick={handleFollowUpClick}
                />
              ))
            })()}
          </div>
        )}

      {/* Smart Suggestions - 动态或静态 */}
      {chatHistory.length === 0 && (
        <div className="flex-shrink-0 px-3 py-2.5 border-t border-slate-100/60 bg-gradient-to-b from-white via-white to-slate-50/50 min-w-0 w-full overflow-hidden">
          <div className="flex items-center gap-1.5 mb-2">
            <div className="h-5 w-5 rounded-md bg-gradient-to-br from-amber-100 to-orange-100 flex items-center justify-center">
              <Zap className="h-3 w-3 text-amber-600" />
            </div>
            <span className="text-[10px] font-bold text-slate-500 tracking-wide uppercase">
              {dynamicSuggestions && dynamicSuggestions.length > 0 ? "分析建议" : "智能建议"}
            </span>
          </div>
          <div className="space-y-1.5">
            {dynamicSuggestions && dynamicSuggestions.length > 0 ? (
              dynamicSuggestions.map((s, i) => (
                <button
                  key={i}
                  className="w-full flex items-center justify-between p-2.5 bg-white hover:bg-blue-50/80 border border-slate-200/60 hover:border-blue-200 rounded-lg transition-all duration-200 group shadow-sm"
                  onClick={s.onClick}
                >
                  <div className="flex flex-col items-start gap-0 min-w-0 flex-1">
                    <span className="text-xs font-semibold text-slate-800 group-hover:text-blue-700 transition-colors">{s.title}</span>
                    <span className="text-[10px] text-slate-400 group-hover:text-blue-500/70 transition-colors truncate max-w-full">{s.description}</span>
                  </div>
                  <div className="h-6 w-6 shrink-0 rounded-md bg-slate-100 group-hover:bg-blue-100 flex items-center justify-center transition-colors">
                    <BarChart3 className="h-3.5 w-3.5 text-slate-400 group-hover:text-blue-600 transition-colors" />
                  </div>
                </button>
              ))
            ) : advisoryOnly ? (
              [
                { id: "result", title: "解读已选结果", description: "先在结果页选中图表，再发送到这里", prompt: "请说明怎样选择结果并发送过来解读。" },
                { id: "metrics", title: "理解评估指标", description: "解释指标含义、方向和使用边界", prompt: "请解释当前项目常见评估指标应该怎样阅读。" },
                { id: "compare", title: "比较模型结果", description: "讨论模型差异，不执行新的训练", prompt: "请告诉我应该怎样比较当前项目中已有模型的结果。" },
              ].map((suggestion, i) => (
                <button
                  key={suggestion.id}
                  className="w-full flex items-center justify-between p-2.5 bg-white hover:bg-blue-50/80 border border-slate-200/60 hover:border-blue-200 rounded-lg transition-all duration-200 group shadow-sm"
                  onClick={() => onSendMessage(suggestion.prompt)}
                >
                  <div className="flex flex-col items-start gap-0 min-w-0">
                    <span className="text-xs font-semibold text-slate-800 group-hover:text-blue-700 transition-colors">{suggestion.title}</span>
                    <span className="text-[10px] text-slate-400 group-hover:text-blue-500/70 transition-colors truncate max-w-full">{suggestion.description}</span>
                  </div>
                  <div className="h-6 w-6 shrink-0 rounded-md bg-slate-100 group-hover:bg-blue-100 flex items-center justify-center transition-colors">
                    {i === 0 ? <BarChart3 className="h-3.5 w-3.5 text-slate-400 group-hover:text-blue-600 transition-colors" /> : <MessageSquare className="h-3.5 w-3.5 text-slate-400 group-hover:text-blue-600 transition-colors" />}
                  </div>
                </button>
              ))
            ) : (
              PROMPTS.SIDEBAR.suggestions.map((suggestion, i) => (
                <button
                  key={suggestion.id}
                  className="w-full flex items-center justify-between p-2.5 bg-white hover:bg-blue-50/80 border border-slate-200/60 hover:border-blue-200 rounded-lg transition-all duration-200 group shadow-sm"
                  onClick={() => onSendMessage(suggestion.prompt)}
                >
                  <div className="flex flex-col items-start gap-0 min-w-0">
                    <span className="text-xs font-semibold text-slate-800 group-hover:text-blue-700 transition-colors">{suggestion.title}</span>
                    <span className="text-[10px] text-slate-400 group-hover:text-blue-500/70 transition-colors truncate max-w-full">{suggestion.description}</span>
                  </div>
                  <div className="h-6 w-6 shrink-0 rounded-md bg-slate-100 group-hover:bg-blue-100 flex items-center justify-center transition-colors">
                    {i === 0 ? (
                      <BarChart3 className="h-3.5 w-3.5 text-slate-400 group-hover:text-blue-600 transition-colors" />
                    ) : (
                      <Zap className="h-3.5 w-3.5 text-slate-400 group-hover:text-blue-600 transition-colors" />
                    )}
                  </div>
                </button>
              ))
            )}
          </div>
        </div>
      )}
      </ScrollArea>
      {/* Footer Input - strict width so it never overflows at any zoom/size */}
      <div className={styles.footer}>
        <div className={styles.composer}>
          <div className={styles.inputBody}>
            <div className={styles.attachments}>
            {pendingImages.length > 0 && (
              <div className="mb-2 grid grid-cols-3 gap-2">
                {pendingImages.map((img) => (
                  <div key={img.id} className="relative overflow-hidden rounded-xl border border-slate-200 bg-slate-50">
                    <img src={img.previewUrl} alt={img.file.name} className="h-20 w-full object-cover" />
                    <button
                      type="button"
                      className="absolute right-1 top-1 rounded-full bg-black/55 p-1 text-white hover:bg-black/70"
                      onClick={() => removePendingImage(img.id)}
                      title="移除图片"
                    >
                      <X className="h-3 w-3" />
                    </button>
                  </div>
                ))}
              </div>
            )}
            {pendingFiles.length > 0 && (
              <div className="mb-2 space-y-1.5">
                <div className="px-1 text-[11px] font-medium text-slate-500">已添加附件</div>
                {pendingFiles.map((doc) => (
                  <div key={doc.id} className="flex items-center gap-2 rounded-xl border border-slate-200 bg-slate-50 px-2.5 py-2">
                    <FileText className="h-4 w-4 text-slate-500" />
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-xs text-slate-700">{doc.file.name}</p>
                      <p className="text-[11px] text-slate-500">{Math.max(1, Math.ceil(doc.file.size / 1024))} KB{doc.file.type ? ` · ${doc.file.type}` : ""}</p>
                    </div>
                    <button
                      type="button"
                      className="rounded-full bg-slate-200 p-1 text-slate-600 hover:bg-slate-300"
                      onClick={() => removePendingFile(doc.id)}
                      title="移除附件"
                    >
                      <X className="h-3 w-3" />
                    </button>
                  </div>
                ))}
              </div>
            )}
            {pendingCharts.length > 0 && (
              <div className="mb-2 space-y-1.5">
                <div className="px-1 text-[11px] font-medium text-slate-500">引用中的结果数据</div>
                {pendingCharts.map((chart) => (
                  <div key={chart.chartName} className="flex items-center gap-2 rounded-xl border border-blue-100 bg-blue-50/70 px-2.5 py-2">
                    <Quote className="h-3.5 w-3.5 shrink-0 text-blue-500" />
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-xs text-slate-700">{chart.chartName}</p>
                      <p className="text-[11px] text-slate-500">
                        {chart.sources.length} 个绘图数据源{chartAnalyses[chart.chartName] ? " · 含已生成解读" : ""}
                      </p>
                    </div>
                    <button
                      type="button"
                      className="rounded-full bg-white p-1 text-slate-500 hover:bg-slate-100"
                      onClick={() => removePendingChart(chart.chartName)}
                      title="移除引用"
                    >
                      <X className="h-3 w-3" />
                    </button>
                  </div>
                ))}
              </div>
            )}
            </div>
            <Textarea
              aria-label="项目咨询消息"
              data-composer-input
              disabled={inputDisabled}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyPress}
              placeholder={advisoryOnly ? "咨询当前项目，或从结果页引用图表后提问…" : "输入消息或拖放文件..."}
              className={styles.textarea}
            />
          </div>
          <div className="flex items-center justify-between gap-1 sm:gap-2 px-2 sm:px-3 py-2 border-t border-slate-100/60 min-w-0 w-full max-w-full box-border">
            <div className="flex items-center gap-0.5 shrink-0 min-w-0">
              {!advisoryOnly && <Button
                variant="ghost" 
                size="icon" 
                className="h-8 w-8 shrink-0 text-slate-400 hover:text-blue-600 hover:bg-blue-50 bg-transparent rounded-xl"
                onClick={handleAttachClick}
                title="上传文件"
              >
                <Paperclip className="h-4 w-4" />
              </Button>}
              {!advisoryOnly && <Button
                variant="ghost" 
                size="icon" 
                className="h-8 w-8 shrink-0 text-slate-400 hover:text-blue-600 hover:bg-blue-50 bg-transparent rounded-xl"
                title="语音输入"
              >
                <Mic className="h-4 w-4" />
              </Button>}
              {advisoryOnly && <span className="px-1 text-[10px] font-medium text-slate-400">只读咨询</span>}
            </div>
            <Button 
              size="sm"
              className="h-8 px-3 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 text-white hover:from-blue-700 hover:to-indigo-700 shadow-md disabled:opacity-50 disabled:cursor-not-allowed text-xs font-semibold shrink-0"
              onClick={handleSend}
              disabled={inputDisabled || isResponding || (!inputValue.trim() && pendingImages.length === 0 && pendingFiles.length === 0 && pendingCharts.length === 0)}
            >
              <Send className="h-3.5 w-3.5 mr-1" />
              {isResponding ? '回复中' : '发送'}
            </Button>
          </div>
        </div>
      </div>
    </aside>
  )
}
