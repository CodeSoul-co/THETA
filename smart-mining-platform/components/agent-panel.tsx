"use client"

import { useState, useEffect, useRef } from "react"
import { Card, CardContent, CardHeader } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {
  Bot,
  Database,
  BarChart2,
  GraduationCap,
  Send,
  CheckCircle2,
  AlertCircle,
  Loader2,
  ChevronDown,
  ChevronUp,
} from "lucide-react"

interface AgentMessage {
  id: string
  agent: "data" | "analyst" | "expert"
  type: "log" | "result" | "warning"
  content: string
  timestamp: Date
  metrics?: Record<string, string | number>
}

interface AgentPanelProps {
  triggerDataAgent: boolean
  triggerAnalysisAgent: boolean
}

export function AgentPanel({ triggerDataAgent, triggerAnalysisAgent }: AgentPanelProps) {
  const [messages, setMessages] = useState<AgentMessage[]>([])
  const [expandedAgent, setExpandedAgent] = useState<string | null>("data")
  const [userInput, setUserInput] = useState("")
  const scrollRef = useRef<HTMLDivElement>(null)

  const agents = {
    data: {
      name: "数据管家",
      icon: Database,
      color: "text-primary",
      bgColor: "bg-primary/10",
    },
    analyst: {
      name: "建模分析师",
      icon: BarChart2,
      color: "text-accent",
      bgColor: "bg-accent/10",
    },
    expert: {
      name: "领域专家",
      icon: GraduationCap,
      color: "text-chart-3",
      bgColor: "bg-chart-3/10",
    },
  }

  useEffect(() => {
    if (triggerDataAgent) {
      const dataMessages: AgentMessage[] = [
        {
          id: "d1",
          agent: "data",
          type: "log",
          content: "开始接收数据文件...",
          timestamp: new Date(),
        },
        {
          id: "d2",
          agent: "data",
          type: "log",
          content: "检测到 3 个文件，总大小 24.5MB",
          timestamp: new Date(),
        },
        {
          id: "d3",
          agent: "data",
          type: "log",
          content: "执行隐私掩码替换：身份证号、电话号码已脱敏",
          timestamp: new Date(),
        },
        {
          id: "d4",
          agent: "data",
          type: "log",
          content: "Emoji转译完成：共转换 127 个特殊字符",
          timestamp: new Date(),
        },
        {
          id: "d5",
          agent: "data",
          type: "result",
          content: "数据预处理完成",
          timestamp: new Date(),
          metrics: {
            处理文件: "3 个",
            脱敏字段: "892 处",
            噪声剔除: "2.3%",
            数据质量: "98.7%",
          },
        },
      ]

      dataMessages.forEach((msg, idx) => {
        setTimeout(() => {
          setMessages((prev) => [...prev, msg])
        }, idx * 800)
      })
    }
  }, [triggerDataAgent])

  useEffect(() => {
    if (triggerAnalysisAgent) {
      const analysisMessages: AgentMessage[] = [
        {
          id: "a1",
          agent: "analyst",
          type: "log",
          content: "加载 LoRA 医疗健康适配器...",
          timestamp: new Date(),
        },
        {
          id: "a2",
          agent: "analyst",
          type: "log",
          content: "执行 UMAP 降维分析 (n_neighbors=15)",
          timestamp: new Date(),
        },
        {
          id: "a3",
          agent: "analyst",
          type: "log",
          content: "HDBSCAN 聚类完成：识别 4 个主题簇",
          timestamp: new Date(),
        },
        {
          id: "a4",
          agent: "analyst",
          type: "result",
          content: "主题建模分析完成",
          timestamp: new Date(),
          metrics: {
            "Cv Score": "0.847",
            "F1-score": "0.912",
            主题数: "4",
            轮廓系数: "0.723",
          },
        },
        {
          id: "e1",
          agent: "expert",
          type: "log",
          content: "基于聚类结果生成医学维度解释...",
          timestamp: new Date(),
        },
        {
          id: "e2",
          agent: "expert",
          type: "result",
          content: "领域解释报告",
          timestamp: new Date(),
          metrics: {
            主题1: "病理诊断相关文献",
            主题2: "临床症状描述",
            主题3: "分子检测方法",
            主题4: "治疗方案研究",
          },
        },
      ]

      analysisMessages.forEach((msg, idx) => {
        setTimeout(() => {
          setMessages((prev) => [...prev, msg])
        }, idx * 1000)
      })
    }
  }, [triggerAnalysisAgent])

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  const getMessageIcon = (msg: AgentMessage) => {
    if (msg.type === "result") return <CheckCircle2 className="w-3 h-3 text-primary" />
    if (msg.type === "warning") return <AlertCircle className="w-3 h-3 text-chart-5" />
    return <Loader2 className="w-3 h-3 animate-spin text-muted-foreground" />
  }

  const handleSend = () => {
    if (!userInput.trim()) return

    const userMsg: AgentMessage = {
      id: `user-${Date.now()}`,
      agent: "expert",
      type: "log",
      content: `用户查询: ${userInput}`,
      timestamp: new Date(),
    }
    setMessages((prev) => [...prev, userMsg])
    setUserInput("")

    // Simulate response
    setTimeout(() => {
      const response: AgentMessage = {
        id: `response-${Date.now()}`,
        agent: "expert",
        type: "result",
        content: "根据您的查询，系统已从知识库中检索到相关信息并进行分析。",
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, response])
    }, 1500)
  }

  const groupedMessages = messages.reduce(
    (acc, msg) => {
      if (!acc[msg.agent]) acc[msg.agent] = []
      acc[msg.agent].push(msg)
      return acc
    },
    {} as Record<string, AgentMessage[]>,
  )

  return (
    <aside className="w-80 h-full bg-sidebar border-l border-sidebar-border flex flex-col">
      <div className="px-4 py-4 border-b border-sidebar-border">
        <div className="flex items-center gap-2">
          <Bot className="w-5 h-5 text-primary" />
          <h3 className="font-semibold text-sidebar-foreground">Multi-Agent 交互</h3>
        </div>
        <p className="text-xs text-muted-foreground mt-1">智能协作分析系统</p>
      </div>

      <ScrollArea className="flex-1">
        <div className="p-3 space-y-3">
          {(Object.keys(agents) as Array<keyof typeof agents>).map((agentKey) => {
            const agent = agents[agentKey]
            const agentMessages = groupedMessages[agentKey] || []
            const isExpanded = expandedAgent === agentKey

            return (
              <Card key={agentKey} className="bg-card/50 border-border">
                <CardHeader
                  className="py-2 px-3 cursor-pointer"
                  onClick={() => setExpandedAgent(isExpanded ? null : agentKey)}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div className={`p-1.5 rounded-md ${agent.bgColor}`}>
                        <agent.icon className={`w-3.5 h-3.5 ${agent.color}`} />
                      </div>
                      <span className="text-sm font-medium text-foreground">{agent.name}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      {agentMessages.length > 0 && (
                        <Badge variant="secondary" className="text-xs h-5">
                          {agentMessages.length}
                        </Badge>
                      )}
                      {isExpanded ? (
                        <ChevronUp className="w-4 h-4 text-muted-foreground" />
                      ) : (
                        <ChevronDown className="w-4 h-4 text-muted-foreground" />
                      )}
                    </div>
                  </div>
                </CardHeader>

                {isExpanded && (
                  <CardContent className="py-2 px-3 pt-0">
                    {agentMessages.length === 0 ? (
                      <p className="text-xs text-muted-foreground py-2">等待任务触发...</p>
                    ) : (
                      <div className="space-y-2">
                        {agentMessages.map((msg) => (
                          <div key={msg.id} className="text-xs space-y-1 p-2 rounded-md bg-muted/30">
                            <div className="flex items-center gap-1.5">
                              {getMessageIcon(msg)}
                              <span className="text-foreground">{msg.content}</span>
                            </div>
                            {msg.metrics && (
                              <div className="grid grid-cols-2 gap-1 mt-2 pt-2 border-t border-border">
                                {Object.entries(msg.metrics).map(([key, value]) => (
                                  <div key={key} className="text-muted-foreground">
                                    <span className="text-foreground/70">{key}:</span>{" "}
                                    <span className="text-primary">{value}</span>
                                  </div>
                                ))}
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                  </CardContent>
                )}
              </Card>
            )
          })}
        </div>
        <div ref={scrollRef} />
      </ScrollArea>

      {/* User Input */}
      <div className="p-3 border-t border-sidebar-border">
        <div className="flex gap-2">
          <Input
            placeholder="输入问题..."
            value={userInput}
            onChange={(e) => setUserInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
            className="flex-1 h-8 text-sm bg-input border-border"
          />
          <Button size="icon" className="h-8 w-8" onClick={handleSend}>
            <Send className="w-3.5 h-3.5" />
          </Button>
        </div>
      </div>
    </aside>
  )
}
