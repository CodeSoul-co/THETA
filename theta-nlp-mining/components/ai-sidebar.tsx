"use client"

import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Bot, Info, CheckCircle2, AlertCircle, Lightbulb, BarChart3, FileText } from "lucide-react"

interface Message {
  type: string
  text: string
}

interface AISidebarProps {
  title: string
  subtitle: string
  messages: Message[]
}

export function AISidebar({ title, subtitle, messages }: AISidebarProps) {
  const getIcon = (type: string) => {
    switch (type) {
      case "info":
        return <Info className="w-4 h-4 text-blue-500" />
      case "success":
        return <CheckCircle2 className="w-4 h-4 text-green-500" />
      case "tip":
        return <Lightbulb className="w-4 h-4 text-yellow-500" />
      case "metric":
        return <BarChart3 className="w-4 h-4 text-purple-500" />
      case "insight":
        return <AlertCircle className="w-4 h-4 text-orange-500" />
      case "recommendation":
        return <Lightbulb className="w-4 h-4 text-cyan-500" />
      case "framework":
        return <FileText className="w-4 h-4 text-indigo-500" />
      case "evidence":
        return <CheckCircle2 className="w-4 h-4 text-emerald-500" />
      case "source":
        return <Info className="w-4 h-4 text-blue-500" />
      default:
        return <Info className="w-4 h-4" />
    }
  }

  return (
    <div className="w-[28%] border-l border-border bg-muted/20 flex flex-col">
      <div className="p-6 border-b border-border">
        <div className="flex items-center gap-3 mb-2">
          <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center">
            <Bot className="w-6 h-6 text-primary" />
          </div>
          <div>
            <h2 className="font-semibold text-foreground">{title}</h2>
            <p className="text-xs text-muted-foreground">{subtitle}</p>
          </div>
        </div>
        <Badge variant="secondary" className="text-xs">
          AI 协作者模式
        </Badge>
      </div>

      <ScrollArea className="flex-1 p-6">
        <div className="space-y-4">
          {messages.map((msg, idx) => (
            <Card key={idx} className="bg-card/50">
              <CardContent className="p-4">
                <div className="flex items-start gap-3">
                  {getIcon(msg.type)}
                  <p className="text-sm text-foreground leading-relaxed flex-1">{msg.text}</p>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </ScrollArea>

      <div className="p-6 border-t border-border">
        <p className="text-xs text-muted-foreground text-center">点击左侧元素触发 AI 对话分析</p>
      </div>
    </div>
  )
}
