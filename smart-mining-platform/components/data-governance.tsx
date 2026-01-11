"use client"

import type React from "react"

import { useState, useCallback } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import {
  Upload,
  FileText,
  Globe,
  BookOpen,
  Shield,
  Smile,
  Filter,
  CheckCircle2,
  Clock,
  AlertCircle,
} from "lucide-react"

interface DataGovernanceProps {
  onUpload: () => void
}

interface ProcessingTask {
  id: string
  name: string
  type: "privacy" | "emoji" | "noise"
  progress: number
  status: "pending" | "processing" | "completed"
}

export function DataGovernance({ onUpload }: DataGovernanceProps) {
  const [isDragging, setIsDragging] = useState(false)
  const [uploadedFiles, setUploadedFiles] = useState<string[]>([])
  const [processingTasks, setProcessingTasks] = useState<ProcessingTask[]>([])

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setIsDragging(false)

      // Simulate file upload
      const files = ["研究报告_2024.pdf", "医学文献_病理学.pdf", "临床案例数据集.csv"]
      setUploadedFiles(files)

      // Start processing simulation
      const tasks: ProcessingTask[] = [
        { id: "1", name: "隐私掩码替换", type: "privacy", progress: 0, status: "processing" },
        { id: "2", name: "Emoji/特殊字符转译", type: "emoji", progress: 0, status: "pending" },
        { id: "3", name: "噪声数据剔除", type: "noise", progress: 0, status: "pending" },
      ]
      setProcessingTasks(tasks)

      // Simulate processing
      simulateProcessing(tasks)
      onUpload()
    },
    [onUpload],
  )

  const simulateProcessing = (tasks: ProcessingTask[]) => {
    let currentTaskIndex = 0

    const processTask = () => {
      if (currentTaskIndex >= tasks.length) return

      const interval = setInterval(() => {
        setProcessingTasks((prev) => {
          const updated = [...prev]
          if (updated[currentTaskIndex].progress < 100) {
            updated[currentTaskIndex].progress += Math.random() * 15 + 5
            updated[currentTaskIndex].status = "processing"
            if (updated[currentTaskIndex].progress >= 100) {
              updated[currentTaskIndex].progress = 100
              updated[currentTaskIndex].status = "completed"
              clearInterval(interval)
              currentTaskIndex++
              if (currentTaskIndex < tasks.length) {
                setTimeout(processTask, 500)
              }
            }
          }
          return updated
        })
      }, 200)
    }

    processTask()
  }

  const getTaskIcon = (type: string) => {
    switch (type) {
      case "privacy":
        return <Shield className="w-4 h-4" />
      case "emoji":
        return <Smile className="w-4 h-4" />
      case "noise":
        return <Filter className="w-4 h-4" />
      default:
        return <FileText className="w-4 h-4" />
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "completed":
        return <CheckCircle2 className="w-4 h-4 text-primary" />
      case "processing":
        return <Clock className="w-4 h-4 text-accent animate-pulse" />
      default:
        return <AlertCircle className="w-4 h-4 text-muted-foreground" />
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-foreground">多源数据管理看板</h2>
          <p className="text-sm text-muted-foreground mt-1">数据管家 · 异构数据处理中心</p>
        </div>
        <Badge variant="outline" className="bg-primary/10 text-primary border-primary/30">
          Data Governance Layer
        </Badge>
      </div>

      {/* Upload Zone */}
      <Card className="bg-card border-border">
        <CardHeader className="pb-3">
          <CardTitle className="text-base font-medium flex items-center gap-2">
            <Upload className="w-4 h-4 text-primary" />
            数据上传区
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div
            onDragOver={(e) => {
              e.preventDefault()
              setIsDragging(true)
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            className={`
              border-2 border-dashed rounded-lg p-8 text-center transition-all cursor-pointer
              ${isDragging ? "border-primary bg-primary/5" : "border-border hover:border-primary/50 hover:bg-muted/30"}
            `}
          >
            <div className="flex flex-col items-center gap-3">
              <div className="flex gap-4">
                <div className="flex flex-col items-center gap-1">
                  <div className="p-3 rounded-lg bg-accent/10">
                    <FileText className="w-6 h-6 text-accent" />
                  </div>
                  <span className="text-xs text-muted-foreground">PDF</span>
                </div>
                <div className="flex flex-col items-center gap-1">
                  <div className="p-3 rounded-lg bg-primary/10">
                    <BookOpen className="w-6 h-6 text-primary" />
                  </div>
                  <span className="text-xs text-muted-foreground">书籍</span>
                </div>
                <div className="flex flex-col items-center gap-1">
                  <div className="p-3 rounded-lg bg-chart-3/10">
                    <Globe className="w-6 h-6 text-chart-3" />
                  </div>
                  <span className="text-xs text-muted-foreground">网络检索</span>
                </div>
              </div>
              <p className="text-sm text-muted-foreground mt-2">
                拖拽文件到此处，或点击上传 PDF、书籍、网络检索公共库文件
              </p>
              <Button variant="outline" size="sm" className="mt-2 bg-transparent">
                选择文件
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Uploaded Files */}
      {uploadedFiles.length > 0 && (
        <Card className="bg-card border-border">
          <CardHeader className="pb-3">
            <CardTitle className="text-base font-medium">已上传文件</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {uploadedFiles.map((file, index) => (
                <div key={index} className="flex items-center gap-3 p-3 rounded-lg bg-muted/30">
                  <FileText className="w-4 h-4 text-primary" />
                  <span className="text-sm text-foreground flex-1">{file}</span>
                  <Badge variant="secondary" className="text-xs">
                    已就绪
                  </Badge>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Processing Status */}
      {processingTasks.length > 0 && (
        <Card className="bg-card border-border">
          <CardHeader className="pb-3">
            <CardTitle className="text-base font-medium">数据预处理状态</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {processingTasks.map((task) => (
              <div key={task.id} className="space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    {getTaskIcon(task.type)}
                    <span className="text-sm text-foreground">{task.name}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-muted-foreground">{Math.round(task.progress)}%</span>
                    {getStatusIcon(task.status)}
                  </div>
                </div>
                <Progress value={task.progress} className="h-1.5" />
              </div>
            ))}
          </CardContent>
        </Card>
      )}
    </div>
  )
}
