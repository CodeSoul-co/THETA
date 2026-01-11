"use client"

import type React from "react"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Upload, FileText, CheckCircle2 } from "lucide-react"
import { AISidebar } from "@/components/ai-sidebar"

export default function SetupPage() {
  const [uploadedFiles, setUploadedFiles] = useState<string[]>([])
  const [fieldMapping, setFieldMapping] = useState({
    filename: "",
    content: "",
    modified: "",
  })
  const [selectedModel, setSelectedModel] = useState("")
  const [isDragging, setIsDragging] = useState(false)

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const files = Array.from(e.dataTransfer.files).map((f) => f.name)
    setUploadedFiles((prev) => [...prev, ...files])
  }

  const completionPercentage =
    (uploadedFiles.length > 0 ? 33 : 0) +
    (fieldMapping.filename && fieldMapping.content && fieldMapping.modified ? 33 : 0) +
    (selectedModel ? 34 : 0)

  const aiMessages = [
    { type: "info", text: "正在解析文件编码 (UTF-8/GBK)..." },
    { type: "success", text: "已自动过滤空行，检测到 1,234 条有效记录" },
    { type: "tip", text: "建议：检测到时间字段格式不统一，是否需要标准化处理？" },
  ]

  return (
    <div className="flex h-screen bg-background">
      {/* Left Main Workspace - 75% */}
      <div className="flex-1 overflow-y-auto p-8">
        <div className="max-w-5xl mx-auto space-y-6">
          {/* Header */}
          <div>
            <h1 className="text-4xl font-bold tracking-tight text-foreground">数据治理与任务配置</h1>
            <p className="text-muted-foreground mt-2">Setup & Governance</p>
          </div>

          {/* Progress Indicator */}
          <Card className="border-accent/20">
            <CardHeader>
              <CardTitle className="text-sm font-medium">配置完成度</CardTitle>
            </CardHeader>
            <CardContent>
              <Progress value={completionPercentage} className="h-2" />
              <p className="text-xs text-muted-foreground mt-2">{completionPercentage}% 完成</p>
            </CardContent>
          </Card>

          {/* Data Upload Section */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Upload className="w-5 h-5" />
                数据上传区
              </CardTitle>
              <CardDescription>支持 CSV、Excel、JSON 格式文件</CardDescription>
            </CardHeader>
            <CardContent>
              <div
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                className={`border-2 border-dashed rounded-lg p-12 text-center transition-colors ${
                  isDragging ? "border-primary bg-primary/5" : "border-border"
                }`}
              >
                <Upload className="w-12 h-12 mx-auto text-muted-foreground mb-4" />
                <p className="text-sm text-muted-foreground mb-2">拖拽文件至此或点击上传</p>
                <Button variant="outline" size="sm">
                  选择文件
                </Button>
              </div>

              {/* Uploaded Files List */}
              {uploadedFiles.length > 0 && (
                <div className="mt-4 space-y-2">
                  <p className="text-sm font-medium">已上传文件：</p>
                  {uploadedFiles.map((file, idx) => (
                    <div key={idx} className="flex items-center gap-2 p-2 bg-muted/30 rounded">
                      <FileText className="w-4 h-4 text-primary" />
                      <span className="text-sm flex-1">{file}</span>
                      <CheckCircle2 className="w-4 h-4 text-green-500" />
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Field Mapping */}
          <Card>
            <CardHeader>
              <CardTitle>字段映射 (Field Mapping)</CardTitle>
              <CardDescription>从 CSV 标头中选择对应的标准字段</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="text-sm font-medium mb-2 block">文件名字段</label>
                <Select
                  value={fieldMapping.filename}
                  onValueChange={(v) => setFieldMapping((prev) => ({ ...prev, filename: v }))}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="选择字段..." />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="file_name">file_name</SelectItem>
                    <SelectItem value="document_title">document_title</SelectItem>
                    <SelectItem value="name">name</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label className="text-sm font-medium mb-2 block">文本内容字段</label>
                <Select
                  value={fieldMapping.content}
                  onValueChange={(v) => setFieldMapping((prev) => ({ ...prev, content: v }))}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="选择字段..." />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="content">content</SelectItem>
                    <SelectItem value="text">text</SelectItem>
                    <SelectItem value="body">body</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label className="text-sm font-medium mb-2 block">修改时间字段</label>
                <Select
                  value={fieldMapping.modified}
                  onValueChange={(v) => setFieldMapping((prev) => ({ ...prev, modified: v }))}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="选择字段..." />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="modified_at">modified_at</SelectItem>
                    <SelectItem value="updated_date">updated_date</SelectItem>
                    <SelectItem value="timestamp">timestamp</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </CardContent>
          </Card>

          {/* Model Selection */}
          <Card>
            <CardHeader>
              <CardTitle>LoRA 专家适配器选择</CardTitle>
              <CardDescription>根据研究领域选择预训练的专家模型</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-3 gap-3">
                {["金融", "心理", "医疗", "公共卫生", "政治", "Twitter"].map((model) => (
                  <Button
                    key={model}
                    variant={selectedModel === model ? "default" : "outline"}
                    onClick={() => setSelectedModel(model)}
                    className="h-16"
                  >
                    {model}
                  </Button>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Action Buttons */}
          <div className="flex gap-3">
            <Button size="lg" disabled={completionPercentage < 100}>
              开始分析
            </Button>
            <Button size="lg" variant="outline">
              保存配置
            </Button>
          </div>
        </div>
      </div>

      {/* Right AI Sidebar - 25% */}
      <AISidebar title="数据管家 Agent" subtitle="Data Steward" messages={aiMessages} />
    </div>
  )
}
