"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Upload, FileText, CheckCircle2, X, Loader2, AlertCircle } from "lucide-react"
import { AISidebar } from "@/components/ai-sidebar"
import { useFileUpload } from "@/hooks/use-file-upload"
import { useAnalysisTask } from "@/hooks/use-analysis-task"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { useRouter } from "next/navigation"
import { configService } from "@/lib/api/services"

export default function SetupPage() {
  const router = useRouter()
  const fileInputRef = useRef<HTMLInputElement>(null)
  const {
    uploadedFiles,
    uploading,
    uploadProgress,
    error: uploadError,
    uploadFile,
    removeFile,
  } = useFileUpload()

  const {
    taskId,
    status: taskStatus,
    progress: taskProgress,
    error: taskError,
    result: taskResult,
    startAnalysis,
    reset: resetTask,
  } = useAnalysisTask()

  const [fieldMapping, setFieldMapping] = useState({
    filename: "",
    content: "",
    modified: "",
  })
  const [selectedModel, setSelectedModel] = useState("")
  const [isDragging, setIsDragging] = useState(false)
  const [parsingFile, setParsingFile] = useState<string | null>(null)
  const [availableHeaders, setAvailableHeaders] = useState<string[]>([])
  const [aiMessages, setAiMessages] = useState([
    { type: "info", text: "请上传数据文件开始配置" },
  ])

  // 当文件上传成功后，从解析数据中获取表头
  useEffect(() => {
    const lastFile = uploadedFiles[uploadedFiles.length - 1]
    if (lastFile && lastFile.parsedData && !parsingFile) {
      setParsingFile(lastFile.fileId)
      setAiMessages([
        { type: "info", text: "正在解析文件..." },
      ])

      // 使用前端解析的数据
      const parsedData = lastFile.parsedData
      setAvailableHeaders(parsedData.headers)
      setAiMessages([
        { type: "success", text: `已自动过滤空行，检测到 ${parsedData.rowCount} 条有效记录` },
        { type: "tip", text: "请选择对应的字段映射" },
      ])
      
      // 尝试自动匹配字段
      autoMatchFields(parsedData.headers)
      
      setParsingFile(null)
    }
  }, [uploadedFiles])

  // 自动匹配字段
  const autoMatchFields = (headers: string[]) => {
    const filenameKeywords = ['filename', 'file_name', 'name', 'title', 'document_title']
    const contentKeywords = ['content', 'text', 'body', 'narrative', 'complaint', 'cleaned_text']
    const modifiedKeywords = ['modified', 'modified_at', 'updated', 'updated_date', 'timestamp', 'date', 'time']

    const filenameMatch = headers.find(h => 
      filenameKeywords.some(k => h.toLowerCase().includes(k.toLowerCase()))
    )
    const contentMatch = headers.find(h => 
      contentKeywords.some(k => h.toLowerCase().includes(k.toLowerCase()))
    )
    const modifiedMatch = headers.find(h => 
      modifiedKeywords.some(k => h.toLowerCase().includes(k.toLowerCase()))
    )

    setFieldMapping(prev => ({
      filename: filenameMatch || prev.filename,
      content: contentMatch || prev.content,
      modified: modifiedMatch || prev.modified,
    }))
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const files = Array.from(e.dataTransfer.files)
    
    for (const file of files) {
      await uploadFile(file)
    }
  }

  const handleFileSelect = () => {
    fileInputRef.current?.click()
  }

  const handleFileInputChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (files) {
      for (const file of Array.from(files)) {
        await uploadFile(file)
      }
      // 重置 input，允许重复选择同一文件
      e.target.value = ""
    }
  }

  const handleStartAnalysis = async () => {
    if (uploadedFiles.length === 0) {
      setAiMessages([{ type: "info", text: "请先上传数据文件" }])
      return
    }

    if (!fieldMapping.filename || !fieldMapping.content || !fieldMapping.modified) {
      setAiMessages([{ type: "info", text: "请完成字段映射配置" }])
      return
    }

    if (!selectedModel) {
      setAiMessages([{ type: "info", text: "请选择 LoRA 专家适配器" }])
      return
    }

    setAiMessages([{ type: "info", text: "正在启动分析任务..." }])

    const newTaskId = await startAnalysis({
      fileId: uploadedFiles[0].fileId,
      fieldMapping,
      model: selectedModel,
    })

    if (newTaskId) {
      setAiMessages([
        { type: "success", text: `分析任务已启动，任务 ID: ${newTaskId}` },
        { type: "info", text: "正在处理数据，请稍候..." },
      ])
    } else {
      setAiMessages([
        { type: "info", text: taskError || "启动分析任务失败，请重试" },
      ])
    }
  }

  // 监听任务状态变化
  useEffect(() => {
    if (taskStatus === 'completed' && taskResult) {
      setAiMessages([
        { type: "success", text: "分析任务已完成！" },
        { type: "info", text: "正在跳转到分析结果页面..." },
      ])
      // 延迟跳转，让用户看到成功消息
      setTimeout(() => {
        router.push(`/analytics?taskId=${taskId}`)
      }, 1500)
    } else if (taskStatus === 'failed') {
      setAiMessages([
        { type: "info", text: taskError || "分析任务失败，请检查配置后重试" },
      ])
    } else if (taskStatus === 'processing') {
      setAiMessages([
        { type: "info", text: `分析进行中... ${Math.round(taskProgress)}%` },
      ])
    }
  }, [taskStatus, taskProgress, taskResult, taskError, taskId, router])

  const handleSaveConfig = async () => {
    try {
      // 尝试调用后端 API
      const response = await configService.saveConfig({
        fieldMapping,
        selectedModel,
        uploadedFiles: uploadedFiles.map(f => f.fileId),
      })
      
      if (response.success) {
        setAiMessages([{ type: "success", text: "配置已保存到服务器" }])
        return
      }
    } catch (error) {
      console.log('后端 API 不可用，保存到本地存储')
    }
    
    // 保存到本地存储
    const configData = {
      fieldMapping,
      selectedModel,
      uploadedFiles: uploadedFiles.map(f => ({
        fileId: f.fileId,
        name: f.name,
        size: f.size,
        type: f.type,
      })),
      savedAt: new Date().toISOString(),
    }
    
    localStorage.setItem("theta-config", JSON.stringify(configData))
    setAiMessages([{ type: "success", text: "配置已保存到本地" }])
  }

  // 加载保存的配置
  useEffect(() => {
    const loadSavedConfig = async () => {
      try {
        // 尝试从后端加载
        const response = await configService.getConfig()
        if (response.success && response.data) {
          setFieldMapping(response.data.fieldMapping)
          setSelectedModel(response.data.selectedModel)
          setAiMessages([{ type: "info", text: "已加载保存的配置" }])
          return
        }
      } catch (error) {
        console.log('后端 API 不可用，从本地存储加载')
      }

      // 从本地存储加载
      const savedConfig = localStorage.getItem("theta-config")
      if (savedConfig) {
        try {
          const configData = JSON.parse(savedConfig)
          if (configData.fieldMapping) {
            setFieldMapping(configData.fieldMapping)
          }
          if (configData.selectedModel) {
            setSelectedModel(configData.selectedModel)
          }
          setAiMessages([{ type: "info", text: "已从本地加载保存的配置" }])
        } catch (err) {
          console.error('加载配置失败:', err)
        }
      }
    }

    loadSavedConfig()
  }, [])

  const completionPercentage =
    (uploadedFiles.length > 0 ? 33 : 0) +
    (fieldMapping.filename && fieldMapping.content && fieldMapping.modified ? 33 : 0) +
    (selectedModel ? 34 : 0)

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

          {/* Error Alerts */}
          {uploadError && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{uploadError}</AlertDescription>
            </Alert>
          )}
          
          {taskError && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{taskError}</AlertDescription>
            </Alert>
          )}

          {/* Task Progress */}
          {taskStatus !== 'idle' && taskStatus !== 'completed' && (
            <Card className="border-primary/20">
              <CardHeader>
                <CardTitle className="text-sm font-medium flex items-center gap-2">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  分析任务进行中
                </CardTitle>
              </CardHeader>
              <CardContent>
                <Progress value={taskProgress} className="h-2" />
                <div className="flex justify-between items-center mt-2">
                  <p className="text-xs text-muted-foreground">
                    状态: {taskStatus === 'pending' ? '等待中' : taskStatus === 'processing' ? '处理中' : '未知'}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    {Math.round(taskProgress)}%
                  </p>
                </div>
                {taskId && (
                  <p className="text-xs text-muted-foreground mt-2">
                    任务 ID: {taskId}
                  </p>
                )}
              </CardContent>
            </Card>
          )}

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
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept=".csv,.xlsx,.xls,.json"
                onChange={handleFileInputChange}
                className="hidden"
              />
              
              <div
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                className={`border-2 border-dashed rounded-lg p-12 text-center transition-colors ${
                  isDragging ? "border-primary bg-primary/5" : "border-border"
                } ${uploading ? "opacity-50 pointer-events-none" : ""}`}
              >
                <Upload className="w-12 h-12 mx-auto text-muted-foreground mb-4" />
                <p className="text-sm text-muted-foreground mb-2">拖拽文件至此或点击上传</p>
                <Button variant="outline" size="sm" onClick={handleFileSelect} disabled={uploading}>
                  {uploading ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      上传中...
                    </>
                  ) : (
                    "选择文件"
                  )}
                </Button>
              </div>

              {/* Uploaded Files List */}
              {uploadedFiles.length > 0 && (
                <div className="mt-4 space-y-2">
                  <p className="text-sm font-medium">已上传文件：</p>
                  {uploadedFiles.map((file) => (
                    <div key={file.id} className="flex items-center gap-2 p-2 bg-muted/30 rounded">
                      <FileText className="w-4 h-4 text-primary" />
                      <span className="text-sm flex-1">{file.name}</span>
                      {uploadProgress[file.name] !== undefined && (
                        <div className="flex items-center gap-2 w-24">
                          <Progress value={uploadProgress[file.name]} className="h-1 flex-1" />
                          <span className="text-xs text-muted-foreground">
                            {Math.round(uploadProgress[file.name])}%
                          </span>
                        </div>
                      )}
                      {uploadProgress[file.name] === undefined && (
                        <CheckCircle2 className="w-4 h-4 text-green-500" />
                      )}
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-6 w-6 p-0"
                        onClick={() => removeFile(file.id)}
                      >
                        <X className="w-4 h-4" />
                      </Button>
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
                  disabled={availableHeaders.length === 0}
                >
                  <SelectTrigger>
                    <SelectValue placeholder={availableHeaders.length === 0 ? "请先上传文件..." : "选择字段..."} />
                  </SelectTrigger>
                  <SelectContent>
                    {availableHeaders.length > 0 ? (
                      availableHeaders.map((header) => (
                        <SelectItem key={header} value={header}>
                          {header}
                        </SelectItem>
                      ))
                    ) : (
                      <>
                        <SelectItem value="file_name">file_name</SelectItem>
                        <SelectItem value="document_title">document_title</SelectItem>
                        <SelectItem value="name">name</SelectItem>
                      </>
                    )}
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label className="text-sm font-medium mb-2 block">文本内容字段</label>
                <Select
                  value={fieldMapping.content}
                  onValueChange={(v) => setFieldMapping((prev) => ({ ...prev, content: v }))}
                  disabled={availableHeaders.length === 0}
                >
                  <SelectTrigger>
                    <SelectValue placeholder={availableHeaders.length === 0 ? "请先上传文件..." : "选择字段..."} />
                  </SelectTrigger>
                  <SelectContent>
                    {availableHeaders.length > 0 ? (
                      availableHeaders.map((header) => (
                        <SelectItem key={header} value={header}>
                          {header}
                        </SelectItem>
                      ))
                    ) : (
                      <>
                        <SelectItem value="content">content</SelectItem>
                        <SelectItem value="text">text</SelectItem>
                        <SelectItem value="body">body</SelectItem>
                      </>
                    )}
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label className="text-sm font-medium mb-2 block">修改时间字段</label>
                <Select
                  value={fieldMapping.modified}
                  onValueChange={(v) => setFieldMapping((prev) => ({ ...prev, modified: v }))}
                  disabled={availableHeaders.length === 0}
                >
                  <SelectTrigger>
                    <SelectValue placeholder={availableHeaders.length === 0 ? "请先上传文件..." : "选择字段..."} />
                  </SelectTrigger>
                  <SelectContent>
                    {availableHeaders.length > 0 ? (
                      availableHeaders.map((header) => (
                        <SelectItem key={header} value={header}>
                          {header}
                        </SelectItem>
                      ))
                    ) : (
                      <>
                        <SelectItem value="modified_at">modified_at</SelectItem>
                        <SelectItem value="updated_date">updated_date</SelectItem>
                        <SelectItem value="timestamp">timestamp</SelectItem>
                      </>
                    )}
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
            <Button 
              size="lg" 
              disabled={completionPercentage < 100 || taskStatus === 'pending' || taskStatus === 'processing'} 
              onClick={handleStartAnalysis}
            >
              {taskStatus === 'pending' || taskStatus === 'processing' ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  分析中...
                </>
              ) : (
                "开始分析"
              )}
            </Button>
            <Button 
              size="lg" 
              variant="outline" 
              onClick={handleSaveConfig}
              disabled={taskStatus === 'pending' || taskStatus === 'processing'}
            >
              保存配置
            </Button>
            {taskStatus !== 'idle' && (
              <Button 
                size="lg" 
                variant="outline" 
                onClick={() => {
                  resetTask()
                  setAiMessages([{ type: "info", text: "任务已取消" }])
                }}
              >
                取消任务
              </Button>
            )}
          </div>
        </div>
      </div>

      {/* Right AI Sidebar - 25% */}
      <AISidebar title="数据管家 Agent" subtitle="Data Steward" messages={aiMessages} />
    </div>
  )
}
