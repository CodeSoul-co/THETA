"use client"

import { useState, useCallback } from "react"
import { motion, AnimatePresence } from "framer-motion"
import {
  Upload,
  FileCog,
  CheckCircle2,
  XCircle,
  Loader2,
  Download,
  Trash2,
  FileText,
  AlertCircle,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog"
import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Checkbox } from "@/components/ui/checkbox"
import { Progress } from "@/components/ui/progress"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { DataCleanAPI, type TaskStatusResponse } from "@/lib/api/dataclean"

interface ProcessingTask {
  id: string
  taskId?: string
  fileName: string
  fileSize: number
  status: 'pending' | 'processing' | 'completed' | 'failed'
  progress: number
  message: string
  error?: string
  createdAt: Date
}

export function DataProcessingView() {
  const [tasks, setTasks] = useState<ProcessingTask[]>([])
  const [showUploadDialog, setShowUploadDialog] = useState(false)
  const [isDragging, setIsDragging] = useState(false)
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])
  const [language, setLanguage] = useState<'chinese' | 'english'>('chinese')
  const [cleanEnabled, setCleanEnabled] = useState(true)
  const [operations, setOperations] = useState<string[]>(['remove_urls', 'remove_html_tags', 'normalize_whitespace'])
  const [supportedFormats, setSupportedFormats] = useState<string[]>([])
  const [loadingFormats, setLoadingFormats] = useState(false)

  // 加载支持的文件格式
  const loadSupportedFormats = useCallback(async () => {
    setLoadingFormats(true)
    try {
      const response = await DataCleanAPI.getSupportedFormats()
      setSupportedFormats(response.formats)
    } catch (error) {
      console.error('加载支持格式失败:', error)
    } finally {
      setLoadingFormats(false)
    }
  }, [])

  // 检查文件格式是否支持
  const isFileSupported = (file: File): boolean => {
    const ext = '.' + file.name.split('.').pop()?.toLowerCase()
    return supportedFormats.includes(ext)
  }

  // 处理文件上传
  const handleFileSelect = (files: FileList | null) => {
    if (!files || files.length === 0) return

    const fileArray = Array.from(files)
    const validFiles = fileArray.filter(isFileSupported)
    const invalidFiles = fileArray.filter(f => !isFileSupported(f))

    if (invalidFiles.length > 0) {
      alert(`以下文件格式不支持: ${invalidFiles.map(f => f.name).join(', ')}`)
    }

    if (validFiles.length > 0) {
      setSelectedFiles(prev => [...prev, ...validFiles])
    }
  }

  // 处理拖拽
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    handleFileSelect(e.dataTransfer.files)
  }

  // 开始处理任务
  const handleStartProcessing = async () => {
    if (selectedFiles.length === 0) return

    // 为每个文件创建任务
    const newTasks: ProcessingTask[] = selectedFiles.map(file => ({
      id: Date.now().toString() + Math.random().toString(36).substr(2, 9),
      fileName: file.name,
      fileSize: file.size,
      status: 'pending',
      progress: 0,
      message: '准备上传...',
      createdAt: new Date(),
    }))

    setTasks(prev => [...prev, ...newTasks])
    setShowUploadDialog(false)
    setSelectedFiles([])

    // 处理每个文件
    newTasks.forEach(async (task, index) => {
      try {
        const file = selectedFiles[index]
        setTasks(prev => prev.map(t =>
          t.id === task.id ? { ...t, status: 'processing', message: '正在上传文件...', progress: 10 } : t
        ))

        // 上传并处理文件
        const response = await DataCleanAPI.processFile(
          file,
          language,
          cleanEnabled,
          cleanEnabled ? operations : undefined
        )

        setTasks(prev => prev.map(t =>
          t.id === task.id
            ? { ...t, taskId: response.task_id, status: 'completed', message: '处理完成', progress: 100 }
            : t
        ))
      } catch (error) {
        setTasks(prev => prev.map(t =>
          t.id === task.id
            ? {
                ...t,
                status: 'failed',
                message: '处理失败',
                error: error instanceof Error ? error.message : '未知错误',
                progress: 0,
              }
            : t
        ))
      }
    })
  }

  // 下载结果
  const handleDownload = async (task: ProcessingTask) => {
    if (!task.taskId) return

    try {
      await DataCleanAPI.downloadResultFile(task.taskId, `cleaned_${task.fileName}.csv`)
    } catch (error) {
      alert('下载失败: ' + (error instanceof Error ? error.message : '未知错误'))
    }
  }

  // 删除任务
  const handleDeleteTask = async (taskId: string) => {
    setTasks(prev => prev.filter(t => t.id !== taskId))
    // 如果有后端任务ID，可以调用删除API
    // await DataCleanAPI.deleteTask(task.taskId)
  }

  // 可用的清洗操作
  const availableOperations = [
    { id: 'remove_urls', label: '移除URL' },
    { id: 'remove_html_tags', label: '移除HTML标签' },
    { id: 'remove_punctuation', label: '移除标点符号' },
    { id: 'remove_stopwords', label: '移除停用词' },
    { id: 'normalize_whitespace', label: '规范化空白字符' },
    { id: 'remove_numbers', label: '移除数字' },
    { id: 'remove_special_chars', label: '移除特殊字符' },
  ]

  return (
    <div className="p-8">
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-semibold text-slate-900 mb-2">数据处理</h2>
          <p className="text-slate-600">上传文件进行文本清洗和格式转换</p>
        </div>
        <Button
          onClick={() => {
            setShowUploadDialog(true)
            loadSupportedFormats()
          }}
          className="bg-blue-600 hover:bg-blue-700 text-white gap-2"
        >
          <Upload className="w-4 h-4" />
          上传文件
        </Button>
      </div>

      {/* 任务列表 */}
      {tasks.length > 0 ? (
        <div className="space-y-4">
          {tasks.map((task) => (
            <motion.div
              key={task.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
            >
              <Card className="border border-slate-200 bg-white p-6 rounded-xl">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <FileText className="w-5 h-5 text-slate-500" />
                      <div>
                        <h3 className="font-semibold text-slate-900">{task.fileName}</h3>
                        <p className="text-sm text-slate-500">
                          {(task.fileSize / 1024).toFixed(2)} KB
                        </p>
                      </div>
                    </div>

                    <div className="mt-4 space-y-2">
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-slate-600">{task.message}</span>
                        <span className="text-slate-500">
                          {task.status === 'processing' && `${task.progress}%`}
                        </span>
                      </div>
                      {task.status === 'processing' && (
                        <Progress value={task.progress} className="h-2" />
                      )}
                      {task.error && (
                        <Alert variant="destructive">
                          <AlertCircle className="h-4 w-4" />
                          <AlertDescription>{task.error}</AlertDescription>
                        </Alert>
                      )}
                    </div>
                  </div>

                  <div className="flex items-center gap-2 ml-4">
                    {task.status === 'completed' && (
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleDownload(task)}
                        className="gap-2"
                      >
                        <Download className="w-4 h-4" />
                        下载
                      </Button>
                    )}
                    {task.status === 'processing' && (
                      <Loader2 className="w-5 h-5 text-blue-600 animate-spin" />
                    )}
                    {task.status === 'completed' && (
                      <CheckCircle2 className="w-5 h-5 text-green-600" />
                    )}
                    {task.status === 'failed' && (
                      <XCircle className="w-5 h-5 text-red-600" />
                    )}
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleDeleteTask(task.id)}
                      className="text-red-600 hover:text-red-700"
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
              </Card>
            </motion.div>
          ))}
        </div>
      ) : (
        <Card className="border border-slate-200 bg-white p-8 rounded-xl text-center">
          <div className="flex flex-col items-center gap-4 py-8">
            <div className="w-16 h-16 rounded-2xl bg-slate-100 flex items-center justify-center">
              <FileCog className="w-8 h-8 text-slate-400" />
            </div>
            <div>
              <p className="text-slate-500 mb-2">暂无处理任务</p>
              <p className="text-sm text-slate-400">点击上方按钮上传文件开始处理</p>
            </div>
          </div>
        </Card>
      )}

      {/* 上传对话框 */}
      <Dialog open={showUploadDialog} onOpenChange={setShowUploadDialog}>
        <DialogContent className="sm:max-w-2xl bg-white">
          <DialogHeader>
            <DialogTitle className="text-slate-900">上传文件进行处理</DialogTitle>
          </DialogHeader>

          <div className="space-y-6 py-4">
            {/* 文件上传区域 */}
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              className={`border-2 border-dashed rounded-xl p-8 text-center transition-colors ${
                isDragging
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-slate-300 hover:border-slate-400'
              }`}
            >
              <Upload className="w-12 h-12 text-slate-400 mx-auto mb-4" />
              <p className="text-slate-600 mb-2">
                拖拽文件到此处，或
                <label className="text-blue-600 cursor-pointer hover:underline ml-1">
                  点击选择文件
                  <input
                    type="file"
                    multiple
                    className="hidden"
                    onChange={(e) => handleFileSelect(e.target.files)}
                    accept={supportedFormats.join(',')}
                  />
                </label>
              </p>
              {loadingFormats ? (
                <p className="text-sm text-slate-400">加载支持格式...</p>
              ) : (
                <p className="text-sm text-slate-400">
                  支持格式: {supportedFormats.join(', ')}
                </p>
              )}
            </div>

            {/* 已选择的文件 */}
            {selectedFiles.length > 0 && (
              <div className="space-y-2">
                <Label>已选择文件 ({selectedFiles.length})</Label>
                <div className="space-y-2 max-h-40 overflow-y-auto">
                  {selectedFiles.map((file, index) => (
                    <div
                      key={index}
                      className="flex items-center justify-between p-2 bg-slate-50 rounded-lg"
                    >
                      <span className="text-sm text-slate-700">{file.name}</span>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() =>
                          setSelectedFiles(prev => prev.filter((_, i) => i !== index))
                        }
                      >
                        <XCircle className="w-4 h-4" />
                      </Button>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* 语言选择 */}
            <div className="space-y-2">
              <Label>处理语言</Label>
              <RadioGroup value={language} onValueChange={(v) => setLanguage(v as 'chinese' | 'english')}>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="chinese" id="chinese" />
                  <Label htmlFor="chinese" className="cursor-pointer">中文</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="english" id="english" />
                  <Label htmlFor="english" className="cursor-pointer">English</Label>
                </div>
              </RadioGroup>
            </div>

            {/* 清洗选项 */}
            <div className="space-y-2">
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="clean"
                  checked={cleanEnabled}
                  onCheckedChange={(checked) => setCleanEnabled(checked as boolean)}
                />
                <Label htmlFor="clean" className="cursor-pointer">启用文本清洗</Label>
              </div>

              {cleanEnabled && (
                <div className="ml-6 space-y-2">
                  <Label className="text-sm">清洗操作</Label>
                  {availableOperations.map((op) => (
                    <div key={op.id} className="flex items-center space-x-2">
                      <Checkbox
                        id={op.id}
                        checked={operations.includes(op.id)}
                        onCheckedChange={(checked) => {
                          if (checked) {
                            setOperations(prev => [...prev, op.id])
                          } else {
                            setOperations(prev => prev.filter(o => o !== op.id))
                          }
                        }}
                      />
                      <Label htmlFor={op.id} className="cursor-pointer text-sm">
                        {op.label}
                      </Label>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setShowUploadDialog(false)
                setSelectedFiles([])
              }}
            >
              取消
            </Button>
            <Button
              onClick={handleStartProcessing}
              disabled={selectedFiles.length === 0}
              className="bg-blue-600 hover:bg-blue-700 text-white"
            >
              开始处理
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
