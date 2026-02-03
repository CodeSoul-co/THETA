"use client"

import { useState, useEffect, useCallback } from "react"
import {
  Database,
  FileText,
  BarChart3,
  Upload,
  Play,
  Check,
  Loader2,
  AlertCircle,
  ChevronRight,
  RefreshCw,
  Eye,
  Download,
  Trash2,
  Settings2,
  Sparkles,
  FileCheck,
  PieChart,
  FileCog,
} from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { ScrollArea } from "@/components/ui/scroll-area"
import { cn } from "@/lib/utils"
import { ETMAgentAPI, TaskResponse, PreprocessingJob } from "@/lib/api/etm-agent"

// ==================== 类型定义 ====================

interface ProjectWorkspaceProps {
  projectId: string
  projectName: string
  rowCount: number
  datasetName?: string
}

interface WorkflowStep {
  id: string
  name: string
  icon: React.ElementType
  description: string
  status: "pending" | "in_progress" | "completed" | "error"
}

// ==================== 工作流步骤配置 ====================

const defaultSteps: WorkflowStep[] = [
  { id: "data", name: "数据管理", icon: Database, description: "查看和管理数据集", status: "pending" },
  { id: "cleaning", name: "数据清洗", icon: FileCog, description: "清洗和预处理文本", status: "pending" },
  { id: "embedding", name: "参数选择", icon: Sparkles, description: "选择模型与参数", status: "pending" },
  { id: "training", name: "模型训练", icon: Play, description: "训练主题模型", status: "pending" },
  { id: "evaluation", name: "评估结果", icon: FileCheck, description: "查看评估指标", status: "pending" },
  { id: "visualization", name: "可视化", icon: PieChart, description: "图表和报告", status: "pending" },
]

// ==================== 组件 ====================

export function ProjectWorkspace({ projectId, projectName, rowCount, datasetName }: ProjectWorkspaceProps) {
  const [activeTab, setActiveTab] = useState("overview")
  const [steps, setSteps] = useState<WorkflowStep[]>(defaultSteps)
  const [preprocessingStatus, setPreprocessingStatus] = useState<{
    has_bow: boolean
    has_embeddings: boolean
    ready_for_training: boolean
  } | null>(null)
  const [currentTask, setCurrentTask] = useState<TaskResponse | null>(null)
  const [preprocessingJob, setPreprocessingJob] = useState<PreprocessingJob | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")

  // 获取数据集名称
  const dataset = datasetName || projectName.replace(/\s+/g, "_").toLowerCase()

  // 检查预处理状态
  const checkPreprocessingStatus = useCallback(async () => {
    try {
      const status = await ETMAgentAPI.checkPreprocessingStatus(dataset)
      setPreprocessingStatus(status)
      
      // 更新步骤状态
      setSteps(prev => prev.map(step => {
        if (step.id === "embedding") {
          return {
            ...step,
            status: status.ready_for_training ? "completed" : "pending"
          }
        }
        return step
      }))
    } catch (err) {
      console.error("Failed to check preprocessing status:", err)
    }
  }, [dataset])

  // 获取任务状态
  const fetchTaskStatus = useCallback(async () => {
    try {
      const tasks = await ETMAgentAPI.getTasks({ dataset, limit: 1 })
      if (tasks.length > 0) {
        setCurrentTask(tasks[0])
        
        // 更新训练步骤状态
        setSteps(prev => prev.map(step => {
          if (step.id === "training") {
            const taskStatus = tasks[0].status
            if (taskStatus === "completed") return { ...step, status: "completed" }
            if (taskStatus === "running") return { ...step, status: "in_progress" }
            if (taskStatus === "failed") return { ...step, status: "error" }
          }
          if (step.id === "evaluation" && tasks[0].status === "completed" && tasks[0].metrics) {
            return { ...step, status: "completed" }
          }
          if (step.id === "visualization" && tasks[0].status === "completed" && tasks[0].visualization_paths) {
            return { ...step, status: "completed" }
          }
          return step
        }))
      }
    } catch (err) {
      console.error("Failed to fetch task status:", err)
    }
  }, [dataset])

  useEffect(() => {
    checkPreprocessingStatus()
    fetchTaskStatus()
  }, [checkPreprocessingStatus, fetchTaskStatus])

  // 定时刷新状态
  useEffect(() => {
    const interval = setInterval(() => {
      if (currentTask?.status === "running" || currentTask?.status === "pending") {
        fetchTaskStatus()
      }
      if (preprocessingJob?.status === "running" || preprocessingJob?.status === "pending") {
        checkPreprocessingStatus()
      }
    }, 3000)

    return () => clearInterval(interval)
  }, [currentTask, preprocessingJob, fetchTaskStatus, checkPreprocessingStatus])

  // 开始预处理
  const startPreprocessing = async () => {
    setLoading(true)
    setError("")
    try {
      const job = await ETMAgentAPI.startPreprocessing({ dataset })
      setPreprocessingJob(job)
      setSteps(prev => prev.map(step => 
        step.id === "embedding" ? { ...step, status: "in_progress" } : step
      ))
    } catch (err) {
      setError(err instanceof Error ? err.message : "预处理启动失败")
    } finally {
      setLoading(false)
    }
  }

  // 开始训练
  const startTraining = async () => {
    setLoading(true)
    setError("")
    try {
      const task = await ETMAgentAPI.createTask({
        dataset,
        mode: "zero_shot",
        num_topics: 20,
      })
      setCurrentTask(task)
      setSteps(prev => prev.map(step => 
        step.id === "training" ? { ...step, status: "in_progress" } : step
      ))
    } catch (err) {
      setError(err instanceof Error ? err.message : "训练启动失败")
    } finally {
      setLoading(false)
    }
  }

  // ==================== 渲染工作流步骤 ====================

  const renderWorkflowSteps = () => (
    <div className="flex items-center justify-between mb-6 overflow-x-auto pb-2">
      {steps.map((step, idx) => (
        <div key={step.id} className="flex items-center min-w-0">
          <div
            onClick={() => setActiveTab(step.id)}
            className={cn(
              "flex items-center gap-2 px-4 py-2 rounded-xl cursor-pointer transition-all whitespace-nowrap",
              activeTab === step.id
                ? "bg-blue-100 text-blue-700 shadow-sm"
                : "hover:bg-slate-100",
              step.status === "completed" && "text-green-700",
              step.status === "error" && "text-red-600"
            )}
          >
            <div className={cn(
              "w-8 h-8 rounded-lg flex items-center justify-center",
              step.status === "completed" ? "bg-green-100" :
              step.status === "in_progress" ? "bg-blue-100" :
              step.status === "error" ? "bg-red-100" :
              "bg-slate-100"
            )}>
              {step.status === "completed" ? (
                <Check className="w-4 h-4 text-green-600" />
              ) : step.status === "in_progress" ? (
                <Loader2 className="w-4 h-4 text-blue-600 animate-spin" />
              ) : step.status === "error" ? (
                <AlertCircle className="w-4 h-4 text-red-600" />
              ) : (
                <step.icon className="w-4 h-4 text-slate-500" />
              )}
            </div>
            <span className="text-sm font-medium hidden lg:block">{step.name}</span>
          </div>
          {idx < steps.length - 1 && (
            <ChevronRight className="w-4 h-4 text-slate-300 mx-2 flex-shrink-0" />
          )}
        </div>
      ))}
    </div>
  )

  // ==================== 渲染概览页 ====================

  const renderOverview = () => (
    <div className="space-y-6">
      {/* 项目概览卡片 */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="bg-gradient-to-br from-blue-50 to-indigo-50 border-blue-100">
          <CardContent className="pt-6">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 rounded-xl bg-blue-100 flex items-center justify-center">
                <Database className="w-6 h-6 text-blue-600" />
              </div>
              <div>
                <p className="text-2xl font-bold text-slate-900">{rowCount.toLocaleString()}</p>
                <p className="text-sm text-slate-500">数据条数</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-green-50 to-emerald-50 border-green-100">
          <CardContent className="pt-6">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 rounded-xl bg-green-100 flex items-center justify-center">
                <Sparkles className="w-6 h-6 text-green-600" />
              </div>
              <div>
                <p className="text-2xl font-bold text-slate-900">
                  {preprocessingStatus?.ready_for_training ? "就绪" : "待处理"}
                </p>
                <p className="text-sm text-slate-500">参数选择状态</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-purple-50 to-violet-50 border-purple-100">
          <CardContent className="pt-6">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 rounded-xl bg-purple-100 flex items-center justify-center">
                <Play className="w-6 h-6 text-purple-600" />
              </div>
              <div>
                <p className="text-2xl font-bold text-slate-900">
                  {currentTask?.status === "completed" ? "已完成" :
                   currentTask?.status === "running" ? "训练中" : "未开始"}
                </p>
                <p className="text-sm text-slate-500">训练状态</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* 快速操作 */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">快速操作</CardTitle>
          <CardDescription>根据当前进度执行下一步</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {!preprocessingStatus?.ready_for_training && (
            <div className="flex items-center justify-between p-4 bg-amber-50 rounded-lg border border-amber-100">
              <div className="flex items-center gap-3">
                <Sparkles className="w-5 h-5 text-amber-600" />
                <div>
                  <p className="font-medium text-slate-900">参数选择</p>
                  <p className="text-sm text-slate-500">需要先选择模型并生成 BOW 和 Embeddings</p>
                </div>
              </div>
              <Button 
                onClick={startPreprocessing} 
                disabled={loading}
                className="bg-amber-600 hover:bg-amber-700"
              >
                {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : "开始预处理"}
              </Button>
            </div>
          )}

          {preprocessingStatus?.ready_for_training && !currentTask && (
            <div className="flex items-center justify-between p-4 bg-blue-50 rounded-lg border border-blue-100">
              <div className="flex items-center gap-3">
                <Play className="w-5 h-5 text-blue-600" />
                <div>
                  <p className="font-medium text-slate-900">开始模型训练</p>
                  <p className="text-sm text-slate-500">数据已就绪，可以开始训练</p>
                </div>
              </div>
              <Button 
                onClick={startTraining} 
                disabled={loading}
                className="bg-blue-600 hover:bg-blue-700"
              >
                {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : "开始训练"}
              </Button>
            </div>
          )}

          {currentTask?.status === "running" && (
            <div className="p-4 bg-blue-50 rounded-lg border border-blue-100">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-3">
                  <Loader2 className="w-5 h-5 text-blue-600 animate-spin" />
                  <div>
                    <p className="font-medium text-slate-900">训练进行中</p>
                    <p className="text-sm text-slate-500">{currentTask.message || currentTask.current_step}</p>
                  </div>
                </div>
                <Badge variant="secondary">{currentTask.progress}%</Badge>
              </div>
              <Progress value={currentTask.progress} className="h-2" />
            </div>
          )}

          {currentTask?.status === "completed" && (
            <div className="flex items-center justify-between p-4 bg-green-50 rounded-lg border border-green-100">
              <div className="flex items-center gap-3">
                <Check className="w-5 h-5 text-green-600" />
                <div>
                  <p className="font-medium text-slate-900">训练完成</p>
                  <p className="text-sm text-slate-500">可查看评估结果和可视化</p>
                </div>
              </div>
              <Button 
                variant="outline"
                onClick={() => setActiveTab("evaluation")}
              >
                查看结果
              </Button>
            </div>
          )}
        </CardContent>
      </Card>

      {/* 错误提示 */}
      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>错误</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}
    </div>
  )

  // ==================== 渲染数据管理 ====================

  const renderDataManagement = () => (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>数据管理</CardTitle>
            <CardDescription>查看和管理项目数据集</CardDescription>
          </div>
          <Button variant="outline" size="sm">
            <Upload className="w-4 h-4 mr-2" />
            上传更多数据
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="p-4 bg-slate-50 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <span className="font-medium text-slate-700">数据集名称</span>
              <span className="text-slate-900">{dataset}</span>
            </div>
            <div className="flex items-center justify-between mb-2">
              <span className="font-medium text-slate-700">数据条数</span>
              <span className="text-slate-900">{rowCount.toLocaleString()}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="font-medium text-slate-700">状态</span>
              <Badge variant="secondary">已上传</Badge>
            </div>
          </div>

          <div className="flex gap-2">
            <Button variant="outline" size="sm">
              <Eye className="w-4 h-4 mr-2" />
              预览数据
            </Button>
            <Button variant="outline" size="sm">
              <Download className="w-4 h-4 mr-2" />
              下载
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  )

  // ==================== 渲染参数选择 ====================

  const renderEmbedding = () => (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>参数选择</CardTitle>
            <CardDescription>选择模型与参数，生成 BOW 矩阵和文档嵌入向量</CardDescription>
          </div>
          <Button 
            onClick={checkPreprocessingStatus}
            variant="ghost"
            size="sm"
          >
            <RefreshCw className="w-4 h-4" />
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {preprocessingStatus && (
          <div className="grid grid-cols-2 gap-4">
            <div className={cn(
              "p-4 rounded-lg border-2",
              preprocessingStatus.has_bow ? "bg-green-50 border-green-200" : "bg-slate-50 border-slate-200"
            )}>
              <div className="flex items-center gap-3 mb-2">
                {preprocessingStatus.has_bow ? (
                  <Check className="w-5 h-5 text-green-600" />
                ) : (
                  <AlertCircle className="w-5 h-5 text-slate-400" />
                )}
                <span className="font-medium">BOW 矩阵</span>
              </div>
              <p className="text-sm text-slate-500">
                {preprocessingStatus.has_bow ? "已生成" : "待生成"}
              </p>
            </div>

            <div className={cn(
              "p-4 rounded-lg border-2",
              preprocessingStatus.has_embeddings ? "bg-green-50 border-green-200" : "bg-slate-50 border-slate-200"
            )}>
              <div className="flex items-center gap-3 mb-2">
                {preprocessingStatus.has_embeddings ? (
                  <Check className="w-5 h-5 text-green-600" />
                ) : (
                  <AlertCircle className="w-5 h-5 text-slate-400" />
                )}
                <span className="font-medium">Embeddings</span>
              </div>
              <p className="text-sm text-slate-500">
                {preprocessingStatus.has_embeddings ? "已生成" : "待生成"}
              </p>
            </div>
          </div>
        )}

        {preprocessingJob && preprocessingJob.status === "running" && (
          <div className="p-4 bg-blue-50 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <span className="font-medium text-slate-700">{preprocessingJob.message}</span>
              <span className="text-slate-500">{preprocessingJob.progress}%</span>
            </div>
            <Progress value={preprocessingJob.progress} className="h-2" />
          </div>
        )}

        {!preprocessingStatus?.ready_for_training && (
          <Button 
            onClick={startPreprocessing}
            disabled={loading}
            className="w-full"
          >
            {loading ? (
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
            ) : (
              <Sparkles className="w-4 h-4 mr-2" />
            )}
            开始参数选择
          </Button>
        )}
      </CardContent>
    </Card>
  )

  // ==================== 渲染训练 ====================

  const renderTraining = () => (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>模型训练</CardTitle>
            <CardDescription>训练主题模型并生成结果</CardDescription>
          </div>
          {currentTask && (
            <Badge variant={
              currentTask.status === "completed" ? "default" :
              currentTask.status === "running" ? "secondary" :
              currentTask.status === "failed" ? "destructive" : "outline"
            }>
              {currentTask.status === "completed" ? "已完成" :
               currentTask.status === "running" ? "训练中" :
               currentTask.status === "failed" ? "失败" : "等待中"}
            </Badge>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {currentTask ? (
          <div className="space-y-4">
            <div className="grid grid-cols-3 gap-4">
              <div className="p-3 bg-slate-50 rounded-lg">
                <p className="text-xs text-slate-500 mb-1">数据集</p>
                <p className="font-medium text-slate-900">{currentTask.dataset}</p>
              </div>
              <div className="p-3 bg-slate-50 rounded-lg">
                <p className="text-xs text-slate-500 mb-1">模式</p>
                <p className="font-medium text-slate-900">{currentTask.mode || "zero_shot"}</p>
              </div>
              <div className="p-3 bg-slate-50 rounded-lg">
                <p className="text-xs text-slate-500 mb-1">主题数</p>
                <p className="font-medium text-slate-900">{currentTask.num_topics || 20}</p>
              </div>
            </div>

            {currentTask.status === "running" && (
              <div className="p-4 bg-blue-50 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <Loader2 className="w-4 h-4 text-blue-600 animate-spin" />
                    <span className="font-medium text-slate-700">
                      {currentTask.current_step || "处理中"}
                    </span>
                  </div>
                  <span className="text-slate-500">{currentTask.progress}%</span>
                </div>
                <Progress value={currentTask.progress} className="h-2" />
                {currentTask.message && (
                  <p className="text-sm text-slate-500 mt-2">{currentTask.message}</p>
                )}
              </div>
            )}

            {currentTask.status === "completed" && currentTask.metrics && (
              <div className="p-4 bg-green-50 rounded-lg">
                <p className="font-medium text-green-700 mb-3">训练完成</p>
                <div className="grid grid-cols-2 gap-3">
                  {Object.entries(currentTask.metrics).slice(0, 4).map(([key, value]) => (
                    <div key={key} className="flex justify-between">
                      <span className="text-sm text-slate-600">{key}</span>
                      <span className="text-sm font-medium text-slate-900">
                        {typeof value === "number" ? value.toFixed(4) : value}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {currentTask.status === "failed" && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>训练失败</AlertTitle>
                <AlertDescription>
                  {currentTask.error_message || "训练过程中发生错误"}
                </AlertDescription>
              </Alert>
            )}
          </div>
        ) : (
          <div className="text-center py-8">
            {preprocessingStatus?.ready_for_training ? (
              <>
                <Play className="w-12 h-12 text-slate-300 mx-auto mb-4" />
                <p className="text-slate-500 mb-4">数据已就绪，可以开始训练</p>
                <Button onClick={startTraining} disabled={loading}>
                  {loading ? (
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  ) : (
                    <Play className="w-4 h-4 mr-2" />
                  )}
                  开始训练
                </Button>
              </>
            ) : (
              <>
                <AlertCircle className="w-12 h-12 text-amber-400 mx-auto mb-4" />
                <p className="text-slate-500 mb-2">需要先完成参数选择</p>
                <Button variant="outline" onClick={() => setActiveTab("embedding")}>
                  前往参数选择
                </Button>
              </>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )

  // ==================== 渲染评估结果 ====================

  const renderEvaluation = () => (
    <Card>
      <CardHeader>
        <CardTitle>评估结果</CardTitle>
        <CardDescription>查看模型评估指标</CardDescription>
      </CardHeader>
      <CardContent>
        {currentTask?.status === "completed" && currentTask.metrics ? (
          <div className="space-y-4">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {Object.entries(currentTask.metrics).map(([key, value]) => (
                <div key={key} className="p-4 bg-slate-50 rounded-lg">
                  <p className="text-xs text-slate-500 mb-1 uppercase">{key}</p>
                  <p className="text-xl font-bold text-slate-900">
                    {typeof value === "number" ? value.toFixed(4) : value}
                  </p>
                </div>
              ))}
            </div>

            {currentTask.topic_words && (
              <div className="mt-6">
                <h4 className="font-medium text-slate-900 mb-3">主题词</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {Object.entries(currentTask.topic_words).slice(0, 6).map(([topicId, words]) => (
                    <div key={topicId} className="p-3 bg-slate-50 rounded-lg">
                      <p className="text-sm font-medium text-slate-700 mb-2">
                        主题 {parseInt(topicId) + 1}
                      </p>
                      <div className="flex flex-wrap gap-1">
                        {(words as string[]).slice(0, 8).map((word, idx) => (
                          <Badge key={idx} variant="secondary" className="text-xs">
                            {word}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="text-center py-8">
            <FileCheck className="w-12 h-12 text-slate-300 mx-auto mb-4" />
            <p className="text-slate-500">训练完成后可查看评估结果</p>
          </div>
        )}
      </CardContent>
    </Card>
  )

  // ==================== 渲染可视化 ====================

  const renderVisualization = () => (
    <Card>
      <CardHeader>
        <CardTitle>可视化图表</CardTitle>
        <CardDescription>主题分析可视化展示</CardDescription>
      </CardHeader>
      <CardContent>
        {currentTask?.status === "completed" ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="aspect-video bg-slate-100 rounded-lg flex items-center justify-center">
              <div className="text-center">
                <BarChart3 className="w-8 h-8 text-slate-400 mx-auto mb-2" />
                <p className="text-sm text-slate-500">主题分布图</p>
              </div>
            </div>
            <div className="aspect-video bg-slate-100 rounded-lg flex items-center justify-center">
              <div className="text-center">
                <PieChart className="w-8 h-8 text-slate-400 mx-auto mb-2" />
                <p className="text-sm text-slate-500">词云图</p>
              </div>
            </div>
            <div className="aspect-video bg-slate-100 rounded-lg flex items-center justify-center">
              <div className="text-center">
                <BarChart3 className="w-8 h-8 text-slate-400 mx-auto mb-2" />
                <p className="text-sm text-slate-500">主题相似度热力图</p>
              </div>
            </div>
            <div className="aspect-video bg-slate-100 rounded-lg flex items-center justify-center">
              <div className="text-center">
                <PieChart className="w-8 h-8 text-slate-400 mx-auto mb-2" />
                <p className="text-sm text-slate-500">文档-主题分布</p>
              </div>
            </div>
          </div>
        ) : (
          <div className="text-center py-8">
            <PieChart className="w-12 h-12 text-slate-300 mx-auto mb-4" />
            <p className="text-slate-500">训练完成后可查看可视化图表</p>
          </div>
        )}
      </CardContent>
    </Card>
  )

  // ==================== 主渲染 ====================

  return (
    <div className="p-4 sm:p-6 lg:p-8">
      {/* 项目标题 */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-slate-900 mb-2">{projectName}</h1>
        <p className="text-slate-500">
          数据集: <span className="text-slate-700">{dataset}</span>
          {rowCount > 0 && (
            <> · <span className="text-slate-700">{rowCount.toLocaleString()}</span> 条数据</>
          )}
        </p>
      </div>

      {/* 工作流步骤 */}
      {renderWorkflowSteps()}

      {/* 内容区域 */}
      <div className="space-y-6">
        {activeTab === "overview" && renderOverview()}
        {activeTab === "data" && renderDataManagement()}
        {activeTab === "cleaning" && (
          <Card>
            <CardHeader>
              <CardTitle>数据清洗</CardTitle>
              <CardDescription>文本预处理和清洗配置</CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-slate-500">数据清洗功能将在参数选择与处理过程中自动执行</p>
            </CardContent>
          </Card>
        )}
        {activeTab === "embedding" && renderEmbedding()}
        {activeTab === "training" && renderTraining()}
        {activeTab === "evaluation" && renderEvaluation()}
        {activeTab === "visualization" && renderVisualization()}
      </div>
    </div>
  )
}
