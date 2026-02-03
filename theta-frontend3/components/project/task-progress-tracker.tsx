"use client"

import { useState, useEffect, useCallback, useRef } from "react"
import {
  Play,
  Pause,
  RotateCcw,
  X,
  Check,
  Loader2,
  AlertCircle,
  ChevronDown,
  ChevronUp,
  Terminal,
  Clock,
  BarChart3,
  RefreshCw,
} from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { cn } from "@/lib/utils"
import { ETMAgentAPI, TaskResponse } from "@/lib/api/etm-agent"

// ==================== 类型定义 ====================

interface TaskProgressTrackerProps {
  taskId: string
  onComplete?: (task: TaskResponse) => void
  onCancel?: () => void
  autoRefresh?: boolean
  refreshInterval?: number
}

interface LogEntry {
  step: string
  status: string
  message: string
  timestamp: string
}

// ==================== 工具函数 ====================

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}秒`
  if (seconds < 3600) {
    const mins = Math.floor(seconds / 60)
    const secs = Math.round(seconds % 60)
    return `${mins}分${secs}秒`
  }
  const hours = Math.floor(seconds / 3600)
  const mins = Math.floor((seconds % 3600) / 60)
  return `${hours}时${mins}分`
}

function getStatusColor(status: string): string {
  switch (status) {
    case "completed":
    case "success":
      return "text-green-600 bg-green-50"
    case "running":
    case "in_progress":
      return "text-blue-600 bg-blue-50"
    case "failed":
    case "error":
      return "text-red-600 bg-red-50"
    case "pending":
    case "waiting":
      return "text-amber-600 bg-amber-50"
    case "cancelled":
      return "text-slate-600 bg-slate-100"
    default:
      return "text-slate-600 bg-slate-50"
  }
}

function getStatusLabel(status: string): string {
  switch (status) {
    case "completed":
      return "已完成"
    case "running":
      return "运行中"
    case "failed":
      return "失败"
    case "pending":
      return "等待中"
    case "cancelled":
      return "已取消"
    default:
      return status
  }
}

// ==================== 组件 ====================

export function TaskProgressTracker({
  taskId,
  onComplete,
  onCancel,
  autoRefresh = true,
  refreshInterval = 2000,
}: TaskProgressTrackerProps) {
  const [task, setTask] = useState<TaskResponse | null>(null)
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [showLogs, setShowLogs] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState("")
  const logsEndRef = useRef<HTMLDivElement>(null)

  // 获取任务状态
  const fetchTask = useCallback(async () => {
    try {
      const taskData = await ETMAgentAPI.getTask(taskId)
      setTask(taskData)
      setError("")

      // 检查是否完成
      if (["completed", "failed", "cancelled"].includes(taskData.status)) {
        if (taskData.status === "completed" && onComplete) {
          onComplete(taskData)
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "获取任务状态失败")
    } finally {
      setLoading(false)
    }
  }, [taskId, onComplete])

  // 获取日志
  const fetchLogs = useCallback(async () => {
    try {
      const logsData = await ETMAgentAPI.getTaskLogs(taskId)
      setLogs(logsData.logs)
    } catch (err) {
      console.error("Failed to fetch logs:", err)
    }
  }, [taskId])

  // 初始加载
  useEffect(() => {
    fetchTask()
    fetchLogs()
  }, [fetchTask, fetchLogs])

  // 自动刷新
  useEffect(() => {
    if (!autoRefresh || !task) return
    if (["completed", "failed", "cancelled"].includes(task.status)) return

    const interval = setInterval(() => {
      fetchTask()
      fetchLogs()
    }, refreshInterval)

    return () => clearInterval(interval)
  }, [autoRefresh, refreshInterval, task, fetchTask, fetchLogs])

  // 自动滚动日志
  useEffect(() => {
    if (showLogs && logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: "smooth" })
    }
  }, [logs, showLogs])

  // 取消任务
  const handleCancel = async () => {
    try {
      await ETMAgentAPI.cancelTask(taskId)
      fetchTask()
      onCancel?.()
    } catch (err) {
      setError(err instanceof Error ? err.message : "取消任务失败")
    }
  }

  // 渲染状态徽章
  const renderStatusBadge = () => {
    if (!task) return null

    const statusColor = getStatusColor(task.status)
    const statusLabel = getStatusLabel(task.status)

    return (
      <Badge className={cn("font-medium", statusColor)}>
        {task.status === "running" && (
          <Loader2 className="w-3 h-3 mr-1 animate-spin" />
        )}
        {task.status === "completed" && <Check className="w-3 h-3 mr-1" />}
        {task.status === "failed" && <AlertCircle className="w-3 h-3 mr-1" />}
        {statusLabel}
      </Badge>
    )
  }

  // 渲染进度条
  const renderProgress = () => {
    if (!task) return null

    return (
      <div className="space-y-2">
        <div className="flex items-center justify-between text-sm">
          <span className="text-slate-600">{task.current_step || "处理中"}</span>
          <span className="font-medium text-slate-900">{task.progress}%</span>
        </div>
        <Progress value={task.progress} className="h-2" />
        {task.message && (
          <p className="text-sm text-slate-500">{task.message}</p>
        )}
      </div>
    )
  }

  // 渲染任务信息
  const renderTaskInfo = () => {
    if (!task) return null

    return (
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="p-3 bg-slate-50 rounded-lg">
          <p className="text-xs text-slate-500 mb-1">数据集</p>
          <p className="font-medium text-slate-900 truncate">{task.dataset || "-"}</p>
        </div>
        <div className="p-3 bg-slate-50 rounded-lg">
          <p className="text-xs text-slate-500 mb-1">模式</p>
          <p className="font-medium text-slate-900">{task.mode || "zero_shot"}</p>
        </div>
        <div className="p-3 bg-slate-50 rounded-lg">
          <p className="text-xs text-slate-500 mb-1">主题数</p>
          <p className="font-medium text-slate-900">{task.num_topics || 20}</p>
        </div>
        <div className="p-3 bg-slate-50 rounded-lg">
          <p className="text-xs text-slate-500 mb-1">耗时</p>
          <p className="font-medium text-slate-900">
            {task.duration_seconds ? formatDuration(task.duration_seconds) : "-"}
          </p>
        </div>
      </div>
    )
  }

  // 渲染日志
  const renderLogs = () => (
    <Collapsible open={showLogs} onOpenChange={setShowLogs}>
      <CollapsibleTrigger asChild>
        <Button variant="ghost" className="w-full justify-between">
          <div className="flex items-center gap-2">
            <Terminal className="w-4 h-4" />
            <span>执行日志 ({logs.length})</span>
          </div>
          {showLogs ? (
            <ChevronUp className="w-4 h-4" />
          ) : (
            <ChevronDown className="w-4 h-4" />
          )}
        </Button>
      </CollapsibleTrigger>
      <CollapsibleContent>
        <ScrollArea className="h-64 mt-2 border rounded-lg bg-slate-900 p-4">
          <div className="space-y-2 font-mono text-sm">
            {logs.length === 0 ? (
              <p className="text-slate-400">暂无日志</p>
            ) : (
              logs.map((log, idx) => (
                <div key={idx} className="flex items-start gap-2">
                  <span className="text-slate-500 flex-shrink-0">
                    {new Date(log.timestamp).toLocaleTimeString()}
                  </span>
                  <span
                    className={cn(
                      "px-1.5 py-0.5 rounded text-xs font-medium flex-shrink-0",
                      log.status === "completed"
                        ? "bg-green-900/50 text-green-400"
                        : log.status === "error"
                        ? "bg-red-900/50 text-red-400"
                        : "bg-blue-900/50 text-blue-400"
                    )}
                  >
                    {log.step}
                  </span>
                  <span className="text-slate-300">{log.message}</span>
                </div>
              ))
            )}
            <div ref={logsEndRef} />
          </div>
        </ScrollArea>
      </CollapsibleContent>
    </Collapsible>
  )

  // 渲染结果概览
  const renderResults = () => {
    if (!task || task.status !== "completed") return null

    return (
      <div className="space-y-4">
        {task.metrics && (
          <div>
            <h4 className="font-medium text-slate-900 mb-2 flex items-center gap-2">
              <BarChart3 className="w-4 h-4" />
              评估指标
            </h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {Object.entries(task.metrics).map(([key, value]) => (
                <div key={key} className="p-3 bg-green-50 rounded-lg border border-green-100">
                  <p className="text-xs text-green-600 mb-1 uppercase">{key}</p>
                  <p className="text-lg font-bold text-green-700">
                    {typeof value === "number" ? value.toFixed(4) : value}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}

        {task.topic_words && Object.keys(task.topic_words).length > 0 && (
          <div>
            <h4 className="font-medium text-slate-900 mb-2">主题词预览</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              {Object.entries(task.topic_words).slice(0, 4).map(([topicId, words]) => (
                <div key={topicId} className="p-3 bg-slate-50 rounded-lg">
                  <p className="text-sm font-medium text-slate-700 mb-1">
                    主题 {parseInt(topicId) + 1}
                  </p>
                  <p className="text-sm text-slate-500 truncate">
                    {(words as string[]).slice(0, 5).join(", ")}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    )
  }

  // 加载中
  if (loading) {
    return (
      <Card>
        <CardContent className="py-8">
          <div className="flex items-center justify-center gap-3">
            <Loader2 className="w-6 h-6 text-blue-600 animate-spin" />
            <span className="text-slate-500">加载任务信息...</span>
          </div>
        </CardContent>
      </Card>
    )
  }

  // 错误
  if (error && !task) {
    return (
      <Card>
        <CardContent className="py-8">
          <div className="text-center">
            <AlertCircle className="w-12 h-12 text-red-400 mx-auto mb-3" />
            <p className="text-red-600 mb-2">{error}</p>
            <Button variant="outline" onClick={fetchTask}>
              <RefreshCw className="w-4 h-4 mr-2" />
              重试
            </Button>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Clock className="w-5 h-5" />
              任务进度
            </CardTitle>
            <CardDescription>任务 ID: {taskId}</CardDescription>
          </div>
          <div className="flex items-center gap-2">
            {renderStatusBadge()}
            {task?.status === "running" && (
              <Button variant="outline" size="sm" onClick={handleCancel}>
                <X className="w-4 h-4 mr-1" />
                取消
              </Button>
            )}
            <Button variant="ghost" size="sm" onClick={fetchTask}>
              <RefreshCw className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* 进度条 */}
        {task && ["running", "pending"].includes(task.status) && renderProgress()}

        {/* 任务信息 */}
        {renderTaskInfo()}

        {/* 结果 */}
        {renderResults()}

        {/* 日志 */}
        {renderLogs()}

        {/* 错误信息 */}
        {task?.status === "failed" && task.error_message && (
          <div className="p-4 bg-red-50 rounded-lg border border-red-200">
            <div className="flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
              <div>
                <p className="font-medium text-red-700">任务执行失败</p>
                <p className="text-sm text-red-600 mt-1">{task.error_message}</p>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

// ==================== 任务列表组件 ====================

interface TaskListProps {
  dataset?: string
  limit?: number
  onSelectTask?: (task: TaskResponse) => void
}

export function TaskList({ dataset, limit = 10, onSelectTask }: TaskListProps) {
  const [tasks, setTasks] = useState<TaskResponse[]>([])
  const [loading, setLoading] = useState(true)

  const fetchTasks = useCallback(async () => {
    try {
      const data = await ETMAgentAPI.getTasks({ dataset, limit })
      setTasks(data)
    } catch (err) {
      console.error("Failed to fetch tasks:", err)
    } finally {
      setLoading(false)
    }
  }, [dataset, limit])

  useEffect(() => {
    fetchTasks()
    const interval = setInterval(fetchTasks, 5000)
    return () => clearInterval(interval)
  }, [fetchTasks])

  if (loading) {
    return (
      <div className="flex items-center justify-center py-8">
        <Loader2 className="w-6 h-6 text-blue-600 animate-spin" />
      </div>
    )
  }

  if (tasks.length === 0) {
    return (
      <div className="text-center py-8">
        <Play className="w-12 h-12 text-slate-300 mx-auto mb-3" />
        <p className="text-slate-500">暂无训练任务</p>
      </div>
    )
  }

  return (
    <div className="space-y-3">
      {tasks.map((task) => (
        <div
          key={task.task_id}
          onClick={() => onSelectTask?.(task)}
          className={cn(
            "p-4 rounded-lg border transition-all cursor-pointer",
            task.status === "running"
              ? "bg-blue-50 border-blue-200 hover:border-blue-300"
              : task.status === "completed"
              ? "bg-white border-slate-200 hover:border-slate-300"
              : task.status === "failed"
              ? "bg-red-50 border-red-200 hover:border-red-300"
              : "bg-white border-slate-200 hover:border-slate-300"
          )}
        >
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <span className="font-medium text-slate-900">{task.dataset}</span>
              <Badge variant="secondary" className="text-xs">
                {task.mode || "zero_shot"}
              </Badge>
            </div>
            <Badge className={getStatusColor(task.status)}>
              {getStatusLabel(task.status)}
            </Badge>
          </div>
          {task.status === "running" && (
            <Progress value={task.progress} className="h-1.5" />
          )}
          <div className="flex items-center justify-between mt-2 text-xs text-slate-500">
            <span>主题数: {task.num_topics || 20}</span>
            <span>
              {task.created_at
                ? new Date(task.created_at).toLocaleString()
                : ""}
            </span>
          </div>
        </div>
      ))}
    </div>
  )
}
