"use client"

import { useRef, useState } from "react"
import { ApiError } from "@/lib/api/api-error"
import { Loader2, FolderPlus } from "lucide-react"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
  DialogDescription,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

// ==================== 类型定义 ====================
// 分析模式、主题数在数据预处理之后再配置

export interface NewProjectData {
  name: string
}

interface NewProjectDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  onSubmit: (data: NewProjectData) => Promise<void> | void
}

// ==================== 组件 ====================

export function NewProjectDialog({
  open,
  onOpenChange,
  onSubmit,
}: NewProjectDialogProps) {
  const [loading, setLoading] = useState(false)
  const [name, setName] = useState("")
  const [error, setError] = useState("")
  const submittingRef = useRef(false)

  const resetForm = () => {
    setName("")
    setError("")
  }

  const handleOpenChange = (isOpen: boolean) => {
    if (submittingRef.current) return
    if (!isOpen) resetForm()
    onOpenChange(isOpen)
  }

  const handleSubmit = async () => {
    if (submittingRef.current) return
    if (!name.trim()) {
      setError("请输入项目名称")
      return
    }

    submittingRef.current = true
    setLoading(true)
    setError("")
    try {
      await onSubmit({ name: name.trim() })
      resetForm()
      onOpenChange(false)
    } catch (error) {
      setError(error instanceof ApiError ? error.message : "创建失败，请稍后重试")
    } finally {
      submittingRef.current = false
      setLoading(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="sm:max-w-lg bg-white">
        <DialogHeader>
          <div className="flex items-center gap-3 mb-2">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center">
              <FolderPlus className="w-5 h-5 text-white" />
            </div>
            <div>
              <DialogTitle className="text-xl font-bold text-slate-900">
                新建分析项目
              </DialogTitle>
              <DialogDescription className="text-slate-500 text-sm">
                先起个名字，创建后上传数据并配置分析参数
              </DialogDescription>
            </div>
          </div>
        </DialogHeader>

        <div className="space-y-5 py-4">
          {/* 项目名称 */}
          <div className="space-y-2">
            <Label htmlFor="project-name" className="text-sm font-medium text-slate-700">
              项目名称
            </Label>
            <Input
              id="project-name"
              value={name}
              disabled={loading}
              onChange={(e) => {
                setName(e.target.value)
                setError("")
              }}
              placeholder="例如：Q1用户反馈分析"
              className="h-11"
              autoFocus
            />
            <p className="text-xs text-slate-400">
              项目名称不能重复；新项目的数据与训练结果独立保存
            </p>
          </div>

          {error && (
            <p role="alert" className="text-sm text-red-600 bg-red-50 px-3 py-2 rounded-lg">{error}</p>
          )}

          {/* 后续流程说明 */}
          <div className="bg-slate-50 rounded-xl p-4">
            <p className="text-sm font-medium text-slate-700 mb-2">创建后您将：</p>
            <div className="space-y-2 text-sm text-slate-600">
              <div className="flex items-center gap-2">
                <span className="w-6 h-6 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center text-xs font-bold">1</span>
                <span>在新页面<strong>上传数据文件</strong>（CSV、TXT、PDF 等）</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-6 h-6 rounded-full bg-slate-200 text-slate-600 flex items-center justify-center text-xs font-bold">2</span>
                <span>上传完成后<strong>选择数据列与参数</strong>，点击开始分析后执行预处理、训练、评估与可视化</span>
              </div>
            </div>
          </div>
        </div>

        <DialogFooter className="gap-2">
          <Button variant="outline" onClick={() => handleOpenChange(false)} disabled={loading}>
            取消
          </Button>
          <Button
            onClick={handleSubmit}
            disabled={loading || !name.trim()}
            className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 min-w-[100px]"
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : "创建项目"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
