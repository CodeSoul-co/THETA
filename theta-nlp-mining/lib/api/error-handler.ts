/**
 * 统一错误处理工具
 */

import { toast } from "sonner"

export interface ApiError {
  message: string
  status?: number
  code?: string
}

/**
 * 处理 API 错误并显示用户友好的提示
 */
export function handleApiError(error: unknown, defaultMessage = "操作失败") {
  let errorMessage = defaultMessage

  if (error instanceof Error) {
    errorMessage = error.message
  } else if (typeof error === "string") {
    errorMessage = error
  } else if (error && typeof error === "object" && "message" in error) {
    errorMessage = String(error.message)
  }

  // 根据错误类型显示不同的提示
  if (errorMessage.includes("网络") || errorMessage.includes("Network") || errorMessage.includes("Failed to fetch")) {
    toast.error("网络连接失败，请检查网络设置")
  } else if (errorMessage.includes("超时") || errorMessage.includes("timeout")) {
    toast.error("请求超时，请稍后重试")
  } else if (errorMessage.includes("401") || errorMessage.includes("未授权")) {
    toast.error("未授权，请重新登录")
  } else if (errorMessage.includes("403") || errorMessage.includes("禁止")) {
    toast.error("没有权限执行此操作")
  } else if (errorMessage.includes("404") || errorMessage.includes("未找到")) {
    toast.error("资源未找到")
  } else if (errorMessage.includes("500") || errorMessage.includes("服务器")) {
    toast.error("服务器错误，请稍后重试")
  } else {
    toast.error(errorMessage)
  }

  console.error("API Error:", error)
}

/**
 * 处理文件上传错误
 */
export function handleUploadError(error: unknown) {
  if (error instanceof Error) {
    if (error.message.includes("大小")) {
      toast.error("文件大小超过限制（最大 50MB）")
    } else if (error.message.includes("类型") || error.message.includes("格式")) {
      toast.error("不支持的文件类型。请上传 CSV、Excel 或 JSON 文件")
    } else {
      toast.error(`文件上传失败: ${error.message}`)
    }
  } else {
    toast.error("文件上传失败，请重试")
  }
}

/**
 * 处理分析任务错误
 */
export function handleTaskError(error: unknown) {
  if (error instanceof Error) {
    if (error.message.includes("配置")) {
      toast.error("请检查配置是否正确")
    } else if (error.message.includes("文件")) {
      toast.error("文件处理失败，请检查文件格式")
    } else {
      toast.error(`分析任务失败: ${error.message}`)
    }
  } else {
    toast.error("分析任务失败，请重试")
  }
}
