/**
 * 文件上传 Hook
 * 支持前端文件解析（无需后端）
 */

import { useState, useCallback } from 'react'
import { parseFile, type ParsedFileData } from '@/lib/utils/file-parser'

export interface UploadedFile {
  id: string
  name: string
  size: number
  type: string
  fileId: string
  parsedData?: ParsedFileData
}

export function useFileUpload() {
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([])
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState<Record<string, number>>({})
  const [error, setError] = useState<string | null>(null)

  const uploadFile = useCallback(async (file: File) => {
    // 验证文件类型
    const allowedTypes = [
      'text/csv',
      'application/vnd.ms-excel',
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      'application/json',
    ]
    
    if (!allowedTypes.includes(file.type) && !file.name.match(/\.(csv|xlsx|xls|json)$/i)) {
      setError('不支持的文件类型。请上传 CSV、Excel 或 JSON 文件。')
      return null
    }

    // 验证文件大小 (最大 50MB)
    const maxSize = 50 * 1024 * 1024
    if (file.size > maxSize) {
      setError('文件大小超过限制（最大 50MB）')
      return null
    }

    setUploading(true)
    setError(null)

    // 模拟上传进度
    const progressInterval = setInterval(() => {
      setUploadProgress((prev) => {
        const current = prev[file.name] || 0
        if (current < 90) {
          return {
            ...prev,
            [file.name]: current + 10,
          }
        }
        return prev
      })
    }, 100)

    try {
      // 前端解析文件
      const parsedData = await parseFile(file)
      
      // 生成文件 ID
      const fileId = `file_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
      
      // 完成上传进度
      clearInterval(progressInterval)
      setUploadProgress((prev) => ({
        ...prev,
        [file.name]: 100,
      }))

      const uploadedFile: UploadedFile = {
        id: fileId,
        name: file.name,
        size: file.size,
        type: file.type,
        fileId,
        parsedData,
      }

      setUploadedFiles((prev) => [...prev, uploadedFile])
      
      // 清除进度
      setTimeout(() => {
        setUploadProgress((prev) => {
          const next = { ...prev }
          delete next[file.name]
          return next
        })
      }, 500)

      return uploadedFile
    } catch (err) {
      clearInterval(progressInterval)
      setError(err instanceof Error ? err.message : '文件解析失败')
      return null
    } finally {
      setUploading(false)
    }
  }, [])

  const removeFile = useCallback((fileId: string) => {
    setUploadedFiles((prev) => prev.filter((f) => f.id !== fileId))
  }, [])

  const clearFiles = useCallback(() => {
    setUploadedFiles([])
    setError(null)
  }, [])

  return {
    uploadedFiles,
    uploading,
    uploadProgress,
    error,
    uploadFile,
    removeFile,
    clearFiles,
  }
}
