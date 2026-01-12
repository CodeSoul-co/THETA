/**
 * API 客户端配置
 * 支持后端 API 调用，如果后端不可用则返回错误（由调用方处理）
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000/api'

export interface ApiResponse<T = any> {
  success: boolean
  data?: T
  error?: string
  message?: string
}

export interface ApiError {
  message: string
  status?: number
  code?: string
}

class ApiClient {
  private baseURL: string

  constructor(baseURL: string = API_BASE_URL) {
    this.baseURL = baseURL
  }

  /**
   * 通用请求方法
   */
  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> {
    const url = `${this.baseURL}${endpoint}`
    
    const defaultHeaders: HeadersInit = {
      'Content-Type': 'application/json',
    }

    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          ...defaultHeaders,
          ...options.headers,
        },
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({
          message: `HTTP ${response.status}: ${response.statusText}`,
        }))
        
        throw {
          message: errorData.message || '请求失败',
          status: response.status,
          code: errorData.code,
        } as ApiError
      }

      const data = await response.json()
      return {
        success: true,
        data,
      }
    } catch (error) {
      if (error && typeof error === 'object' && 'message' in error) {
        return {
          success: false,
          error: (error as ApiError).message,
        }
      }
      return {
        success: false,
        error: error instanceof Error ? error.message : '未知错误',
      }
    }
  }

  /**
   * GET 请求
   */
  async get<T>(endpoint: string, params?: Record<string, any>): Promise<ApiResponse<T>> {
    const queryString = params
      ? '?' + new URLSearchParams(params).toString()
      : ''
    return this.request<T>(`${endpoint}${queryString}`, {
      method: 'GET',
    })
  }

  /**
   * POST 请求
   */
  async post<T>(endpoint: string, data?: any): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, {
      method: 'POST',
      body: data ? JSON.stringify(data) : undefined,
    })
  }

  /**
   * PUT 请求
   */
  async put<T>(endpoint: string, data?: any): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, {
      method: 'PUT',
      body: data ? JSON.stringify(data) : undefined,
    })
  }

  /**
   * DELETE 请求
   */
  async delete<T>(endpoint: string): Promise<ApiResponse<T>> {
    return this.request<T>(endpoint, {
      method: 'DELETE',
    })
  }

  /**
   * 文件上传
   */
  async uploadFile(
    endpoint: string,
    file: File,
    onProgress?: (progress: number) => void
  ): Promise<ApiResponse> {
    const url = `${this.baseURL}${endpoint}`
    const formData = new FormData()
    formData.append('file', file)

    return new Promise((resolve) => {
      const xhr = new XMLHttpRequest()

      // 上传进度
      if (onProgress) {
        xhr.upload.addEventListener('progress', (e) => {
          if (e.lengthComputable) {
            const progress = (e.loaded / e.total) * 100
            onProgress(progress)
          }
        })
      }

      xhr.addEventListener('load', () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const data = JSON.parse(xhr.responseText)
            resolve({
              success: true,
              data,
            })
          } catch {
            resolve({
              success: true,
              data: xhr.responseText,
            })
          }
        } else {
          resolve({
            success: false,
            error: `HTTP ${xhr.status}: ${xhr.statusText}`,
          })
        }
      })

      xhr.addEventListener('error', () => {
        resolve({
          success: false,
          error: '网络错误',
        })
      })

      xhr.open('POST', url)
      xhr.send(formData)
    })
  }
}

// 导出单例
export const apiClient = new ApiClient()

// 导出类型
export type { ApiClient }
