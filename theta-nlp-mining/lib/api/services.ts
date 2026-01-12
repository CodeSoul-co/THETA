/**
 * API 服务层
 * 封装具体的业务 API 调用
 * 支持后端 API 调用，如果后端不可用则返回错误（由调用方处理模拟逻辑）
 */

import { apiClient, ApiResponse } from './client'
import { API_ENDPOINTS } from './endpoints'

// ==================== 类型定义 ====================

export interface UploadResponse {
  fileId: string
  filename: string
  size: number
  type: string
}

export interface FileParseResponse {
  headers: string[]
  rowCount: number
  sampleRows: Record<string, any>[]
}

export interface AnalyzeRequest {
  fileId: string
  fieldMapping: {
    filename: string
    content: string
    modified: string
  }
  model: string
}

export interface AnalyzeResponse {
  taskId: string
  status: 'pending' | 'processing' | 'completed' | 'failed'
}

export interface TaskStatusResponse {
  taskId: string
  status: 'pending' | 'processing' | 'completed' | 'failed'
  progress?: number
  result?: any
  error?: string
}

export interface ConfigData {
  fieldMapping: {
    filename: string
    content: string
    modified: string
  }
  selectedModel: string
  uploadedFiles: string[]
}

export interface RAGSearchRequest {
  query: string
  knowledgeBases?: string[]
}

export interface RAGSearchResponse {
  results: {
    content: string
    citations: Citation[]
    score: number
  }[]
}

export interface Citation {
  id: number
  source: string
  type: 'paper' | 'pdf' | 'database'
  page?: number
  url?: string
}

export interface KnowledgeBase {
  id: string
  name: string
  type: string
  count: number
}

export interface TimelineData {
  month: string
  count: number
}

export interface UMAPPoint {
  x: number
  y: number
  cluster: number
  documentId?: string
}

export interface KeywordData {
  word: string
  weight: number
}

// ==================== 文件上传服务 ====================

export const fileService = {
  /**
   * 上传文件
   */
  async uploadFile(
    file: File,
    onProgress?: (progress: number) => void
  ): Promise<ApiResponse<UploadResponse>> {
    return apiClient.uploadFile(API_ENDPOINTS.upload, file, onProgress)
  },

  /**
   * 解析文件（获取表头等信息）
   */
  async parseFile(fileId: string): Promise<ApiResponse<FileParseResponse>> {
    return apiClient.get<FileParseResponse>(API_ENDPOINTS.parseFile, { fileId })
  },
}

// ==================== 分析服务 ====================

export const analyzeService = {
  /**
   * 启动分析任务
   */
  async startAnalysis(data: AnalyzeRequest): Promise<ApiResponse<AnalyzeResponse>> {
    return apiClient.post<AnalyzeResponse>(API_ENDPOINTS.analyze, data)
  },

  /**
   * 查询任务状态
   */
  async getTaskStatus(taskId: string): Promise<ApiResponse<TaskStatusResponse>> {
    return apiClient.get<TaskStatusResponse>(API_ENDPOINTS.taskStatus(taskId))
  },
}

// ==================== 配置服务 ====================

export const configService = {
  /**
   * 保存配置
   */
  async saveConfig(config: ConfigData): Promise<ApiResponse> {
    return apiClient.post(API_ENDPOINTS.config, config)
  },

  /**
   * 获取配置
   */
  async getConfig(): Promise<ApiResponse<ConfigData>> {
    return apiClient.get<ConfigData>(API_ENDPOINTS.config)
  },
}

// ==================== RAG 服务 ====================

export const ragService = {
  /**
   * RAG 搜索
   */
  async search(query: string, knowledgeBases?: string[]): Promise<ApiResponse<RAGSearchResponse>> {
    return apiClient.post<RAGSearchResponse>(API_ENDPOINTS.ragSearch, {
      query,
      knowledgeBases,
    })
  },

  /**
   * 获取知识库列表
   */
  async getKnowledgeBases(): Promise<ApiResponse<KnowledgeBase[]>> {
    return apiClient.get<KnowledgeBase[]>(API_ENDPOINTS.knowledgeBases)
  },

  /**
   * 获取文档内容
   */
  async getDocument(docId: string): Promise<ApiResponse<any>> {
    return apiClient.get(API_ENDPOINTS.document(docId))
  },
}

// ==================== 分析数据服务 ====================

export const analyticsService = {
  /**
   * 获取时序数据
   */
  async getTimelineData(taskId?: string): Promise<ApiResponse<TimelineData[]>> {
    return apiClient.get<TimelineData[]>(API_ENDPOINTS.analytics.timeline, {
      taskId,
    })
  },

  /**
   * 获取 UMAP 数据
   */
  async getUMAPData(taskId?: string): Promise<ApiResponse<UMAPPoint[]>> {
    return apiClient.get<UMAPPoint[]>(API_ENDPOINTS.analytics.umap, {
      taskId,
    })
  },

  /**
   * 获取层次聚类数据
   */
  async getHierarchyData(taskId?: string): Promise<ApiResponse<any>> {
    return apiClient.get(API_ENDPOINTS.analytics.hierarchy, {
      taskId,
    })
  },

  /**
   * 获取关键词数据
   */
  async getKeywordsData(taskId?: string): Promise<ApiResponse<KeywordData[]>> {
    return apiClient.get<KeywordData[]>(API_ENDPOINTS.analytics.keywords, {
      taskId,
    })
  },
}

// ==================== AI Agent 服务 ====================

export const agentService = {
  /**
   * 发送 AI Agent 查询
   */
  async query(message: string, context?: any): Promise<ApiResponse<any>> {
    return apiClient.post(API_ENDPOINTS.agentQuery, {
      message,
      context,
    })
  },
}
