/**
 * API 端点定义
 * 统一管理所有 API 路径
 */

export const API_ENDPOINTS = {
  // 文件上传
  upload: '/upload',
  
  // 文件解析
  parseFile: '/parse-file',
  
  // 分析任务
  analyze: '/analyze',
  taskStatus: (taskId: string) => `/task/${taskId}/status`,
  
  // 配置管理
  config: '/config',
  
  // RAG 相关
  ragSearch: '/rag/search',
  knowledgeBases: '/knowledge-bases',
  document: (docId: string) => `/documents/${docId}`,
  
  // 分析数据
  analytics: {
    timeline: '/analytics/timeline',
    umap: '/analytics/umap',
    hierarchy: '/analytics/hierarchy',
    keywords: '/analytics/keywords',
  },
  
  // AI Agent
  agentStream: '/agent/stream',
  agentQuery: '/agent/query',
} as const
