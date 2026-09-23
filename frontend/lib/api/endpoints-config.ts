/**
 * THETA API 端点配置
 * 自动生成自 theta_project/config/api_endpoints.yaml
 *
 * 使用方式：
 *   import { API_ENDPOINTS, buildUrl } from '@/lib/api/endpoints-config';
 *
 *   // 固定路径
 *   fetch(`${API_BASE}${API_ENDPOINTS.auth.login}`, ...)
 *
 *   // 带参数路径
 *   const statusUrl = buildUrl('training', 'status', { job_id: 123 });
 *   // => /api/train/123/status
 *
 * ⚠️ 修改此文件时，请同步更新 theta_project/config/api_endpoints.yaml
 */

// =============================================================================
// 基础 URL（前端通过环境变量指定后端地址）
// =============================================================================

export const API_BASE_URLS = {
  development: "http://localhost:8000",
  production: "/api/backend",
  backup: "http://47.86.49.93:8000",
} as const;

// 前端当前使用的基础 URL（从环境变量或默认值）
export const getApiBaseUrl = (): string => {
  return (process.env.NEXT_PUBLIC_API_URL as string) || API_BASE_URLS.production;
};

// =============================================================================
// 端点配置
// =============================================================================

export const API_ENDPOINTS = {
  // --------------------------------------------------------------------------
  // 认证模块
  // --------------------------------------------------------------------------
  auth: {
    /** 用户注册 */
    register: "/api/auth/register",
    /** 用户登录 */
    login: "/api/auth/login",
    /** 获取当前用户信息 */
    me: "/api/auth/me",
  },

  // --------------------------------------------------------------------------
  // 文件上传模块
  // --------------------------------------------------------------------------
  upload: {
    /** 上传文件到后端本地存储 */
    file: "/api/upload",
    /** 测试上传接口（无需认证） */
    test: "/api/upload/test",
    /** 通知上传完成（OSS 直传后回调） */
    complete: "/api/upload/complete",
    /** 获取文件列表 */
    files: "/api/files",
    /** 获取 STS 临时凭证（前端直传 OSS） */
    sts_token: "/api/oss/sts-token",
  },

  // --------------------------------------------------------------------------
  // 预处理模块
  // --------------------------------------------------------------------------
  preprocessing: {
    /** 检查预处理状态 */
    check: "/api/preprocessing/check/{dataset}",
    /** 启动预处理任务 */
    start: "/api/preprocessing/start",
    /** 获取预处理任务状态 */
    status: "/api/preprocessing/{job_id}",
  },

  // --------------------------------------------------------------------------
  // 训练任务模块
  // --------------------------------------------------------------------------
  training: {
    /** 提交训练任务 */
    start: "/api/train/start",
    /** 查询训练状态 */
    status: "/api/train/{job_id}/status",
    /** 获取训练指标（loss/accuracy 曲线） */
    metrics: "/api/train/{job_id}/metrics",
    /** 获取训练摘要 */
    summary: "/api/train/{job_id}/summary",
    /** 获取训练任务列表 */
    jobs: "/api/train/jobs",
    /** DLC 回调通知 */
    callback: "/api/train/callback",
  },

  // --------------------------------------------------------------------------
  // 数据查询模块
  // --------------------------------------------------------------------------
  data: {
    /** 获取有结果的数据集列表 */
    oss_datasets: "/api/data/oss-datasets",
    /** 获取可用模型列表 */
    results_models: "/api/results/{dataset}/models",
    /** 获取主题词 */
    topic_words: "/api/results/{dataset}/topic-words",
    /** 获取评估指标 */
    metrics: "/api/results/{dataset}/metrics",
    /** 获取可视化文件列表 */
    visualizations: "/api/results/{dataset}/visualizations",
  },

  // --------------------------------------------------------------------------
  // AI 对话模块
  // --------------------------------------------------------------------------
  chat: {
    /** 发送对话消息 */
    send: "/api/agent/chat",
    /** 获取会话历史 */
    history: "/api/chat/history/{session_id}",
  },

  // --------------------------------------------------------------------------
  // 结果解读模块
  // --------------------------------------------------------------------------
  interpret: {
    /** AI 解读评估指标 */
    metrics: "/api/interpret/metrics",
    /** AI 解读主题语义 */
    topics: "/api/interpret/topics",
    /** AI 生成分析摘要 */
    summary: "/api/interpret/summary",
  },

  // --------------------------------------------------------------------------
  // 图表分析模块
  // --------------------------------------------------------------------------
  vision: {
    /** 使用 Qwen-VL 分析图表 */
    analyze_chart: "/api/vision/analyze-chart",
  },

  // --------------------------------------------------------------------------
  // 数据集管理
  // --------------------------------------------------------------------------
  datasets: {
    /** 删除数据集 */
    delete: "/api/datasets/{dataset}",
  },

  // --------------------------------------------------------------------------
  // 系统接口
  // --------------------------------------------------------------------------
  system: {
    /** 健康检查 */
    health: "/health",
    /** Swagger 文档 */
    docs: "/docs",
    /** ReDoc 文档 */
    redoc: "/redoc",
  },
} as const;

// =============================================================================
// 辅助函数
// =============================================================================

export type EndpointCategory = keyof typeof API_ENDPOINTS;

export function buildUrl(
  category: EndpointCategory,
  key: string,
  params?: Record<string, string | number>
): string {
  const categoryEndpoints = API_ENDPOINTS[category] as Record<string, string>;
  let path = categoryEndpoints[key];

  if (!path) {
    throw new Error(`Endpoint not found: ${category}.${key}`);
  }

  if (params) {
    for (const [k, v] of Object.entries(params)) {
      path = path.replace(`{${k}}`, String(v));
    }
  }

  return path;
}

export function buildFullUrl(
  category: EndpointCategory,
  key: string,
  params?: Record<string, string | number>,
  baseUrl?: string
): string {
  const base = baseUrl || getApiBaseUrl();
  const path = buildUrl(category, key, params);
  return `${base}${path}`;
}

// =============================================================================
// 类型定义
// =============================================================================

// 端点元数据（从 api_endpoints.yaml 提取）
export interface EndpointMeta {
  method: "GET" | "POST" | "PUT" | "DELETE" | "PATCH";
  auth_required: boolean;
  description: string;
}

// 端点路径参数类型（从 buildUrl 的 params 推断）
export type EndpointParams<C extends EndpointCategory, K extends keyof typeof API_ENDPOINTS[C]> =
  Parameters<typeof buildUrl>[2] extends undefined ? undefined : Parameters<typeof buildUrl>[2];

// 构建请求 URL 的类型安全封装
export function buildRequestUrl<C extends EndpointCategory>(
  category: C,
  key: string,
  params?: Record<string, string | number>
): string {
  return buildUrl(category, key, params);
}

// 常用请求构建
export const API = {
  // Auth
  auth: {
    register: () => buildFullUrl("auth", "register"),
    login: () => buildFullUrl("auth", "login"),
    me: () => ({
      url: buildFullUrl("auth", "me"),
      credentials: "same-origin" as RequestCredentials,
    }),
  },

  // Upload
  upload: {
    file: () => ({
      url: buildFullUrl("upload", "file"),
      credentials: "same-origin" as RequestCredentials,
    }),
    stsToken: (datasetName: string) => ({
      url: buildFullUrl("upload", "sts_token") + `?dataset_name=${encodeURIComponent(datasetName)}`,
      credentials: "same-origin" as RequestCredentials,
    }),
  },

  // Training
  training: {
    start: () => buildFullUrl("training", "start"),
    status: (jobId: number) => buildFullUrl("training", "status", { job_id: jobId }),
    metrics: (jobId: number) => buildFullUrl("training", "metrics", { job_id: jobId }),
    summary: (jobId: number) => buildFullUrl("training", "summary", { job_id: jobId }),
    jobs: () => buildFullUrl("training", "jobs"),
  },

  // Data
  data: {
    datasets: () => buildFullUrl("data", "oss_datasets"),
    models: (dataset: string) => buildFullUrl("data", "results_models", { dataset }),
    topicWords: (dataset: string, model?: string) => {
      const base = buildFullUrl("data", "topic_words", { dataset });
      return model ? `${base}?model=${encodeURIComponent(model)}` : base;
    },
    metrics: (dataset: string, model?: string) => {
      const base = buildFullUrl("data", "metrics", { dataset });
      return model ? `${base}?model=${encodeURIComponent(model)}` : base;
    },
    visualizations: (dataset: string, model?: string) => {
      const base = buildFullUrl("data", "visualizations", { dataset });
      return model ? `${base}?model=${encodeURIComponent(model)}` : base;
    },
  },

  // Chat
  chat: {
    send: () => buildFullUrl("chat", "send"),
    history: (sessionId: string) => buildFullUrl("chat", "history", { session_id: sessionId }),
  },
} as const;
