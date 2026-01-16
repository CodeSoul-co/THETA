/**
 * ETM Agent API Client
 * 用于与后端 LangGraph Agent 通信
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// ==================== 类型定义 ====================

export interface TaskResponse {
  task_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  message: string;
  dataset?: string;
  mode?: string;
  num_topics?: number;
  result?: any;
  error?: string;
  created_at?: string;
  updated_at?: string;
}

export interface CreateTaskRequest {
  dataset: string;
  mode: 'zero_shot' | 'unsupervised' | 'supervised';
  num_topics?: number;
  vocab_size?: number;
  epochs?: number;
  batch_size?: number;
}

export interface DatasetInfo {
  name: string;
  path: string;
  file_count?: number;
  total_size?: string;
}

export interface ResultInfo {
  dataset: string;
  mode: string;
  path: string;
  created_at?: string;
  files?: string[];
}

export interface VisualizationInfo {
  name: string;
  type: string;
  path: string;
  url?: string;
}

export interface TopicWord {
  word: string;
  weight: number;
}

export interface MetricsResponse {
  coherence?: number;
  diversity?: number;
  perplexity?: number;
  [key: string]: any;
}

export interface PreprocessingJob {
  job_id: string;
  dataset: string;
  model: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  message: string;
  created_at?: string;
  bow_file?: string;
  embedding_file?: string;
}

export interface PreprocessingStatus {
  has_bow: boolean;
  has_embeddings: boolean;
  bow_file?: string;
  embedding_file?: string;
}

// ==================== API 请求函数 ====================

async function fetchApi<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  
  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

// ==================== ETM Agent API ====================

export const ETMAgentAPI = {
  // ========== 健康检查 ==========
  async healthCheck(): Promise<{ status: string; gpu_available: boolean }> {
    return fetchApi('/api/health');
  },

  // ========== 数据集管理 ==========
  async getDatasets(): Promise<DatasetInfo[]> {
    return fetchApi('/api/datasets');
  },

  async getDataset(name: string): Promise<DatasetInfo> {
    return fetchApi(`/api/datasets/${name}`);
  },

  // ========== 任务管理 ==========
  async getTasks(): Promise<TaskResponse[]> {
    return fetchApi('/api/tasks');
  },

  async getTask(taskId: string): Promise<TaskResponse> {
    return fetchApi(`/api/tasks/${taskId}`);
  },

  async createTask(request: CreateTaskRequest): Promise<TaskResponse> {
    return fetchApi('/api/tasks', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  },

  async cancelTask(taskId: string): Promise<{ message: string }> {
    return fetchApi(`/api/tasks/${taskId}`, {
      method: 'DELETE',
    });
  },

  // ========== 结果查询 ==========
  async getResults(): Promise<ResultInfo[]> {
    return fetchApi('/api/results');
  },

  async getResult(dataset: string, mode: string): Promise<ResultInfo> {
    return fetchApi(`/api/results/${dataset}/${mode}`);
  },

  async getTopicWords(dataset: string, mode: string, topK: number = 10): Promise<TopicWord[][]> {
    return fetchApi(`/api/results/${dataset}/${mode}/topics?top_k=${topK}`);
  },

  async getMetrics(dataset: string, mode: string): Promise<MetricsResponse> {
    return fetchApi(`/api/results/${dataset}/${mode}/metrics`);
  },

  // ========== 可视化 ==========
  async getVisualizations(dataset: string, mode: string): Promise<VisualizationInfo[]> {
    return fetchApi(`/api/visualizations/${dataset}/${mode}`);
  },

  // ========== 聊天接口 ==========
  async chat(message: string): Promise<{ response: string; task?: CreateTaskRequest }> {
    return fetchApi('/api/chat', {
      method: 'POST',
      body: JSON.stringify({ message }),
    });
  },

  // ========== 预处理 (Embedding) ==========
  async getEmbeddingModels(): Promise<{ models: string[] }> {
    return fetchApi('/api/preprocessing/models');
  },

  async startPreprocessing(params: {
    dataset: string;
    model?: string;
  }): Promise<PreprocessingJob> {
    return fetchApi('/api/preprocessing/start', {
      method: 'POST',
      body: JSON.stringify(params),
    });
  },

  async getPreprocessingJob(jobId: string): Promise<PreprocessingJob> {
    return fetchApi(`/api/preprocessing/${jobId}`);
  },

  async getPreprocessingJobs(): Promise<PreprocessingJob[]> {
    return fetchApi('/api/preprocessing');
  },

  async cancelPreprocessingJob(jobId: string): Promise<{ message: string }> {
    return fetchApi(`/api/preprocessing/${jobId}`, {
      method: 'DELETE',
    });
  },

  async checkPreprocessingStatus(dataset: string): Promise<PreprocessingStatus> {
    return fetchApi(`/api/preprocessing/check/${dataset}`);
  },
};

export default ETMAgentAPI;
