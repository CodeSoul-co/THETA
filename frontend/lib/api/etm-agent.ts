/**
 * THETA 工作台 API 客户端。
 * 手动数据操作经 /api/backend 访问本机服务，对话经 /api/v3 访问本机 Agent。
 */

import { apiFetch, API_BASE, AGENT_BASE } from './config';
import { ApiError } from './api-error';
import { SimpleETMAPI, BackendAPI, type TrainStatusResponse } from './backend';

const USER_ID_REGEX = /^[a-zA-Z0-9_-]+$/;

function getCurrentUserId(): string {
  if (typeof window === 'undefined') {
    return '';
  }

  const storedUser = localStorage.getItem('user');
  if (!storedUser) {
    return '';
  }

  try {
    const parsed = JSON.parse(storedUser) as {
      id?: number;
      user_id?: number | string;
      username?: string;
    };
    // 优先使用数据库数字用户 ID（与 OSS 路径 data/{id}/、DLC /mnt/data/{id}/ 一致）
    const rawId = parsed.id ?? parsed.user_id;
    if (rawId !== undefined && rawId !== null && String(rawId).trim() !== '') {
      const sid = String(rawId).trim();
      if (/^\d+$/.test(sid)) return sid;
    }
    const candidate = (parsed.username || '').trim();
    if (!candidate) return '';
    if (!USER_ID_REGEX.test(candidate)) {
      throw new Error('user_id 格式无效，只允许字母、数字、下划线和连字符');
    }
    return candidate;
  } catch (error) {
    if (error instanceof Error && error.message.includes('user_id 格式无效')) {
      throw error;
    }
    return '';
  }
}

function ensureUserIdOrThrow(): string {
  const userId = getCurrentUserId();
  if (!userId) {
    throw new Error('缺少 user_id，请先登录后再执行该操作');
  }
  return userId;
}

export function getVisualizationUrl(
  userId: string,
  datasetName: string,
  modelName: string,
  chartName: string,
  ossDomain = 'theta-prod-20260123.oss-cn-shanghai.aliyuncs.com',
): string {
  if (!USER_ID_REGEX.test(userId)) {
    throw new Error('user_id 格式无效，只允许字母、数字、下划线和连字符');
  }
  return `https://${ossDomain}/${encodeURIComponent(userId)}/result/${encodeURIComponent(datasetName)}/${encodeURIComponent(modelName)}/visualization/global/${encodeURIComponent(chartName)}`;
}

export function getVisualizationAssetUrl(
  userId: string,
  datasetName: string,
  modelName: string,
  relativePath: string,
  ossDomain = 'theta-prod-20260123.oss-cn-shanghai.aliyuncs.com',
): string {
  if (!USER_ID_REGEX.test(userId)) {
    throw new Error('user_id 格式无效，只允许字母、数字、下划线和连字符');
  }
  const normalizedPath = relativePath.replace(/^\/+/, '');
  return `https://${ossDomain}/${encodeURIComponent(userId)}/result/${encodeURIComponent(datasetName)}/${encodeURIComponent(modelName)}/visualization/${normalizedPath.split('/').map(encodeURIComponent).join('/')}`;
}

// ==================== 类型定义 ====================

export interface TaskResponse {
  task_id: string;
  status: 'pending_upload' | 'submitting_dlc' | 'training' | 'completed' | 'error'
    | 'pending' | 'running' | 'failed' | 'cancelled';
  current_step?: string;
  progress: number;
  message?: string;

  dataset?: string;
  mode?: string;
  num_topics?: number;

  metrics?: Record<string, number>;
  topic_words?: Record<string, string[]>;
  visualization_paths?: string[];

  created_at?: string;
  updated_at?: string;
  completed_at?: string;
  duration_seconds?: number;

  dlc_job_id?: string;
  dlc_status?: string;
  is_dlc?: boolean;
  error_message?: string;

  result?: any;
  error?: string;
}

function normalizeTaskStatus(status: string): TaskResponse['status'] {
  switch (status) {
    case 'creating':
      return 'pending';
    case 'running':
    case 'cancelling':
      return 'training';
    case 'succeeded':
      return 'completed';
    case 'pending':
    case 'pending_upload':
    case 'submitting_dlc':
    case 'training':
    case 'completed':
    case 'error':
    case 'failed':
    case 'cancelled':
      return status;
    default:
      return 'error';
  }
}

export interface CreateTaskRequest {
  embedding_provider?: 'local' | 'cloud';
  cloud_confirmed?: boolean;
  cloud_selection?: { provider: string; endpoint: string; model: string };
  external_request_limit?: number;
  plot_language?: string;
  stopwords_id?: number;
  model_params?: Record<string, Record<string, unknown>>;
  text_column?: string;
  meta_columns?: string[];
  time_column?: string;
  label_column?: string;
  dataset: string;
  language?: string;
  mode: 'zero_shot' | 'unsupervised' | 'supervised';
  num_topics?: number;
  vocab_size?: number;
  epochs?: number;
  batch_size?: number;
  learning_rate?: number;
  hidden_dim?: number;
  patience?: number;
  model_size?: string;
  models?: string;
}

export interface DatasetInfo {
  name: string;
  path: string;
  file_count?: number;
  total_size?: string;
  size?: number;
  created_at?: string;
}

/** 数据库中的用户项目（需登录） */
export interface ProjectInfo {
  id: number;
  name: string;
  dataset_name?: string | null;
  mode: string;
  num_topics: number;
  status: string;
  pipeline_status?: string | null;
  task_id?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
}

export interface ResultInfo {
  dataset: string;
  mode: string;
  timestamp: string;
  path: string;
  num_topics?: number;
  vocab_size?: number;
  epochs_trained?: number;
  metrics?: Record<string, number>;
  has_model?: boolean;
  has_theta?: boolean;
  has_beta?: boolean;
  has_topic_words?: boolean;
  has_visualizations?: boolean;
}

export interface TopicWord { word: string; weight: number; }

export interface MetricsResponse {
  coherence?: number;
  diversity?: number;
  perplexity?: number;
  [key: string]: any;
}

export type PreprocessingJobStatus =
  | 'pending' | 'bow_generating' | 'bow_completed'
  | 'embedding_generating' | 'embedding_completed'
  | 'running' | 'completed' | 'failed';

export interface PreprocessingJob {
  job_id: string;
  dataset: string;
  model?: string;
  status: PreprocessingJobStatus;
  progress: number;
  message: string | null;
  current_stage?: string | null;
  error_message?: string | null;
  created_at?: string;
  updated_at?: string;
  bow_path?: string | null;
  embedding_path?: string | null;
  vocab_path?: string | null;
}

export interface PreprocessingStatus {
  managed_by_training?: boolean;
  dataset?: string;
  has_bow: boolean;
  has_embeddings: boolean;
  ready_for_training: boolean;
  bow_path?: string | null;
  embedding_path?: string | null;
  vocab_path?: string | null;
}

export interface AgentChatImagePayload {
  name: string;
  mimeType: string;
  size: number;
  dataUrl: string;
}

export interface AgentChatFilePayload {
  name: string;
  mimeType: string;
  size: number;
  dataUrl: string;
}

export interface AgentChartDataSource {
  name: string;
  path: string;
  format: 'csv' | 'json';
  content: string;
  truncated: boolean;
  totalCharacters: number;
}

export interface AgentChartDataPayload {
  referenceId?: string;
  chartName: string;
  chartPath: string;
  dataset: string;
  model: string;
  sources: AgentChartDataSource[];
}

// ==================== 主 API ====================

export const ETMAgentAPI = {
  // ========== 健康检查 ==========
  async healthCheck(): Promise<{ status: string; gpu_available?: boolean }> {
    return apiFetch(API_BASE, '/health');
  },

  // ========== 后端配置 ==========
  async getConfig(): Promise<{
    oss_bucket: string;
    supported_models: string[];
    supported_modes: string[];
    supported_model_sizes: string[];
    default_num_topics: number;
    default_epochs: number;
  }> {
    return apiFetch(API_BASE, '/config');
  },

  // ========== 项目管理（数据库，需登录） ==========
  // 后端暂不支持项目管理，简化处理

  async getProjects(): Promise<ProjectInfo[]> {
    return apiFetch(API_BASE, '/api/projects');
  },

  async createProject(data: { name: string; dataset_name?: string; mode?: string; num_topics?: number }): Promise<ProjectInfo> {
    return apiFetch(API_BASE, '/api/projects', { method: 'POST', body: JSON.stringify(data) });
  },

  async updateProject(
    id: number,
    data: Partial<{ name: string; dataset_name: string; mode: string; num_topics: number; status: string; pipeline_status: string; task_id: string | null }>,
  ): Promise<ProjectInfo> {
    return apiFetch(API_BASE, `/api/projects/${id}`, { method: 'PATCH', body: JSON.stringify(data) });
  },

  async deleteProject(id: number): Promise<void> {
    await apiFetch(API_BASE, `/api/projects/${id}`, { method: 'DELETE' });
  },

  // ========== 数据集管理 ==========

  async getDatasets(): Promise<DatasetInfo[]> {
    try {
      console.log('[getDatasets] Fetching files...');
      const files = await BackendAPI.getFiles();
      console.log('[getDatasets] Files received:', files);

      // 按 dataset_name 分组
      const datasetMap = new Map<string, DatasetInfo>();

      for (const f of files) {
        const dsName = f.dataset_name || '未命名';

        if (datasetMap.has(dsName)) {
          // 已有该数据集，更新文件数和路径
          const existing = datasetMap.get(dsName)!;
          existing.file_count = (existing.file_count || 1) + 1;
          existing.path = f.file_path || existing.path;
        } else {
          // 新建数据集
          datasetMap.set(dsName, {
            name: dsName,
            path: f.file_path || '',
            file_count: 1,
            created_at: f.created_at,
          });
        }
      }

      const result = Array.from(datasetMap.values());
      console.log('[getDatasets] Final datasets:', result);
      return result;
    } catch {
      return [];
    }
  },

  async uploadDataset(
    files: File[],
    datasetName: string,
    onProgress?: (progress: number) => void,
  ): Promise<{
    success: boolean;
    message: string;
    dataset_name: string;
    file_count: number;
    total_size: number;
    files: string[];
    job_id: string;
  }> {
    if (!files || files.length === 0) throw new Error('请选择文件');

    const totalFiles = files.length;
    const uploadedFiles: string[] = [];
    let lastFileId: string | null = null;
    let totalSize = 0;

    try {
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        console.log();

        const baseProgress = (i / totalFiles) * 95;
        const fileProgress = (p: number) => {
          const scaledP = baseProgress + (p / 100) * (95 / totalFiles);
          onProgress?.(Math.round(scaledP));
        };

        const result = await SimpleETMAPI.uploadDataset(file, datasetName, fileProgress);
        uploadedFiles.push(file.name);
        lastFileId = String(result.file_id);
        totalSize += file.size;
      }

      onProgress?.(100);

      return {
        success: true,
        message: `上传成功，共 ${uploadedFiles.length} 个文件`,
        dataset_name: datasetName,
        file_count: uploadedFiles.length,
        total_size: totalSize,
        files: uploadedFiles,
        job_id: lastFileId || "",
      }
    } catch (err) {
      throw err;
    }
  },

  async _uploadDatasetFormData(
    files: File[],
    datasetName: string,
    onProgress?: (progress: number) => void,
  ): Promise<any> {
    // 此函数已弃用，使用 SimpleETMAPI.uploadDataset 代替
    throw new Error('此上传方式已弃用，请使用 SimpleETMAPI.uploadDataset');
  },

  async deleteDataset(name: string): Promise<{ success: boolean; message: string }> {
    const token = localStorage.getItem('access_token');
    if (!token) return { success: false, message: '未登录' };

    try {
      const res = await fetch(`${API_BASE}/api/datasets/${encodeURIComponent(name)}`, {
        method: 'DELETE',
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: '删除失败' }));
        return { success: false, message: err.detail || '删除失败' };
      }
      return { success: true, message: '删除成功' };
    } catch (e) {
      return { success: false, message: e instanceof Error ? e.message : '网络错误' };
    }
  },

  // ========== 任务管理（job_id 体系） ==========

  async getTasks(params?: {
    status?: string; dataset?: string; limit?: number; offset?: number;
  }): Promise<TaskResponse[]> {
    const jobs = await BackendAPI.getTrainJobs();
    return jobs.filter(job => !params?.dataset || job.dataset_name === params.dataset).map(job => ({
      task_id: String(job.job_id), dataset: job.dataset_name, status: normalizeTaskStatus(job.status),
      progress: job.progress ?? 0, message: job.message, error_message: job.error_message, created_at: job.created_at,
    }));
  },

  // ========== 训练状态轮询 ==========

  async getTrainStatusByJobId(jobId: number): Promise<{status: string; job_id: number} | null> {
    try {
      const result = await BackendAPI.getTrainStatus(jobId);
      return result;
    } catch {
      return null;
    }
  },
  async getTrainJobs(): Promise<TrainStatusResponse[]> {
    try {
      const jobs = await BackendAPI.getTrainJobs();
      return jobs;
    } catch {
      return [];
    }
  },


  async getTaskStats(): Promise<{
    total: number; pending: number; running: number; completed: number; failed: number; cancelled: number;
  }> {
    // 后端暂无统计 API，暂时返回空数据
    return {
      total: 0,
      pending: 0,
      running: 0,
      completed: 0,
      failed: 0,
      cancelled: 0,
    };
  },

  async getTask(taskId: string): Promise<TaskResponse> {
    const jobId = parseInt(taskId, 10);
    try {
      const status = await BackendAPI.getTrainStatus(jobId);
      return {
        task_id: String(status.job_id),
        status: normalizeTaskStatus(status.status),
        progress: status.progress ?? (status.status === 'succeeded' ? 100 : 0),
        dataset: status.dataset_name,
        current_step: status.current_step,
        message: status.message || status.status,
        error_message: status.error_message,
        created_at: status.created_at,
      };
    } catch (e) {
      return {
        task_id: taskId,
        status: 'error',
        progress: 0,
        message: e instanceof Error ? e.message : '获取任务状态失败',
      };
    }
  },

  async getTaskLogs(taskId: string, _tail: number = 50): Promise<{
    task_id: string; status: string; logs: any[]; total_count: number;
  }> {
    // taskId here is the backend job_id; maps to /api/train/{job_id}/status
    try {
      const status = await BackendAPI.getTrainStatus(parseInt(taskId, 10));
      return {
        task_id: taskId,
        status: status.status ?? status.message ?? "unknown",
        logs: [],
        total_count: 0,
      };
    } catch {
      return { task_id: taskId, status: "unknown", logs: [], total_count: 0 };
    }
  },

  async createTask(request: CreateTaskRequest & { job_id?: string }): Promise<TaskResponse> {
    const userId = getCurrentUserId();

    // 使用后端适配器提交训练任务
    // job_id 是 file_id
    const fileId = parseInt(request.job_id || '0', 10);
    if (!fileId) {
      throw new Error('请先上传数据文件');
    }

    try {
      const result = await SimpleETMAPI.startTraining({
        file_id: fileId,
        dataset_name: request.dataset,
        model_type: request.models || 'theta',
        model_size: request.model_size || '0.6B',
        mode: request.mode || 'zero_shot',
        num_topics: request.num_topics,
        epochs: request.epochs,
        batch_size: request.batch_size,
        learning_rate: request.learning_rate,
        hidden_dim: request.hidden_dim,
        patience: request.patience,
        language: request.language,
        plot_language: request.plot_language,
        embedding_provider: request.embedding_provider,
        cloud_confirmed: request.cloud_confirmed,
        cloud_selection: request.cloud_selection,
        external_request_limit: request.external_request_limit,
        stopwords_id: request.stopwords_id,
        vocab_size: request.vocab_size,
        model_params: request.model_params,
        text_column: request.text_column,
        meta_columns: request.meta_columns,
        time_column: request.time_column,
        label_column: request.label_column,
      });

      return {
        task_id: String(result.job_id),
        status: normalizeTaskStatus(result.status),
        progress: 0,
        message: '训练任务已提交',
        dataset: request.dataset,
        mode: request.mode,
        num_topics: request.num_topics,
      };
    } catch (e) {
      const msg = e instanceof Error ? e.message : '请先上传数据文件，再创建训练任务';
      throw new Error(msg);
    }
  },

  async cancelTask(taskId: string): Promise<{ message: string }> {
    // 后端暂无取消任务 API
    return { message: '取消功能暂不支持' };
  },

  async pollTaskUntilDone(
    taskId: string,
    onProgress?: (task: TaskResponse) => void,
    interval = 5000,
    timeout = 3_600_000,
  ): Promise<TaskResponse> {
    const jobId = parseInt(taskId, 10);
    try {
      const finalStatus = await SimpleETMAPI.pollTraining(
        jobId,
        (status) => {
          onProgress?.({
            task_id: String(status.job_id),
            status: normalizeTaskStatus(status.status),
            progress: status.status === 'succeeded' ? 100 : status.status === 'failed' ? 0 : 50,
            message: status.message || status.status,
        error_message: status.error_message,
          });
        },
        interval,
        timeout,
      );

      return {
        task_id: String(finalStatus.job_id),
        status: normalizeTaskStatus(finalStatus.status),
        progress: finalStatus.status === 'succeeded' ? 100 : finalStatus.status === 'failed' ? 0 : 50,
        message: finalStatus.message || finalStatus.status,
      };
    } catch (e) {
      return {
        task_id: taskId,
        status: 'error',
        progress: 0,
        message: e instanceof Error ? e.message : '训练轮询失败',
      };
    }
  },

  // ========== 结果查询 ==========

  async getResults(): Promise<ResultInfo[]> {
    // 后端暂无 results API，暂时返回空数组
    return [];
  },

  async getResultInfo(dataset: string, mode: string): Promise<ResultInfo> {
    // 后端暂无此 API，暂时返回空数据
    return {
      dataset,
      mode,
      timestamp: new Date().toISOString(),
      path: '',
    };
  },

  /**
   * 获取训练结果文件下载链接
   * 后端暂无此 API，暂时返回空
   */
  async getJobResults(jobId: string): Promise<{ job_id: string; result_base: string; files: Record<string, string> }> {
    return { job_id: jobId, result_base: '', files: {} };
  },

  async getTopicWords(dataset: string, mode: string, topK = 10, jobId?: string): Promise<Record<string, string[]>> {
    // 如果提供了 job_id，尝试从后端获取
    if (jobId) {
      try {
        const job_id = parseInt(jobId, 10);
        const results = await SimpleETMAPI.getTrainingResults(job_id);
        const topWords: Record<string, string[]> = {};
        if (results.summary?.top_words) {
          results.summary.top_words.forEach((words, i) => {
            topWords[`topic_${i}`] = words.slice(0, topK);
          });
        }
        return topWords;
      } catch {
        return {};
      }
    }
    return {};
  },

  async getTopicProportions(dataset: string, mode: string, jobId?: string): Promise<{ topics: string[]; proportions: number[] }> {
    // 后端暂无此 API，暂时返回空数据
    return { topics: [], proportions: [] };
  },

  getTopicWordImportanceUrl(dataset: string, mode: string, topicIndex: number, modelName = 'theta'): string {
    const userId = getCurrentUserId();
    if (!userId) {
      return `${API_BASE}/api/results/${encodeURIComponent(dataset)}/${encodeURIComponent(mode)}/visualizations/topics/topic_${topicIndex}/word_importance.png`;
    }
    return getVisualizationAssetUrl(userId, dataset, modelName, `topic_${topicIndex}/word_importance.png`);
  },

  async getMetrics(dataset: string, mode: string, jobId?: string): Promise<MetricsResponse> {
    // 如果提供了 job_id，使用后端 API 获取
    if (jobId) {
      try {
        const job_id = parseInt(jobId, 10);
        const results = await SimpleETMAPI.getTrainingResults(job_id);
        return results.metrics || {};
      } catch {
        return {};
      }
    }
    // 否则尝试使用原有的 API（可能会失败）
    try {
      return await apiFetch(API_BASE, `/api/results/${dataset}/${mode}/metrics`);
    } catch {
      return {};
    }
  },

  async getDatasetPreview(dataset: string, jobId?: string): Promise<{ columns: string[]; rows: string[][] }> {
    return apiFetch(API_BASE, `/api/datasets/${encodeURIComponent(dataset)}/preview${jobId ? `?file_id=${encodeURIComponent(jobId)}` : ''}`);
  },

  async listVisualizations(dataset: string, mode: string): Promise<Array<{ name: string; path: string; type: string; size?: number }>> {
    // 后端暂无可视化列表 API，暂时返回空数组
    return [];
  },

  async getModelComparison(dataset: string): Promise<{
    dataset: string;
    rows: Array<Record<string, unknown>>;
    columns: Array<{ key: string; label: string; direction: string }>;
  }> {
    // 后端暂无模型对比 API，暂时返回空数据
    return { dataset, rows: [], columns: [] };
  },

  async exportResults(
    dataset: string,
    mode: string,
    types: string[] = ['metrics', 'topic_words', 'visualizations'],
  ): Promise<void> {
    // 后端暂无导出 API，抛出错误
    throw new Error('导出功能暂不支持');
  },

  // ========== 预处理 ==========

  async startPreprocessing(params: { dataset: string; text_column?: string; config?: any }): Promise<PreprocessingJob> {
    const userId = getCurrentUserId();
    return apiFetch(API_BASE, '/api/preprocessing/start', {
      method: 'POST',
      body: JSON.stringify({ ...(userId ? { user_id: userId } : {}), ...params }),
    });
  },

  async getPreprocessingJob(jobId: string): Promise<PreprocessingJob> {
    return apiFetch(API_BASE, `/api/preprocessing/${encodeURIComponent(jobId)}`);
  },

  async checkPreprocessingStatus(dataset: string): Promise<PreprocessingStatus> {
    try {
      return await apiFetch(API_BASE, `/api/preprocessing/check/${encodeURIComponent(dataset)}`);
    } catch (error) {
      if (error instanceof ApiError && error.status === 404) {
        return { has_bow: false, has_embeddings: false, ready_for_training: false, managed_by_training: true };
      }
      throw error;
    }
  },

  // ========== AI 对话 ==========

  async advisory(
    message: string,
    context?: Record<string, unknown>,
    charts?: AgentChartDataPayload[],
    purpose?: 'consultation' | 'chart-analysis',
    consultation?: { scope: string; id: string; requestId: string },
  ): Promise<{ message: string }> {
    const response = await apiFetch<{ ok: boolean; data: { message: string } }>('', '/api/v3/advisory', {
      method: 'POST',
      timeoutMs: 120_000,
      body: JSON.stringify({ message, context, charts, ...(purpose ? { purpose } : {}), ...(consultation ? { consultation } : {}) }),
    });
    return response.data;
  },

  async chat(
    message: string,
    context?: Record<string, unknown>,
    options?: { sessionId?: string; images?: AgentChatImagePayload[]; files?: AgentChatFilePayload[] },
  ): Promise<{ message: string; response?: string; action?: string; task_id?: string; data?: Record<string, unknown> }> {
    try {
      const raw = await apiFetch<any>(AGENT_BASE || API_BASE, '/api/agent/chat', {
        method: 'POST',
        body: JSON.stringify({
          message,
          session_id: options?.sessionId ?? context?.session_id ?? 'default',
          context: context || undefined,
          images: options?.images,
          files: options?.files,
        }),
      });
      return { message: raw.message, response: raw.message, action: undefined, data: undefined };
    } catch {
      try {
        return await apiFetch(API_BASE, '/api/chat', {
          method: 'POST',
          body: JSON.stringify({ message, context, images: options?.images, files: options?.files, session_id: options?.sessionId }),
        });
      } catch {
        return { message: '暂时无法连接 AI 服务，请稍后再试。' };
      }
    }
  },

  async *chatStream(
    message: string,
    sessionId = 'default',
    context?: Record<string, unknown>,
    images?: AgentChatImagePayload[],
    files?: AgentChatFilePayload[],
  ): AsyncGenerator<{ type: string; content: string }> {
    const token = typeof window !== 'undefined' ? localStorage.getItem('access_token') : null;
    const response = await fetch(`${AGENT_BASE}/api/agent/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify({ message, session_id: sessionId, context, images, files }),
    });

    if (!response.ok) throw new Error(`SSE 请求失败 (HTTP ${response.status})`);

    const reader = response.body?.getReader();
    if (!reader) throw new Error('无法读取响应流');

    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6).trim();
          if (data === '[DONE]') return;
          try {
            yield JSON.parse(data);
          } catch { /* skip malformed chunks */ }
        }
      }
    }
  },

  // ========== Agent 高级功能 ==========

  async getAgentTools(): Promise<{ tools: Array<{ name: string; description: string }> }> {
    return apiFetch(AGENT_BASE, '/api/agent/tools');
  },

  async clearAgentSession(sessionId = 'default'): Promise<void> {
    await apiFetch(AGENT_BASE, `/api/agent/sessions/${sessionId}`, { method: 'DELETE' });
  },

  async listAgentSessions(): Promise<{ sessions: string[] }> {
    return apiFetch(AGENT_BASE, '/api/agent/sessions');
  },

  // ========== 结果解读（Agent API） ==========

  async interpretMetrics(jobId: string, language = 'zh'): Promise<any> {
    return apiFetch(AGENT_BASE, '/api/interpret/metrics', {
      method: 'POST',
      body: JSON.stringify({ job_id: jobId, language }),
    });
  },

  async interpretTopics(jobId: string, language = 'zh', useLlm = true): Promise<any> {
    return apiFetch(AGENT_BASE, '/api/interpret/topics', {
      method: 'POST',
      body: JSON.stringify({ job_id: jobId, language, use_llm: useLlm }),
    });
  },

  async generateSummary(jobId: string, language = 'zh'): Promise<any> {
    return apiFetch(AGENT_BASE, '/api/interpret/summary', {
      method: 'POST',
      body: JSON.stringify({ job_id: jobId, language }),
    });
  },

  async analyzeChart(jobId: string, chartName: string, analysisType = 'general', language = 'zh', dataset?: string, chartUrl?: string): Promise<any> {
    return apiFetch(AGENT_BASE, '/api/vision/analyze-chart', {
      method: 'POST',
      body: JSON.stringify({ job_id: jobId, chart_name: chartName, analysis_type: analysisType, language, dataset, chart_url: chartUrl }),
    });
  },

  async getJobTopics(jobId: string): Promise<{ job_id: string; topics: any[] }> {
    // 后端暂无此 API，尝试从训练结果获取
    try {
      const job_id = parseInt(jobId, 10);
      const results = await SimpleETMAPI.getTrainingResults(job_id);
      return {
        job_id: jobId,
        topics: results.summary?.top_words?.map((words, i) => ({
          topic_id: i,
          words,
        })) || [],
      };
    } catch {
      return { job_id: jobId, topics: [] };
    }
  },

  async getJobCharts(jobId: string): Promise<{ job_id: string; charts: any; wordclouds: string[]; downloads: any }> {
    // 后端暂无此 API
    return { job_id: jobId, charts: {}, wordclouds: [], downloads: {} };
  },

  /**
   * 列出 OSS 上指定数据集的所有图表文件（png/jpg/pdf/html）。
   * 等同于：ossutil ls "oss://…/result/baseline/{dataset}/" -r | grep "\.png\|\.jpg\|\.pdf\|\.html"
   */
  async listOssChartFiles(dataset: string): Promise<{
    dataset: string;
    charts: Array<{ key: string; path: string; ext: string; size: number; url: string }>;
    total: number;
    note?: string;
  }> {
    try {
      return await apiFetch(API_BASE, `/api/data/oss-charts/${encodeURIComponent(dataset)}`);
    } catch {
      return { dataset, charts: [], total: 0, note: 'OSS 图表文件列举失败' };
    }
  },

  /**
   * 列出 OSS 上所有拥有可视化图表的数据集名称（用于选择器）。
   */
  async listOssDatasets(): Promise<{
    datasets: Array<{ name: string; chart_count: number }>;
    note?: string;
  }> {
    try {
      return await apiFetch(API_BASE, `/api/data/oss-datasets`);
    } catch {
      return { datasets: [], note: 'OSS 数据集列表获取失败' };
    }
  },

  getDownloadUrl(jobId: string, filename: string): string {
    return `${AGENT_BASE || API_BASE}/api/download/${jobId}/${filename}`;
  },

  // ========== 对话历史 ==========

  async saveConversationHistory(sessionId: string, messages: Array<{ role: string; content: string }>): Promise<any> {
    try {
      return await apiFetch(API_BASE, '/api/chat/history', {
        method: 'POST',
        body: JSON.stringify({ session_id: sessionId, messages }),
      });
    } catch { return { message: 'ok', session_id: sessionId, message_count: 0 }; }
  },

  async getConversationHistory(sessionId: string): Promise<any> {
    try {
      return await apiFetch(API_BASE, `/api/chat/history/${sessionId}`);
    } catch { return { session_id: sessionId, messages: [], count: 0 }; }
  },

  async clearConversationHistory(sessionId: string): Promise<any> {
    try {
      return await apiFetch(API_BASE, `/api/chat/history/${sessionId}`, { method: 'DELETE' });
    } catch { return { message: 'ok', session_id: sessionId }; }
  },

  // ========== 智能建议 ==========

  async getSuggestions(context?: Record<string, unknown>): Promise<any> {
    try {
      return await apiFetch(API_BASE, '/api/chat/suggestions', {
        method: 'POST',
        body: JSON.stringify(context || {}),
      });
    } catch {
      return {
        suggestions: [
          { text: '开始分析', action: 'start', description: '上传数据并运行分析流水线' },
          { text: '查看结果', action: 'results', description: '查看已有的分析结果' },
        ],
      };
    }
  },
};

// ==================== 辅助函数 ====================

// DLC 状态 → { message, progress } 映射（与阿里云控制台一致）
const DLC_STATUS_MAP: Record<string, { message: string; progress: number }> = {
  Creating:   { message: '任务创建中', progress: 5 },
  Created:    { message: '任务已创建，等待调度', progress: 8 },
  Queuing:    { message: '排队等待资源', progress: 10 },
  Waiting:    { message: '等待资源分配', progress: 12 },
  Scheduling: { message: '正在调度资源', progress: 15 },
  Preparing:  { message: '环境准备中', progress: 20 },
  Running:    { message: '训练运行中', progress: 50 },
  Stopping:   { message: '任务停止中', progress: 95 },
  Succeeded:  { message: '训练完成', progress: 100 },
  Failed:     { message: '训练失败', progress: 0 },
  Stopped:    { message: '训练已停止', progress: 0 },
};

function normalizeJob(raw: any): TaskResponse {
  const dlcStatus: string | undefined = raw.dlc_status;
  const dlcInfo = dlcStatus ? DLC_STATUS_MAP[dlcStatus] : undefined;

  let progress: number;
  if (raw.status === 'completed') progress = 100;
  else if (raw.status === 'error') progress = 0;
  else if (dlcInfo) progress = dlcInfo.progress;
  else progress = 50;

  // 计算已运行时长（仅 DLC 阶段）
  let elapsedSec = 0;
  if (raw.created_at && raw.status !== 'completed' && raw.status !== 'error') {
    elapsedSec = Math.floor((Date.now() - new Date(raw.created_at).getTime()) / 1000);
  }

  const dlcElapsed = elapsedSec > 0 ? `（已运行 ${elapsedSec} 秒）` : '';
  const message: string | undefined =
    raw.message ||
    (dlcInfo ? `${dlcInfo.message}${dlcElapsed}` : undefined) ||
    (dlcStatus ? `DLC: ${dlcStatus}${dlcElapsed}` : undefined) ||
    raw.error;

  return {
    task_id: raw.job_id ?? raw.task_id ?? '',
    status: raw.status ?? 'pending',
    progress,
    message,
    dataset: raw.dataset_name ?? raw.dataset ?? undefined,
    mode: raw.mode ?? undefined,
    num_topics: raw.num_topics ?? undefined,
    created_at: raw.created_at ?? undefined,
    completed_at: raw.completed_at ?? undefined,
    dlc_job_id: raw.dlc_job_id ?? undefined,
    dlc_status: dlcStatus,
    error_message: raw.error ?? undefined,
  };
}

export default ETMAgentAPI;
