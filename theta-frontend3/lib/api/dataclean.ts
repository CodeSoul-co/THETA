/**
 * DataClean API 客户端
 * 用于与后端 DataClean API 服务通信
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_DATACLEAN_API_URL || 'http://localhost:8001';

export interface CleanTextRequest {
  text: string;
  language?: 'chinese' | 'english';
  operations?: string[];
}

export interface CleanTextResponse {
  cleaned_text: string;
  original_length: number;
  cleaned_length: number;
}

export interface ProcessFileResponse {
  task_id: string;
  status: string;
  message: string;
  file_count?: number;
}

export interface TaskStatusResponse {
  task_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  message: string;
  result_file?: string;
  error?: string;
}

export interface SupportedFormatsResponse {
  formats: string[];
  count: number;
}

/**
 * DataClean API 客户端类
 */
export class DataCleanAPI {
  /**
   * 健康检查
   */
  static async healthCheck() {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) {
      throw new Error('API 服务不可用');
    }
    return response.json();
  }

  /**
   * 获取支持的文件格式
   */
  static async getSupportedFormats(): Promise<SupportedFormatsResponse> {
    const response = await fetch(`${API_BASE_URL}/api/formats`);
    if (!response.ok) {
      throw new Error('获取支持格式失败');
    }
    return response.json();
  }

  /**
   * 清洗文本内容
   */
  static async cleanText(request: CleanTextRequest): Promise<CleanTextResponse> {
    const response = await fetch(`${API_BASE_URL}/api/clean/text`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: '清洗文本失败' }));
      throw new Error(error.detail || '清洗文本失败');
    }

    return response.json();
  }

  /**
   * 上传并处理单个文件
   */
  static async processFile(
    file: File,
    language: 'chinese' | 'english' = 'chinese',
    clean: boolean = true,
    operations?: string[]
  ): Promise<ProcessFileResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('language', language);
    formData.append('clean', clean.toString());
    if (operations && operations.length > 0) {
      formData.append('operations', operations.join(','));
    }

    const response = await fetch(`${API_BASE_URL}/api/upload/process`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: '文件处理失败' }));
      throw new Error(error.detail || '文件处理失败');
    }

    return response.json();
  }

  /**
   * 批量上传并处理文件
   */
  static async processBatchFiles(
    files: File[],
    language: 'chinese' | 'english' = 'chinese',
    clean: boolean = true,
    operations?: string[]
  ): Promise<ProcessFileResponse> {
    const formData = new FormData();
    files.forEach((file) => {
      formData.append('files', file);
    });
    formData.append('language', language);
    formData.append('clean', clean.toString());
    if (operations && operations.length > 0) {
      formData.append('operations', operations.join(','));
    }

    const response = await fetch(`${API_BASE_URL}/api/upload/batch`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: '批量处理失败' }));
      throw new Error(error.detail || '批量处理失败');
    }

    return response.json();
  }

  /**
   * 获取任务状态
   */
  static async getTaskStatus(taskId: string): Promise<TaskStatusResponse> {
    const response = await fetch(`${API_BASE_URL}/api/task/${taskId}`);
    if (!response.ok) {
      throw new Error('获取任务状态失败');
    }
    return response.json();
  }

  /**
   * 下载处理结果
   */
  static async downloadResult(taskId: string): Promise<Blob> {
    const response = await fetch(`${API_BASE_URL}/api/download/${taskId}`);
    if (!response.ok) {
      throw new Error('下载结果失败');
    }
    return response.blob();
  }

  /**
   * 下载处理结果并触发浏览器下载
   */
  static async downloadResultFile(taskId: string, filename: string = 'result.csv') {
    const blob = await this.downloadResult(taskId);
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
  }

  /**
   * 删除任务
   */
  static async deleteTask(taskId: string) {
    const response = await fetch(`${API_BASE_URL}/api/task/${taskId}`, {
      method: 'DELETE',
    });
    if (!response.ok) {
      throw new Error('删除任务失败');
    }
    return response.json();
  }
}
