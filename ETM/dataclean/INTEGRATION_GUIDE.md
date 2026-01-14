# DataClean API 集成指南

## 概述

DataClean API 提供了完整的文本文件清洗和转换功能，可以轻松集成到前端应用中。

## 快速集成步骤

### 1. 启动后端服务

```bash
cd ETM/dataclean
python3 api.py
# 或使用启动脚本
./start_api.sh
```

服务将在 `http://localhost:8001` 启动

### 2. 在前端项目中添加 API 客户端

#### 创建 API 客户端文件

```typescript
// src/api/dataclean.ts
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

export class DataCleanAPI {
  // 健康检查
  static async healthCheck() {
    const response = await fetch(`${API_BASE_URL}/health`);
    return response.json();
  }

  // 获取支持的文件格式
  static async getSupportedFormats() {
    const response = await fetch(`${API_BASE_URL}/api/formats`);
    return response.json();
  }

  // 清洗文本
  static async cleanText(request: CleanTextRequest): Promise<CleanTextResponse> {
    const response = await fetch(`${API_BASE_URL}/api/clean/text`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });
    return response.json();
  }

  // 上传并处理单个文件
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
    if (operations) {
      formData.append('operations', operations.join(','));
    }

    const response = await fetch(`${API_BASE_URL}/api/upload/process`, {
      method: 'POST',
      body: formData,
    });
    return response.json();
  }

  // 批量上传文件
  static async processBatchFiles(
    files: File[],
    language: 'chinese' | 'english' = 'chinese',
    clean: boolean = true,
    operations?: string[]
  ): Promise<ProcessFileResponse> {
    const formData = new FormData();
    files.forEach(file => {
      formData.append('files', file);
    });
    formData.append('language', language);
    formData.append('clean', clean.toString());
    if (operations) {
      formData.append('operations', operations.join(','));
    }

    const response = await fetch(`${API_BASE_URL}/api/upload/batch`, {
      method: 'POST',
      body: formData,
    });
    return response.json();
  }

  // 下载处理结果
  static async downloadResult(taskId: string): Promise<Blob> {
    const response = await fetch(`${API_BASE_URL}/api/download/${taskId}`);
    return response.blob();
  }

  // 获取任务状态
  static async getTaskStatus(taskId: string) {
    const response = await fetch(`${API_BASE_URL}/api/task/${taskId}`);
    return response.json();
  }
}
```

### 3. 在 React/Next.js 中使用

#### 创建文件上传组件

```tsx
// components/DataCleanUpload.tsx
'use client';

import { useState } from 'react';
import { DataCleanAPI } from '@/api/dataclean';

export default function DataCleanUpload() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await DataCleanAPI.processFile(
        file,
        'chinese',
        true,
        ['remove_urls', 'remove_html_tags', 'remove_stopwords']
      );

      if (response.status === 'completed') {
        // 下载结果
        const blob = await DataCleanAPI.downloadResult(response.task_id);
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'cleaned_data.csv';
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);

        setResult(`处理完成！已处理 ${response.file_count} 个文件`);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '处理失败');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 max-w-md mx-auto">
      <h2 className="text-2xl font-bold mb-4">数据清洗工具</h2>
      
      <div className="mb-4">
        <label className="block text-sm font-medium mb-2">
          选择文件
        </label>
        <input
          type="file"
          onChange={handleFileChange}
          accept=".txt,.pdf,.csv,.json,.xml,.html,.docx"
          className="block w-full text-sm text-gray-500
            file:mr-4 file:py-2 file:px-4
            file:rounded-full file:border-0
            file:text-sm file:font-semibold
            file:bg-blue-50 file:text-blue-700
            hover:file:bg-blue-100"
        />
      </div>

      <button
        onClick={handleUpload}
        disabled={loading || !file}
        className="w-full bg-blue-600 text-white py-2 px-4 rounded
          hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
      >
        {loading ? '处理中...' : '上传并处理'}
      </button>

      {result && (
        <div className="mt-4 p-3 bg-green-100 text-green-800 rounded">
          {result}
        </div>
      )}

      {error && (
        <div className="mt-4 p-3 bg-red-100 text-red-800 rounded">
          错误: {error}
        </div>
      )}
    </div>
  );
}
```

#### 创建文本清洗组件

```tsx
// components/TextCleaner.tsx
'use client';

import { useState } from 'react';
import { DataCleanAPI } from '@/api/dataclean';

export default function TextCleaner() {
  const [text, setText] = useState('');
  const [cleanedText, setCleanedText] = useState('');
  const [loading, setLoading] = useState(false);

  const handleClean = async () => {
    if (!text.trim()) return;

    setLoading(true);
    try {
      const response = await DataCleanAPI.cleanText({
        text,
        language: 'chinese',
        operations: ['remove_urls', 'remove_stopwords', 'normalize_whitespace']
      });
      setCleanedText(response.cleaned_text);
    } catch (err) {
      console.error('清洗失败:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <h2 className="text-2xl font-bold mb-4">文本清洗</h2>
      
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium mb-2">
            原始文本
          </label>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={10}
            className="w-full p-3 border rounded"
            placeholder="输入要清洗的文本..."
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium mb-2">
            清洗后文本
          </label>
          <textarea
            value={cleanedText}
            readOnly
            rows={10}
            className="w-full p-3 border rounded bg-gray-50"
            placeholder="清洗后的文本将显示在这里..."
          />
        </div>
      </div>

      <button
        onClick={handleClean}
        disabled={loading || !text.trim()}
        className="mt-4 bg-blue-600 text-white py-2 px-4 rounded
          hover:bg-blue-700 disabled:bg-gray-400"
      >
        {loading ? '清洗中...' : '清洗文本'}
      </button>
    </div>
  );
}
```

### 4. 环境变量配置

在 `.env.local` 或 `.env` 文件中添加：

```env
NEXT_PUBLIC_DATACLEAN_API_URL=http://localhost:8001
```

### 5. 添加到现有 API 路由（可选）

如果项目已有 API 路由，可以添加代理：

```typescript
// pages/api/dataclean/[...path].ts (Next.js Pages Router)
// 或 app/api/dataclean/[...path]/route.ts (Next.js App Router)

import { NextRequest, NextResponse } from 'next/server';

const DATACLEAN_API_URL = process.env.DATACLEAN_API_URL || 'http://localhost:8001';

export async function POST(
  request: NextRequest,
  { params }: { params: { path: string[] } }
) {
  const path = params.path.join('/');
  const body = await request.text();
  
  const response = await fetch(`${DATACLEAN_API_URL}/api/${path}`, {
    method: 'POST',
    headers: {
      'Content-Type': request.headers.get('Content-Type') || 'application/json',
    },
    body,
  });

  const data = await response.json();
  return NextResponse.json(data);
}
```

## 与现有 ETM Agent API 集成

如果项目已有 `ETM/agent/api/app.py`，可以将 DataClean API 合并到同一个服务中：

```python
# 在 ETM/agent/api/app.py 中添加

from fastapi import APIRouter
from ETM.dataclean.api import app as dataclean_app

# 创建子路由
dataclean_router = APIRouter(prefix="/dataclean")

# 将 dataclean 的路由添加到主应用
app.include_router(dataclean_router)

# 或者直接挂载整个应用
app.mount("/dataclean", dataclean_app)
```

## 部署建议

### 开发环境
- 直接运行 `python3 api.py`
- 使用 `--reload` 参数支持热重载

### 生产环境
- 使用 Gunicorn + Uvicorn workers
- 配置 Nginx 反向代理
- 使用环境变量管理配置
- 添加日志和监控

```bash
# 使用 Gunicorn
gunicorn api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8001
```

## 常见问题

### 1. CORS 错误
确保 API 服务已配置 CORS 中间件（已包含在代码中）

### 2. 文件大小限制
默认没有限制，生产环境建议添加：
```python
from fastapi import Request
from fastapi.exceptions import RequestValidationError

@app.middleware("http")
async def check_file_size(request: Request, call_next):
    # 添加文件大小检查
    pass
```

### 3. 超时问题
大文件处理可能需要较长时间，考虑：
- 使用异步任务队列（Celery）
- 实现进度回调
- 增加超时时间

## 下一步

1. 根据前端需求调整 API 响应格式
2. 添加用户认证和授权
3. 实现文件存储和清理机制
4. 添加更多清洗选项
5. 优化大文件处理性能
