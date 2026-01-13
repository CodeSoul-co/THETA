# DataClean API 使用文档

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
pip install fastapi uvicorn python-multipart
```

### 2. 启动 API 服务

```bash
# 方式1: 直接运行
python api.py

# 方式2: 使用 uvicorn
uvicorn api:app --host 0.0.0.0 --port 8001 --reload
```

服务将在 `http://localhost:8001` 启动

### 3. 访问 API 文档

- Swagger UI: `http://localhost:8001/docs`
- ReDoc: `http://localhost:8001/redoc`

## API 端点

### 1. 健康检查

```http
GET /health
```

**响应:**
```json
{
  "status": "ok",
  "version": "1.0.0",
  "service": "DataClean API"
}
```

### 2. 获取支持的文件格式

```http
GET /api/formats
```

**响应:**
```json
{
  "formats": [".txt", ".pdf", ".csv", ".json", ".xml", ".html", ".docx"],
  "count": 7
}
```

### 3. 文本清洗

```http
POST /api/clean/text
Content-Type: application/json

{
  "text": "这是测试文本 https://example.com",
  "language": "chinese",
  "operations": ["remove_urls", "remove_stopwords"]
}
```

**响应:**
```json
{
  "cleaned_text": "测试文本",
  "original_length": 20,
  "cleaned_length": 4
}
```

### 4. 单文件上传和处理

```http
POST /api/upload/process
Content-Type: multipart/form-data

file: [文件]
language: chinese
clean: true
operations: remove_urls,remove_html_tags
```

**响应:**
```json
{
  "task_id": "uuid-string",
  "status": "completed",
  "message": "文件处理完成",
  "file_count": 1
}
```

### 5. 批量文件上传和处理

```http
POST /api/upload/batch
Content-Type: multipart/form-data

files: [文件1, 文件2, ...]
language: chinese
clean: true
operations: remove_urls,remove_stopwords
```

**响应:**
```json
{
  "task_id": "uuid-string",
  "status": "completed",
  "message": "成功处理 3 个文件",
  "file_count": 3
}
```

### 6. 获取任务状态

```http
GET /api/task/{task_id}
```

**响应:**
```json
{
  "task_id": "uuid-string",
  "status": "completed",
  "progress": 100.0,
  "message": "处理完成",
  "result_file": "/api/download/uuid-string",
  "error": null
}
```

### 7. 下载处理结果

```http
GET /api/download/{task_id}
```

返回 CSV 文件下载

## 前端调用示例

### JavaScript/TypeScript (Fetch API)

```javascript
// 1. 上传并处理单个文件
async function processFile(file, language = 'chinese', clean = true) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('language', language);
  formData.append('clean', clean);
  formData.append('operations', 'remove_urls,remove_html_tags,remove_stopwords');
  
  const response = await fetch('http://localhost:8001/api/upload/process', {
    method: 'POST',
    body: formData
  });
  
  const result = await response.json();
  return result;
}

// 2. 批量上传文件
async function processBatchFiles(files, language = 'chinese') {
  const formData = new FormData();
  files.forEach(file => {
    formData.append('files', file);
  });
  formData.append('language', language);
  formData.append('clean', 'true');
  
  const response = await fetch('http://localhost:8001/api/upload/batch', {
    method: 'POST',
    body: formData
  });
  
  const result = await response.json();
  return result;
}

// 3. 下载处理结果
async function downloadResult(taskId) {
  const response = await fetch(`http://localhost:8001/api/download/${taskId}`);
  const blob = await response.blob();
  
  // 创建下载链接
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'result.csv';
  document.body.appendChild(a);
  a.click();
  window.URL.revokeObjectURL(url);
  document.body.removeChild(a);
}

// 4. 文本清洗
async function cleanText(text, language = 'chinese') {
  const response = await fetch('http://localhost:8001/api/clean/text', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      text: text,
      language: language,
      operations: ['remove_urls', 'remove_stopwords', 'normalize_whitespace']
    })
  });
  
  const result = await response.json();
  return result.cleaned_text;
}
```

### React 组件示例

```tsx
import React, { useState } from 'react';

function DataCleanUpload() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  
  const handleUpload = async () => {
    if (!file) return;
    
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('language', 'chinese');
      formData.append('clean', 'true');
      
      const response = await fetch('http://localhost:8001/api/upload/process', {
        method: 'POST',
        body: formData
      });
      
      const data = await response.json();
      
      if (data.status === 'completed') {
        // 下载结果
        const downloadUrl = `http://localhost:8001/api/download/${data.task_id}`;
        window.open(downloadUrl, '_blank');
        setResult('处理完成！');
      }
    } catch (error) {
      console.error('处理失败:', error);
      setResult('处理失败');
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div>
      <input
        type="file"
        onChange={(e) => setFile(e.target.files?.[0] || null)}
        accept=".txt,.pdf,.csv,.json,.xml,.html,.docx"
      />
      <button onClick={handleUpload} disabled={loading || !file}>
        {loading ? '处理中...' : '上传并处理'}
      </button>
      {result && <p>{result}</p>}
    </div>
  );
}
```

### Vue 组件示例

```vue
<template>
  <div>
    <input
      type="file"
      @change="handleFileChange"
      accept=".txt,.pdf,.csv,.json,.xml,.html,.docx"
    />
    <button @click="handleUpload" :disabled="loading || !file">
      {{ loading ? '处理中...' : '上传并处理' }}
    </button>
    <p v-if="result">{{ result }}</p>
  </div>
</template>

<script setup>
import { ref } from 'vue';

const file = ref(null);
const loading = ref(false);
const result = ref(null);

const handleFileChange = (e) => {
  file.value = e.target.files[0];
};

const handleUpload = async () => {
  if (!file.value) return;
  
  loading.value = true;
  try {
    const formData = new FormData();
    formData.append('file', file.value);
    formData.append('language', 'chinese');
    formData.append('clean', 'true');
    
    const response = await fetch('http://localhost:8001/api/upload/process', {
      method: 'POST',
      body: formData
    });
    
    const data = await response.json();
    
    if (data.status === 'completed') {
      const downloadUrl = `http://localhost:8001/api/download/${data.task_id}`;
      window.open(downloadUrl, '_blank');
      result.value = '处理完成！';
    }
  } catch (error) {
    console.error('处理失败:', error);
    result.value = '处理失败';
  } finally {
    loading.value = false;
  }
};
</script>
```

## 清洗操作选项

可用的清洗操作：
- `remove_urls`: 移除URL
- `remove_html_tags`: 移除HTML标签
- `remove_punctuation`: 移除标点符号
- `remove_stopwords`: 移除停用词
- `normalize_whitespace`: 规范化空白字符
- `remove_numbers`: 移除数字
- `remove_special_chars`: 移除特殊字符

## 错误处理

所有 API 端点可能返回以下错误：

```json
{
  "detail": "错误信息"
}
```

常见错误码：
- `400`: 请求参数错误
- `404`: 资源不存在
- `500`: 服务器内部错误

## 注意事项

1. **文件大小限制**: 默认没有限制，生产环境建议添加文件大小限制
2. **临时文件清理**: 临时文件存储在系统临时目录，建议定期清理
3. **CORS 配置**: 生产环境应限制允许的源域名
4. **并发处理**: 当前实现是同步处理，大文件可能需要较长时间
5. **任务存储**: 当前使用内存存储任务，生产环境应使用 Redis 或数据库

## 生产环境部署建议

1. 使用 Gunicorn + Uvicorn workers
2. 配置 Nginx 反向代理
3. 使用 Redis 存储任务状态
4. 添加文件大小和类型验证
5. 实现文件清理定时任务
6. 添加认证和授权
7. 配置日志记录
8. 添加监控和告警
