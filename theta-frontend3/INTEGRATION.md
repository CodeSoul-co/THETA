# DataClean API 集成说明

## 概述

前端项目已集成 DataClean API 后端服务，可以在"数据处理"页面使用文件上传、文本清洗等功能。

## 配置

### 1. 环境变量配置

创建 `.env.local` 文件（如果不存在）：

```bash
# DataClean API 配置
NEXT_PUBLIC_DATACLEAN_API_URL=http://localhost:8001
```

如果后端部署在其他地址，请修改为实际地址：

```bash
NEXT_PUBLIC_DATACLEAN_API_URL=https://your-api-domain.com
```

### 2. 启动后端服务

在启动前端之前，确保 DataClean API 后端服务正在运行：

```bash
cd ETM/dataclean
python3 api.py
# 或
./start_api.sh
```

后端服务默认运行在 `http://localhost:8001`

## 功能说明

### 数据处理页面

访问路径：点击侧边栏的"数据处理"

功能包括：
- **文件上传**：支持拖拽或点击选择文件
- **批量处理**：可以同时上传多个文件
- **语言选择**：支持中文和英文处理
- **文本清洗**：可配置多种清洗操作
- **进度跟踪**：实时显示处理进度
- **结果下载**：处理完成后可下载 CSV 结果

### 支持的文件格式

- `.txt` - 文本文件
- `.pdf` - PDF 文件
- `.csv` - CSV 文件
- `.json` - JSON 文件
- `.xml` - XML 文件
- `.html` - HTML 文件
- `.docx` - Word 文档

### 清洗操作选项

- **移除URL**：移除文本中的网址链接
- **移除HTML标签**：清理 HTML 标记
- **移除标点符号**：删除标点符号
- **移除停用词**：过滤常见停用词
- **规范化空白字符**：统一空白字符格式
- **移除数字**：删除数字内容
- **移除特殊字符**：清理特殊符号

## API 客户端

API 客户端位于 `lib/api/dataclean.ts`，提供以下方法：

```typescript
import { DataCleanAPI } from '@/lib/api/dataclean'

// 健康检查
await DataCleanAPI.healthCheck()

// 获取支持格式
await DataCleanAPI.getSupportedFormats()

// 清洗文本
await DataCleanAPI.cleanText({
  text: '文本内容',
  language: 'chinese',
  operations: ['remove_urls', 'remove_stopwords']
})

// 处理文件
await DataCleanAPI.processFile(file, 'chinese', true, ['remove_urls'])

// 批量处理
await DataCleanAPI.processBatchFiles(files, 'chinese', true)

// 下载结果
await DataCleanAPI.downloadResultFile(taskId, 'result.csv')
```

## 使用示例

### 在组件中使用

```tsx
import { DataCleanAPI } from '@/lib/api/dataclean'

function MyComponent() {
  const handleUpload = async (file: File) => {
    try {
      const result = await DataCleanAPI.processFile(
        file,
        'chinese',
        true,
        ['remove_urls', 'remove_stopwords']
      )
      console.log('处理完成:', result)
    } catch (error) {
      console.error('处理失败:', error)
    }
  }

  return (
    <input
      type="file"
      onChange={(e) => {
        const file = e.target.files?.[0]
        if (file) handleUpload(file)
      }}
    />
  )
}
```

## 故障排查

### 1. API 连接失败

**问题**：无法连接到后端服务

**解决方案**：
- 检查后端服务是否运行：`curl http://localhost:8001/health`
- 确认 `.env.local` 中的 API URL 配置正确
- 检查防火墙设置

### 2. CORS 错误

**问题**：浏览器控制台显示 CORS 错误

**解决方案**：
- 确保后端 API 已配置 CORS（已在 `api.py` 中配置）
- 检查 API URL 是否正确

### 3. 文件上传失败

**问题**：文件上传后处理失败

**解决方案**：
- 检查文件格式是否支持
- 查看浏览器控制台的错误信息
- 检查后端服务日志

### 4. 下载失败

**问题**：无法下载处理结果

**解决方案**：
- 确认任务状态为 `completed`
- 检查任务 ID 是否正确
- 查看后端服务日志

## 开发建议

1. **错误处理**：所有 API 调用都应包含错误处理
2. **加载状态**：在处理文件时显示加载状态
3. **用户反馈**：使用 Toast 或 Alert 组件提示用户操作结果
4. **文件验证**：在上传前验证文件格式和大小

## 相关文档

- [DataClean API 使用文档](../../ETM/dataclean/API_USAGE.md)
- [后端设置文档](../../ETM/dataclean/BACKEND_SETUP.md)
- [集成指南](../../ETM/dataclean/INTEGRATION_GUIDE.md)
