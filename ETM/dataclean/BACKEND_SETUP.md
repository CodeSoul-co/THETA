# DataClean 后端 API 设置完成

## 📁 创建的文件

1. **`api.py`** - FastAPI 后端服务主文件
   - 提供 RESTful API 接口
   - 支持文件上传、文本清洗、批量处理等功能

2. **`API_USAGE.md`** - API 使用文档
   - 详细的 API 端点说明
   - 前端调用示例（JavaScript/React/Vue）

3. **`INTEGRATION_GUIDE.md`** - 集成指南
   - 如何集成到前端项目
   - React/Next.js 组件示例
   - 与现有 API 服务集成方法

4. **`start_api.sh`** - 启动脚本
   - 快速启动 API 服务

5. **`test_api.py`** - API 测试脚本
   - 用于测试 API 功能

## 🚀 快速开始

### 1. 安装依赖

```bash
cd ETM/dataclean
pip3 install -r requirements.txt
```

### 2. 启动服务

```bash
# 方式1: 使用启动脚本
./start_api.sh

# 方式2: 直接运行
python3 api.py

# 方式3: 使用 uvicorn（支持热重载）
uvicorn api:app --host 0.0.0.0 --port 8001 --reload
```

### 3. 访问 API 文档

- Swagger UI: http://localhost:8001/docs
- ReDoc: http://localhost:8001/redoc

### 4. 测试 API

```bash
python3 test_api.py
```

## 📡 API 端点列表

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/api/formats` | GET | 获取支持的文件格式 |
| `/api/clean/text` | POST | 清洗文本内容 |
| `/api/upload/process` | POST | 上传并处理单个文件 |
| `/api/upload/batch` | POST | 批量上传并处理文件 |
| `/api/task/{task_id}` | GET | 获取任务状态 |
| `/api/download/{task_id}` | GET | 下载处理结果 |

## 💻 前端集成示例

### 最简单的使用方式

```javascript
// 1. 上传文件并处理
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('language', 'chinese');
formData.append('clean', 'true');

const response = await fetch('http://localhost:8001/api/upload/process', {
  method: 'POST',
  body: formData
});

const result = await response.json();

// 2. 下载结果
if (result.status === 'completed') {
  window.open(`http://localhost:8001/api/download/${result.task_id}`, '_blank');
}
```

### React Hook 示例

```tsx
import { useState } from 'react';

function useDataClean() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const processFile = async (file: File) => {
    setLoading(true);
    setError(null);
    
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('language', 'chinese');
      formData.append('clean', 'true');
      
      const response = await fetch('http://localhost:8001/api/upload/process', {
        method: 'POST',
        body: formData
      });
      
      const result = await response.json();
      
      if (result.status === 'completed') {
        // 下载结果
        window.open(`http://localhost:8001/api/download/${result.task_id}`, '_blank');
        return result;
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return { processFile, loading, error };
}
```

## 🔧 配置选项

### 环境变量

```bash
# 设置 API 端口（默认 8001）
export PORT=8001

# 启动服务
python3 api.py
```

### CORS 配置

生产环境建议修改 `api.py` 中的 CORS 设置：

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend-domain.com"],  # 限制域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 📝 下一步

1. **根据前端需求调整**
   - 修改 API 响应格式
   - 添加更多清洗选项
   - 优化文件处理流程

2. **生产环境优化**
   - 添加文件大小限制
   - 实现异步任务处理
   - 添加用户认证
   - 配置日志和监控

3. **与现有服务集成**
   - 合并到 `ETM/agent/api/app.py`
   - 统一 API 路由
   - 共享认证和中间件

## 📚 相关文档

- `API_USAGE.md` - 详细的 API 使用文档
- `INTEGRATION_GUIDE.md` - 前端集成指南
- `README.md` - 工具功能说明

## ⚠️ 注意事项

1. **临时文件**: 处理后的文件存储在系统临时目录，建议定期清理
2. **并发处理**: 当前实现是同步处理，大文件可能需要较长时间
3. **任务存储**: 使用内存存储任务，重启服务会丢失
4. **安全性**: 生产环境需要添加文件类型验证、大小限制等

## 🐛 问题排查

### API 无法启动
- 检查端口是否被占用
- 确认所有依赖已安装
- 查看错误日志

### 文件上传失败
- 检查文件格式是否支持
- 确认文件大小在合理范围内
- 查看服务器日志

### CORS 错误
- 确认 API 服务 CORS 配置正确
- 检查前端请求的 Origin

## 📞 支持

如有问题，请查看：
1. API 文档: http://localhost:8001/docs
2. 测试脚本: `test_api.py`
3. 使用文档: `API_USAGE.md`
