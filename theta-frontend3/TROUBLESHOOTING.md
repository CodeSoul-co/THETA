# 故障排除指南 (Troubleshooting Guide)

## 控制台错误

### Chrome 扩展错误 (可忽略)

**错误信息:**
```
GET chrome-extension://pejdijmoenmkgeppbflobdenhhabjlaj/utils.js net::ERR_FILE_NOT_FOUND
```

**原因:**
这些错误来自浏览器扩展（如密码管理器、自动填充工具等），它们尝试注入脚本到页面中。这不是应用程序的问题。

**解决方案:**
1. **可以忽略** - 这些错误不影响应用功能
2. **禁用相关扩展** - 如果干扰开发，可以临时禁用密码管理器等扩展
3. **使用隐私模式** - 在隐私模式下测试，大部分扩展会被禁用

### WebSocket 连接错误

**错误信息:**
```
ETM WebSocket error: {...}
WebSocket connection failed. Will retry in 3 seconds...
```

**原因:**
后端 WebSocket 服务未启动或无法连接。

**解决方案:**
1. 确保后端服务正在运行 (`http://localhost:8000`)
2. 检查后端 WebSocket 端点是否正常 (`/api/ws`)
3. WebSocket 会自动重连，无需手动操作

### 模型文件未找到错误

**错误信息:**
```
Embedding model not found at: /Users/qidu/Documents/code/THETA/qwen3_embedding_0.6B
```

**原因:**
Qwen 嵌入模型文件未下载到指定位置。

**解决方案:**
1. 模型路径在 `langgraph_agent/backend/app/core/config.py` 中配置为：
   ```python
   QWEN_MODEL_PATH = BASE_DIR / "qwen3_embedding_0.6B"
   ```
   
2. **下载模型:**
   - 从 ModelScope 或 Hugging Face 下载 `qwen3_embedding_0.6B` 模型
   - 将模型文件夹放到项目根目录 `/Users/qidu/Documents/code/THETA/qwen3_embedding_0.6B`
   
3. **或修改配置:**
   - 如果模型在其他位置，可以通过环境变量 `QWEN_MODEL_PATH` 指定路径

## 登录问题

### 登录后无法跳转

**检查项:**
1. 确认后端认证服务正在运行
2. 检查浏览器控制台的网络请求，查看 `/api/auth/login-json` 是否成功
3. 检查 `localStorage` 中是否有 `access_token`

### 登录状态丢失

**原因:**
- Token 过期（默认 30 天）
- 浏览器清除缓存或 localStorage

**解决方案:**
1. 重新登录
2. 使用"记住我"功能延长登录状态

### 后端认证 API 404

**检查项:**
1. 确认后端服务已启动: `http://localhost:8000`
2. 检查路由注册: `/api/auth/login-json`, `/api/auth/register`
3. 确认 `email-validator` 依赖已安装

## 开发环境

### HMR (Hot Module Replacement) 相关问题

**信息:**
```
[HMR] connected
```

这是正常的开发模式信息，表示热更新已连接。

### Vercel Analytics 调试信息

**信息:**
```
[Vercel Web Analytics] Debug mode is enabled
```

这是正常的开发模式信息，在生产环境中会自动禁用。

## 获取帮助

如果遇到其他问题：
1. 查看浏览器控制台的完整错误信息
2. 检查后端日志 (`langgraph_agent/backend/`)
3. 确认所有依赖已正确安装
4. 检查环境变量配置