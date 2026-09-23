# THETA Legacy API 文档（详细版）

## 1. 入口与链路

- 浏览器代理入口：`/Users/erwin/Downloads/codespace/-THETA-/frontend/app/api/backend/[...path]/route.ts`
- 默认转发目标：`/Users/erwin/Downloads/codespace/-THETA-/frontend/app/api/backend/[...path]/route.ts` 中 `THETA_LEGACY_API_URL`
  - 默认值：`https://theta-backend-nu.vercel.app`
  - 方法：`GET/POST/PUT/POST/DELETE/PATCH`
  - 支持本地 mock 认证（仅当 `THETA_LOCAL_AUTH_ENABLED=true` 或开发环境）
- 本地后端封装：
  - `API_ENDPOINTS`: `/Users/erwin/Downloads/codespace/-THETA-/frontend/lib/api/endpoints-config.ts`
  - 统一请求：`apiFetch`: `/Users/erwin/Downloads/codespace/-THETA-/frontend/lib/api/config.ts`
  - 认证相关：`/Users/erwin/Downloads/codespace/-THETA-/frontend/lib/api/auth.ts`
  - 业务封装：`/Users/erwin/Downloads/codespace/-THETA-/frontend/lib/api/backend.ts`
  - Agent 分支：`/Users/erwin/Downloads/codespace/-THETA-/frontend/lib/api/etm-agent.ts`

## 2. 通用约定

- 大多数接口以 REST 路径转发，不做字段改写。
- `apiFetch` 默认添加 `Content-Type: application/json` 与 `Authorization`。
- `401` 会清理本地 token 并触发跳转。
- 非 2xx 时抛出 `Error(detail)`，并非统一 `{ ok: false }` envelope。
- `/api/auth/login` 为表单提交（`application/x-www-form-urlencoded`），其余常见接口多为 JSON。

## 3. 主要接口清单（按模块）

### 3.1 认证

- `POST /api/auth/register` 用户注册
- `POST /api/auth/login` 用户登录（表单）
- `GET /api/auth/me` 获取当前用户
- 代理层本地 mock：`/api/auth/logout`、`/api/auth/verify`

### 3.2 文件与存储

- `POST /api/upload` 文件上传入口（主接口）
- `POST /api/upload/test` 测试上传
- `POST /api/upload/complete` 上传完成回调
- `GET /api/files` 文件列表
- `GET /api/oss/sts-token?dataset_name=...` 获取 OSS 临时凭证
- `GET /api/oss/sts-token?dataset_name=&filename=&content_type=` 可选扩展参数

### 3.3 预处理与训练

- `POST /api/preprocessing/start`
- `GET /api/preprocessing/check/{dataset}`
- `GET /api/preprocessing/{job_id}`
- `POST /api/train/start`
- `GET /api/train/{job_id}/status`
- `GET /api/train/{job_id}/metrics`
- `GET /api/train/{job_id}/summary`
- `GET /api/train/jobs`
- `POST /api/train/callback`
- `POST /api/train/{job_id}/cancel`（后端封装有显式使用）

### 3.4 结果、模型与可视化

- `GET /api/data/oss-datasets`
- `GET /api/results/{dataset}/models`
- `GET /api/results/{dataset}/topic-words`
- `GET /api/results/{dataset}/metrics`
- `GET /api/results/{dataset}/visualizations`
- `DELETE /api/datasets/{dataset}`

### 3.5 分析与交互

- `POST /api/agent/chat`
- `GET /api/chat/history/{session_id}`
- `GET /api/chat/suggestions`（部分分支使用）
- `POST /api/interpret/metrics`
- `POST /api/interpret/topics`
- `POST /api/interpret/summary`
- `POST /api/vision/analyze-chart`

### 3.6 系统

- `GET /health`
- `GET /docs`
- `GET /redoc`

## 4. 示例调用

### 4.1 登录（注意是 form body）

```bash
curl -X POST /api/backend/api/auth/login \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  -d 'username=xxx&password=yyy'
```

### 4.2 获取训练状态

```bash
curl -X GET /api/backend/api/train/123/status
```

### 4.3 上传完成回调

```bash
curl -X POST /api/backend/api/upload/complete \
  -H 'Content-Type: application/json' \
  -d '{"dataset_name":"demo.csv","filename":"demo.csv","oss_path":"abc/demo.csv"}'
```

## 5. 注意事项（你优化 v3 时避免误改 legacy）

1. 后端路径由前端代理层统一转发，路由文件里不做业务逻辑。
2. 若你只优化 v3，请优先改 `/app/api/v3` 和 `theta_project/theta-cli-agent/src/web-api/*`。
3. legacy 的鉴权与返回错误没有统一 envelope，不要在 v3 改动时回填到 legacy。
4. 任何 `API_BASE` 相关变更都要同步 `API_ENDPOINTS` 和 `buildUrl`。
