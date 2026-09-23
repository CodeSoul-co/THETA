# THETA 本地工作台 API

本仓库统一维护本地版。旧托管平台、账号服务和 OSS 直传不再作为运行入口。

- 浏览器数据入口：`/api/backend`，仅转发到 `THETA_MANUAL_LOCAL_API_URL`（默认 `http://127.0.0.1:4321`）。
- 对话入口：`/api/v3`，仅转发到 `THETA_AGENT_API_URL`（默认 `http://127.0.0.1:4318`）。
- 使用仓库工作台启动器或桌面应用启动；代理要求前后端均在本机，拒绝跨站请求。桌面服务还要求每次启动生成的应用密钥。
- 不再使用 `NEXT_PUBLIC_API_URL`、旧托管后端地址、用户登录 Cookie 或远端后备地址。

## 手动数据流程

1. `POST /api/projects` 创建本地项目。
2. `POST /api/upload?filename=...&dataset_name=...` 上传原始文件字节，返回文件 `id`。
3. `GET /api/datasets/{dataset}/preview?file_id={id}` 返回 `columns` 和前五行 `rows`，用于文本列、时间列、标签列和元数据选择。
4. `GET /api/preprocessing/check/{dataset}` 返回由训练流程管理的预处理状态。
5. `POST /api/train/start` 提交本地训练；通过 `/api/train/{id}/status` 查询状态。

数据文件、项目和训练结果保存在本机。用户在设置中自行填写的云端 LLM / Embedding 地址与密钥属于独立的模型服务配置，不是远程工作台后端；安装包不包含用户密钥。
