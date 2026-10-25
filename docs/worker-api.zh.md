# 远程训练 Worker 接入

中文 | [English](worker-api.md)

**云计算资源即将推出。本版本预留配置入口与兼容接口约定。**

Agent、对话历史、项目管理、数据预览与分析报告撰写保留在用户本机。默认使用内置的本地 Worker；在“设置 → 计算资源”填写兼容服务的地址、密钥、币种与每个模型实验的费用上限后，对话和手动模式都可提交远程训练。本仓库提供客户端接入能力，不包含已部署的商业算力服务。

点击“完成”即可保存。密钥在本机加密，安装包不包含用户配置。费用上限为 0 时只允许免费任务；多模型、多主题数组合按实验数累计。训练配置页会明确确认数据上传与费用上限。仅上传本次选定文件及独立训练、验证、测试集；不转发本机的模型密钥或对话历史。远程所需的权重、嵌入服务和对应凭据由计算服务配置。

项目标识在上传前按本机工作区生成独立命名空间，因此不同电脑的同名手动项目仍然隔离。

## 接口约定

控制接口使用 `Authorization: Bearer <key>`。正式服务使用 HTTPS，本机调试允许回环 HTTP；不跟随重定向。响应为直接 JSON 对象，不套 `ok/data`。错误返回 HTTP 错误状态及 `{"error":{"message":"具体原因"}}`。

| 方法和路径 | 请求与响应 |
| --- | --- |
| `GET /v1/capabilities` | `{protocol:"theta-worker-v1",models:["lda","bertopic"]}` |
| `POST /v1/preview` | `{dataset,plan}` → `{readiness:{ready,issues},rowCount,quote:{id,maxCost,currency,expiresAt}}`；只检查配置和报价，不上传数据、不训练、不产生模型调用费用 |
| `POST /v1/datasets` | 表单字段 `file`、`projectId`、`sha256` → `{datasetId,projectId,sha256}`；服务端须核验实际文件内容 |
| `POST /v1/jobs` | `{clientJobId,idempotencyKey,projectId,dataset:{datasetId,sha256},plan,quoteId,maxCost,currency}` → `{id,projectId}`；持久化入队后立即响应 |
| `GET /v1/projects/:projectId/jobs/by-client-id/:clientJobId` | 根据客户端标识核对网络中断时的提交结果 |
| `GET /v1/projects/:projectId/jobs/:jobId` | 状态、阶段、百分比、消息、错误、阶段历史、训练遥测及诊断 |
| `POST /v1/projects/:projectId/jobs/:jobId/cancel` | 幂等取消本任务 |
| `GET /v1/projects/:projectId/jobs/:jobId/results` | 完整结果清单 |

预览中的数据描述只含标识、文件名、哈希、大小，不含本机路径。`plan` 使用仓库 `TrainingPlan` 结构，保留模型参数、列选择、设备、时限和 `dataSplit`。比例划分保留比例、随机或顺序方式及种子；独立上传保留各角色及其列选择，提交时将本机数据描述替换为远端 `datasetId` 与哈希。本机模型路径和密钥参数不得传到服务端。

服务端必须在每次请求中校验账号与项目、数据、任务的归属，不能仅凭客户端传来的项目标识授权。幂等键须按账号与项目隔离，先持久化再计费或入队：相同内容重试返回原任务，冲突内容拒绝。服务端必须兑现划分方式、时限、嵌入请求上限和报价上限；报价过期后重新确认，不静默加价。已经提交的任务绑定原服务与凭据，切回本地不会改变旧任务的查询地址。

状态使用 `queued`、`running`、`completed`、`failed`、`cancelled`。训练遥测遵循 `agent/src/domain/research.ts` 的 `JobTelemetry`，支持轮次、批次和嵌入进度。错误应包含模型、阶段、原因及处理建议；单个模型失败不影响其他模型已有结果。

## 结果完整回传

清单使用 `schema:"theta-results-v1"`、`status:"ready"`，包含 `projectId`、远端 `jobId`、`files` 以及原生报告的 `evidence`、`sourceData`、`quality`、`summary`、`diagnostics`。

每个文件记录 `name`、下载 `url`、小写 SHA-256、`sizeBytes`、`kind`。必须包含所有训练产物、指标、主题词、矩阵、数据划分结果、图表、绘图数据及日志。保留原生相对目录：训练输出位于 `training/`，图表和绘图表格位于 `native/`，报告入口为根目录 `index.html`。保留唯一的 `split_results.json`，其中含全量、训练、验证、测试四组；划分图表保留 `splits/<role>/` 层级。证据字段复用原生报告结构，所有 `path` 指向清单内相对文件名；HTML 使用相对资源链接，不携带服务器绝对路径。

下载地址允许带签名查询参数。API 密钥仅发送给相同来源的地址；对象存储须自行提供有时效的授权链接。文件路径不能含绝对路径、上级跳转、Windows 保留名或大小写冲突，也不能附带自动执行脚本。

远端声称完成后，本机仍显示“正在接收并校验结果与图表”，直到全部文件大小和哈希核验通过、由本机能力 API 登记完成，再展示完成状态。文件下载失败不会自动重训或再次计费；恢复连接并重新打开应用可重试同一结果的下载。已经回传的结果可在本地读取、下载及生成分析报告；修改过的产物不会被当作原始证据。

## 配置与验证范围

桌面配置保存在正常数据目录的 `compute-settings.json`。源码 Web 的对话和手动服务共用 Agent 目录中的计算配置。自定义启动方式请给两个服务设置相同的 `THETA_COMPUTE_SETTINGS_HOME`，项目存储仍保持分开。`THETA_WORKER_API_URL` 是内部能力接口地址，不能用作商业训练服务配置入口。

仓库测试覆盖真实 HTTP 请求、上传和划分数据、项目隔离、费用上限、确认绑定、两种训练模式、切换配置后恢复查询、结果校验失败，以及本地默认行为。Python 测试覆盖下载产物登记和本地报告证据生成。远端测试使用隔离的接口服务，实际商业服务的计费、权限、GPU 调度与兼容性仍须针对供应商验收；需要服务器模型与运行环境才能重现的原生图表编辑，不会在本机伪造替代结果。
