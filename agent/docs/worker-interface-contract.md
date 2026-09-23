# Worker 替换契约（保持 API 不变的边界）

换掉 worker 的实现代码时，下面这些边界必须保持兼容。每一节都给出：谁会调用、必须返回什么、以及改动后怎么验证。新增能力时在末尾的“扩展清单”登记。

## 哪些代码是你的、哪些是边界

| 范围 | 归属 | 说明 |
| --- | --- | --- |
| `trainning/worker/pipeline.py` 的命令构造与执行、`service.py` 的队列消费 | 可替换 | 只要仍产出本文档第 4、5 节的消息与产物 |
| `src/models/**` 引擎实现与算法 | 可替换 | 只要仍接受第 3 节的列与 CLI 参数 |
| 各 profile 的 Python 环境内容（依赖、编译产物、隔离 venv） | 可替换 | 路径由 `THETA_WORKER_<PROFILE>_PYTHON` 指定，指纹变化会体现在确认卡 |
| `agent/workers/__main__.py` 操作名与 JSON 信封 | 边界 | 改名或改形状会直接打断 Node 侧调用 |
| `trainning/worker/protocol.py` 的 `ExecutionSpec` v2 | 边界 | 与 Go 控制面共用的协议 |
| `agent/workers/local_compute.py` 与 `trainning/worker/dataset.py` 的规范化 | 边界 | 本地/分布式 worker 产生同一引擎输入列契约 |
| `agent/workers/results_reader.py`、`result_analysis.py` | 边界 | 结果文件命名与目录绑定的唯一解析点 |
| `agent/workers/runtime_environments.py` 的变量名与 profile 归属 | 边界 | 确认卡与回执据此绑定授权 |
| `agent/workers/pipeline_adapter.py`、`model_contract.py` | 半替换 | Agent 对 pipeline/引擎声明的适配；你的 worker 若改 CLI 参数，需要同步这里 |

## 1. Node ↔ Python 能力传输

调用方：`agent/src/adapters/python-worker.ts`（`PythonCapabilityWorker`）。

```text
<python> -m workers <operation>      # cwd = agent/
stdin : 一个 JSON 请求对象（≤ 2 MiB）
stdout: 一行 {"ok": true, "data": <任意 JSON>}
失败  : {"ok": false, "error": "<说明>"} 且退出码非 0
```

`data` 的形状由操作名决定，调用方按字段读取，因此**不能改名、不能把对象换成数组**。当前操作清单（`agent/workers/__main__.py` 的 `HANDLERS`）：

```text
statistics.methods / statistics.inspect / statistics.preview / statistics.execute
models.list / models.inspect
runtime.environments / runtime.environment / runtime.check / runtime.config
compute.preview / compute.submit / compute.status / compute.cancel / compute.results / compute.remote_report
dataset.import / dataset.discover / dataset.use / dataset.understand / dataset.profile
dataset.preprocess
plan.validate
```

新增能力 = 在 `HANDLERS` 注册新名字 + Node 侧新工具；已存在的名字不要改语义。

`dataset.preprocess` 的受限代码、原始数据保留、派生版本与输出契约见 [conversation-preprocessing.md](conversation-preprocessing.md)。

## 2. 运行环境路由

`runtime.environments` 返回四个 profile，字段固定：

```json
{ "profile": "classic", "revision": "v1", "python": "/abs/bin/python", "mode": "configured|shared",
  "available": true, "configurationVariable": "THETA_WORKER_CLASSIC_PYTHON", "models": ["lda", "..."] }
```

`runtime.environment`（单 profile）在此基础上多返回 `pythonVersion`、`prefix`、`dependencyFingerprint`、`isolatedVenv`、`inheritsSystemPackages`、`sandbox`。确认卡与结果回执会带上 profile/revision/依赖指纹，用来把授权绑定到具体运行环境。

模型归属（`agent/workers/runtime_environments.py` 的 `PROFILES`）：

| profile | 模型 | 解释器变量 |
| --- | --- | --- |
| classic | lda, btm, hdp, dtm, stm | `THETA_WORKER_CLASSIC_PYTHON` |
| neural | etm, nvdm, gsm, prodlda, ctm, bertopic, theta | `THETA_WORKER_NEURAL_PYTHON` |
| reports | —（结果整理/报告） | `THETA_WORKER_REPORTS_PYTHON` |
| statistics | —（自由分析） | `THETA_WORKER_STATISTICS_PYTHON` |

每个 profile 还可配 `THETA_WORKER_<PROFILE>_REVISION`。语义要点：**配置了就用配置的，`available=false` 直接报错，绝不回退到别的环境**；未配置才用控制进程解释器。

## 3. 引擎输入契约（数据格式转换的出口）

`agent/workers/local_compute.py`（本地）和 `trainning/worker/dataset.py`（分布式）把上传格式转成引擎 CSV，列名固定。分布式协议通过 `params` 中的 `input.text_column`、`input.time_column`、`input.label_column`、`input.covariates` 传递映射；规范化后这些字段会被剥离，永远不会成为引擎参数：

| 列 | 来源 | 传给引擎的参数 |
| --- | --- | --- |
| `text` | 方案选定的文本列 | 位置参数 |
| `label` | 标签列（可选） | `--label_col label` |
| `year` + `timestamp` | 同一时间列（可选） | `--time_column year` |
| `cov_0` … `cov_N` | 协变量按顺序 | `--covariate_columns cov_0 …` |

引擎入口必须存在 `src/models/run_pipeline.py`（`engine_root()` 用它判断仓库有效性）、`prepare_data.py`、`main.py`。转换前后都会校验托管副本 sha256，被改动即拒绝训练。

神经模型的公开结构参数为 `hidden_sizes`（1～10 层，每层 4～8192）、`dropout`、`learning_rate`、`patience` 和 `no_early_stopping`；未传 `hidden_sizes` 时继续使用 `hidden_dim` + `num_layers`。THETA `zero_shot` 可把 `embedding_provider` 指向 OpenAI-compatible 远端服务，并传模型名、API 基址、密钥环境变量名和输出维度；任务中不得传密钥明文，密钥只存在 Worker 宿主环境。

## 4. Worker 执行协议（ExecutionSpec v2）

`trainning/worker/protocol.py` 定义的 `job.ready` 消息，Agent 与 Go 控制面共用：

```json
{ "schema_version": 2, "type": "job.ready", "event_id": "...", "task_id": 1, "attempt": 1,
  "task_version": 1, "user_id": "...", "priority": 0,
  "dataset": { "ref": "...", "project_id": "...", "object_key": "dataset/data.csv", "filename": "data.csv", "format": "csv", "sha256": "..." },
  "model": { "id": 1, "name": "lda", "framework": "theta" },
  "runtime": { "id": 1, "key": "classic-cpu", "version": "v1", "executor": "theta_pipeline", "image": "local" },
  "params": { "num_topics": 3 },
  "resources": { "accelerator": "none", "cpu_cores": 2, "memory_mb": 4096, "gpu_count": 0, "gpu_memory_mb": 0, "timeout_seconds": 3600 },
  "output_prefix": "<jobId>/", "created_at": "..." }
```

固定约束：`schema_version` 必须是 2、`type` 必须是 `job.ready`、`runtime.executor` 必须是 `theta_pipeline`、`dataset.filename` 不含路径、`output_prefix` 无 `..`、`priority` 在 0–255。改这些等于换协议，需要同时改 Go 控制面与结果回执。

## 5. 结果产物契约

结果根目录必须命名为 `<trainingRunId>` 或 `<trainingRunId>__<后缀>`（读取时按路径段校验绑定关系），树内不得有符号链接，读取前后 tree hash 必须一致。

| 类型 | 识别规则 |
| --- | --- |
| JSON 证据 | `metrics*.json`、`topic_words*.json`、`info*.json`、`training_history*.json`、`topic_evolution*.json`、`covariate_effects*.json`、`covariate_info*.json`；单文件 ≤ 1 MiB，最多 12 个 |
| 矩阵 | `theta*.npy`、`beta*.npy`，单文件 ≤ 32 MiB，最多 6 个 |
| 表格 | `*.csv` / `*.tsv`，≤ 1 MiB，最多 32 个 |
| 图 | `*.png|jpg|jpeg|svg|pdf|html`，最多 240 个 |

命名决定解读分类（`topic_words*`→话题、`metrics*`→指标）；改名会让结果解读拿不到证据，因此属于 API 的一部分。

分布式 Worker 的 `model.tar.gz` 保持原有 `result/` 目录，并向后兼容地加入：

```text
result/                 # 模型、指标、原生表格和图表
workspace/              # 训练实际使用的词表、矩阵、texts/source_rows 与行溯源材料
input/data.csv          # Worker 实际送入引擎的规范化 UTF-8 数据
```

归档不得含符号链接、硬链接、设备文件或越界路径。Worker 完成事件的
`output_files.integrity` 必须提供归档 SHA-256、大小，以及规范化输入
SHA-256。现有 `archive`、`manifest`、`log`、`files` 字段保持不变。

Agent 的 `compute.remote_report` 只接受 Agent 私有持久目录中已经过归档
SHA-256 校验并安全解包的路径。它生成与本地训练一致的
`theta.result-report.v2`；只有源数据 hash、规范化输入和
`source_rows/texts` 同时通过校验，才把 `matrixRowsAligned` 标为 true。

## 6. 状态与阶段

`status` 取值 `queued | running | completed | failed | cancelled`；`phase` 是自由字符串（现有：`scheduling`、`downloading`、`preparing_data`、`training`、`evaluating`、`visualizing`、`uploading`、`cancelling`、`completed`、`failed`、`cancelled`）。`compute.status` 返回的 job 对象至少要有 `id`、`status`、`phase`、`percent`；失败时 `error`/`diagnostics` 用于解释原因。阶段百分比只是阶段标记，不代表完成比例。

## 替换步骤与验证

1. 把你的解释器路径写进 `agent/.env.local`（第 2 节的变量），`./theta doctor --json` 确认 `available=true`、依赖指纹是新的。
2. 跑 `node scripts/check-worker-contract.mjs`：只读检查上述边界是否漂移，不训练、不写数据。
3. 跑 `pnpm test:workers`（Python 侧契约）与 `pnpm smoke:training`（真实小 CPU LDA 闭环）。
4. 跑 `node scripts/acceptance-conversion.mjs`（Excel/JSONL → 引擎 CSV → 真实训练）。
5. 全绿后再推；任何一步红，先看它对应哪一节边界。

## 扩展清单（新增能力时同步这里）

- 新操作名 → `agent/workers/__main__.py` 注册 + Node 侧工具 + 本文档第 1 节。
- 新模型 → `capabilities.MODELS` + `runtime_environments.PROFILES` 归属 + 根目录引擎支持 + `knowledge/` 参数文档。
- 新数据格式 → 同步 `agent/workers/dataset/readers.py`、`trainning/worker/dataset.py`、前端 accept 列表与两侧转换测试。
- 新结果文件类型 → `results_reader.EVIDENCE_NAME` 或 `result_analysis.presentation_evidence` + 本文档第 5 节。
- 新执行器（非 `theta_pipeline`）→ 需要同时改 `protocol.py` 校验、Go 控制面 runtime 表与确认卡文案，属于协议升级而不是替换。
