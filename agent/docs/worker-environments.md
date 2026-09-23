# Worker 环境与上云迁移

Agent 仍自主选择工具。宿主负责将工具调用送进合适的环境；模型不能提供解释器路径或自行安装依赖。计算、外部 embedding、结果报告和深入解读继续遵守原来的确认边界。

| 环境 | 工作范围 | 宿主配置 |
| --- | --- | --- |
| classic | LDA、BTM、HDP、DTM、STM | `THETA_WORKER_CLASSIC_PYTHON` |
| neural | ETM、NVDM、GSM、ProdLDA、CTM、BERTopic、THETA | `THETA_WORKER_NEURAL_PYTHON` |
| reports | 读取结果矩阵、证据和调用仓库已有可视化 | `THETA_WORKER_REPORTS_PYTHON` |

每项配置填写环境内 Python 的绝对路径。未配置时使用 `THETA_PYTHON` 对应的控制环境，并明确显示 `shared`。指定的 Python 缺失时直接失败，不回退。`configured` 仅表示指定了解释器；`isolatedVenv` 和 `inheritsSystemPackages` 才反映虚拟环境的包隔离情况。这不是操作系统安全沙箱，也没有硬性 CPU、内存或显存配额。

## 本地准备

可先只拆分 neural，其余维持现有环境。以下是供运维执行的示例，不会在 Agent 对话中自动运行；安装会访问软件源，但不会预先下载模型权重：

```sh
# 在仓库根目录，以已安装的 Python 3.10+ 创建不继承系统包的环境。
python3 -m venv agent/.local/runtimes/neural
agent/.local/runtimes/neural/bin/python -m pip install -r src/models/requirements.txt
# 当前引擎还会使用 NLTK，THETA 微调使用 PEFT；按所用能力准备。
agent/.local/runtimes/neural/bin/python -m pip install nltk peft
```

将实际绝对路径填入私有 `agent/.env.local`，例如：

```dotenv
THETA_WORKER_NEURAL_PYTHON=/absolute/repository/agent/.local/runtimes/neural/bin/python
THETA_WORKER_NEURAL_REVISION=neural-v1
```

classic、reports 同样可创建独立环境。根目录现有 requirements 是完整引擎依赖基线，不是三个精简锁文件：传统训练入口仍导入 Torch 等公共模块，不能只装 gensim 就宣称就绪。不同操作系统、Python 和 CUDA 版本需分别验证依赖；安装完整依赖不等于所有模型均已加载验收。中文字体、NLTK 资源、SBERT/Qwen 权重需作为单独资产准备，不能由训练偷偷下载。继续使用 `SBERT_MODEL_PATH` 等已有模型路径配置。

重启终端应用后检查：

```sh
./theta doctor --json
./theta
```

在对话中要求“检查 BERTopic 的本地运行环境，不训练”，Agent 会在 neural 环境探测对应依赖和权重。`doctor` 检查三个环境身份及 LDA 依赖，不声称已验证全部模型。环境的依赖指纹包含安装包版本和 Python 版本；变更后需重新确认训练。不要在任务运行中原地升级环境，应创建新环境并更新配置和 revision。旧的、不包含 runtime 信息的待执行授权需要重新提案确认；已运行任务继续使用原进程。

## 本地执行与可迁移边界

`workers/runtime_environments.py` 是唯一模型到环境的映射。JSON 入口在调用能力前切换解释器，进程 PID 保持不变，终端超时和中断能作用于实际 worker。训练提交、就绪检查和后台 `compute.run` 共用选中的解释器，继续调用根目录 `trainning/worker` 和 `src/models`，不复制训练算法或可视化代码。

训练请求的 `execution.runtime` 和任务状态记录 profile、revision、实际解释器、Python 版本和依赖指纹；已有 `ExecutionSpec.runtime.key/version` 使用 profile/revision。这使本地任务能够追溯到运行环境。训练确认绑定完整运行环境，改变配置或包版本后旧确认不再有效。

状态、取消、模型清单、已有产物路径清单保留在控制环境，损坏的训练或报告环境不会阻止诊断。数据导入和理解仍使用控制环境的数据读取依赖。reports 环境只负责现有结果能力，不能因此绕过报告或深入解读授权。

## 迁移至云端

1. 分别将通过验收的 classic、neural、reports 环境构建成固定版本的镜像，锁定 Python、包版本及系统依赖；将同一 profile/revision 登记到云端 runtime。GPU 环境在云机器按实际 CUDA/驱动构建，不能复制 macOS 的 venv。镜像应包含同一版本的 `agent/workers`、`trainning/worker` 和根目录计算源码。
2. 本地绝对解释器路径只属于本地部署。云端 runtime 使用镜像摘要和登记的 runtime ID；由宿主维护映射，不能把本地路径直接发送给调度器作为执行命令。
3. 沿用 `ComputeGateway` 的 submit/status/cancel/results 边界。仓库已有 Go 适配器支持通过 `THETA_COMPUTE_URL` 和私有 `THETA_COMPUTE_BINDINGS` 接入登记好的服务，配置格式见 Agent README。数据/结果迁移到对象存储后用内容 hash 和对象引用保持版本校验；本地 SQLite 不是跨机器调度队列。
4. Go 提交协议已可传输 Agent 的命名空间扩展参数和 `input.*` 列映射，分布式 worker 会按当前引擎源码校验并路由；外部 embedding 预算、Agent 指定时限和结构化结果报告仍有 README 中列出的限制。本地改动不代表远端已部署，迁移相应模型前仍须重建 worker 镜像并重新做云端验收，禁止静默丢弃参数。

本地可先将多个 profile 指向同一已验证解释器，再逐个迁移到独立环境。云端扩容按对应 worker 队列和资源需求进行；容器的只读模型缓存、临时工作目录、网络权限及资源限额应由部署层落实，不能把 venv 当作这些限制。
