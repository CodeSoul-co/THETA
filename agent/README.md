# THETA Agent

独立启动的主题建模与自由分析 Agent。用自然语言讨论研究目标、导入文件、理解数据、提出模型方案、确认并执行训练、解释结果，也可以在同一对话中继续比较实验。工具由对话模型按需选择，没有强制用户逐页完成的工作流。

定位为以主题建模为核心的数据分析 Agent，覆盖科研、文本、商业、政策等场景。它先判断用户要回答的问题、分析单位与证据需求，再按需探索数据、讨论研究设计或准备模型；不会将所有问题套成客服分析，也不会对没有文本的数值表强行做主题建模。默认主题模式保留；自由模式另提供登记的统计、计量、预测、生存与优化工具，主题模型成为按需选择的文本工具。方法可用不等于识别条件成立。

支持的文本分析默认建议“描述性统计＋主题建模”：前者说明数据覆盖、时间/分组分布与分析限制，后者探索议题结构；根据目标和现有证据按需调用工具。用户可只选其中一部分，组合建议不会自动启动训练或绕过结果解读确认。

首次理解数据后，尚未明确下一步时提供两个选择：快速开展描述性统计＋一个推荐模型，或者先聊清楚用户目的。选择快速分析后直接准备具体方案和训练确认卡，不强制补全背景问卷；选择澄清目的则每次只问关键问题。用户已有明确指令时直接接续。

系统提示集中在 `src/conversation/prompts.ts`：`INSTRUCTIONS` 管理对话、场景理解、方法选择、工具与授权纪律；`INTERPRETATION_INSTRUCTIONS` 管理单独获批的研究解读，要求联系原文与矩阵证据、区分观察/估计/推断，按当前问题组织结果。执行权限仍由宿主校验，不能仅靠修改提示词绕过。修改后运行 `pnpm build` 并重启 CLI 生效。

开发改动合入 `dev` 集成验证，再合入 `main`。已复用原 `frontend/` 前端，对话模式通过 `web/server.ts` 接入同一 Agent 核心。启动方式见 [网页对话说明](../frontend/README.md)；免登录的开源发行版见 [开源使用指南](../docs/opensource-agent.md)。

## 自由分析模式

网页在现有工作台的“对话”中选择“自由分析”；CLI 使用 `/mode free`。独立 Python statistics worker 提供 138 个登记方法，按批次确认执行，导出论文表格、图和复现脚本。需要单独安装统计环境。

- [安装、使用与工具协议](docs/free-analysis.md)
- [逐项方法、参数与示例](docs/statistics/methods.md)
- [SPSSPRO 公开目录实现状态与缺项](docs/statistics/spsspro-coverage.md)
- [开源验证记录](../docs/opensource-release-validation.md)

兼容目标为 Python 方法等价与 esttab 风格表格，不是商业软件所有功能或指定版本逐值替代。

## 内置绘图技能

已内置 MIT 授权的 [Data Viz Skill](https://github.com/AdamsukS/data-viz-skill)，固定版本包含 22 个 Python 模板。Agent 可读取技能、选择模板并交付可修改工程；示例预览明确标记，准备工程不会被计作已完成用户图表。见 [推荐说明、工具与本地渲染方法](docs/data-viz-skill.md)。

## 启动

**网页双模式工作台：** 仓库根目录执行 `./theta-web start`，同时后台启动前端、对话 API、手动 API；`./theta-web restart` 更新编译并重启，`./theta-web status` 查看状态。固定复用当前仓库 `.theta_agent/` 和 `.local/manual-workbench/`。环境、日志和停止方式见 [网页启动说明](../frontend/README.md)。下列命令是独立 CLI 的入口。

要求 Node.js ≥ 22.13、pnpm，以及 Python ≥ 3.10（计算环境按根目录引擎依赖准备）。

```bash
cd agent
pnpm install --frozen-lockfile
pnpm build
pnpm start
```

也可以从仓库根目录执行 `./theta`。首次启动缺少编译产物时入口会调用本包 TypeScript 构建；修改源码后运行 `pnpm build`。普通对话无需 MySQL/Redis/Go 服务。

模型配置放在私有 `agent/.env.local`，参考 `.env.example`。优先级为进程环境 → Agent `.env.local` → Agent `.env` → 仓库 `.env.local` → 仓库 `.env`。`THETA_ENV_FILE` 可指定唯一环境文件。支持 DeepSeek、OpenAI、MiniMax、GLM 和自定义 Chat Completions 兼容端点；这不代表支持所有供应商的原生 API。

### Docker Agent 服务

可部署的 Agent API 使用稳定前缀 `/api/v1/agent`，通过现有 Business API
校验 Session 和 CSRF，再由原有适配器调用 Worker；不会开放 Redis、训练进程或
Worker 内部端口。localhost 开发入口 `/api/v3` 继续保留，但 deployed 模式会将其
关闭。

```bash
docker compose -f deploy/docker-compose.yml build
docker compose -f deploy/docker-compose.yml up -d
curl http://localhost:8088/healthz
curl http://localhost:8088/api/v1/agent/capabilities
```

正式部署时把 [Nginx location](deploy/nginx-agent-location.conf) 合并进现有 HTTPS
虚拟主机，并持久挂载 `/data`。接口、鉴权、SSE、确认卡和前端接入流程见
[Agent Frontend API](docs/THETA_AGENT_API.md) 与
[OpenAPI 3.1](docs/openapi.agent.yaml)。

终端内支持 `/model`、`/models`、`/attach 文件路径`、`/new`、`/sessions`、`/resume`、`/help`、`/deny`、`/exit`。拖入文件路径也会导入数据。界面展示模型、会话、数据数量、工具活动、训练进度和待确认卡片；Ctrl+C 中断当前对话生成，明确取消训练才停止后台任务。

可以直接说“看看有哪些已有数据”“用客服反馈分析服务改进机会”“解释这次训练的主题表和图”。Agent 在索要上传前用 `datasets_discover` 查看仓库 `data/`（宿主可用 `THETA_DATA_DIR` 指定其他目录），用户选定后通过 `dataset_use` 引用版本固定的托管副本。发现工具不读取正文、不启动训练；跳过隐藏文件、符号链接和已知模型/工作目录，单次扫描最多 5000 个条目，每页 50 个候选。未发现数据时才提示上传。

`dataset_understand` 和 `research_continue` 为业务理解提供最多 8 段、每段 450 字符的文本摘录，以及可分组维度和时间覆盖。摘录会进入当前对话模型的上下文；常见邮箱、手机号和凭据样式会被遮盖，但这不是完整匿名化。Agent 根据内容与用户目标讨论业务对象、现象、决策与证据边界，技术格式画像由 `dataset_read` 按需检查。小样本摘录不能用于推断总体主题占比。

首次完成数据理解后，宿主自动将回答、用户目标、数据引用和待确认事项保存到 `.theta_agent/contexts/context-*.md`，后续问答与结果解释会注入选中的 Markdown。`/context` 查看当前文档，`/contexts` 列出本地上下文，`/context 标识` 创建本会话的复用副本；模型也可按需使用 `context_save` 更新目标和理解。复用上下文不会自动附加旧数据、修改原会话或继承计算授权。当前上下文库属于本机用户；未来多租户服务需在存储适配层增加身份隔离。

上传支持仅路径，也支持附带指令，例如 `/attach "/path/业务 数据.csv" 先理解内容，不训练`；不附带指令时沿用前文目标。执行期间输入新消息会被保留，并中断当前对话调用后继续处理，后台计算仍需明确取消才停止。

普通回复允许直接输出文本，不强制调用工具。每轮默认最多 8 轮自主模型调用、12 次工具请求，随后使用无工具的回答总结已有证据；整体等待上限为 180 秒。相同读取在本轮复用缓存，多次重复命中会提前结束继续检查。模型请求、HTTP 尝试/重试和工具调用分别记录，不把工具计时误作整轮计时。完成后保留逐条记录，`/trace` 可重看最近一轮；会话保留最近 1000 条事件。SDK `onEvent` 与 `ExecutionEvent` 为不依赖 UI 的 v1 事件接口，包含 turnId/callId、类型、状态、时间和耗时；JSON 工具入口也返回 `executionEvents`，网页通过 SSE 同步这些事件，并支持停止当前生成和恢复历史对话。

## 替换 worker 实现时保持 API 不变

换掉 `trainning/worker`、引擎或 Python 能力层的实现代码时，五个边界必须保持兼容：Node↔Python 能力传输（`python -m workers <operation>` + `{"ok":true,"data":…}`）、运行环境路由（`THETA_WORKER_<PROFILE>_PYTHON` 与 profile/依赖指纹）、引擎输入列契约（`text`/`label`/`year`+`timestamp`/`cov_N`）、`ExecutionSpec` v2 消息、结果产物命名与目录绑定。逐条定义、必填字段与扩展方式见 [Worker 替换契约](docs/worker-interface-contract.md)。

改动后用只读核对与真实闭环确认没有漂移：

```bash
node scripts/check-worker-contract.mjs   # 只读，不训练；漂移时列出具体边界并非零退出
pnpm test:workers
pnpm smoke:training
node scripts/acceptance-conversion.mjs
```

## 模块边界

```text
agent/
  cli/                  # 独立终端应用；无 src 子目录
    bin/                # 可执行入口
    shell/              # 输入、斜杠命令、会话切换及交互测试
    ui/                 # 终端布局、渲染、进度、确认卡片及界面测试
    commands/           # 外部 Agent 的 JSON 命令入口
  src/                  # Agent 可复用核心
    conversation/       # 原生工具调用循环与上下文
    domain/             # 研究、方案、确认和计算端口的数据类型
    tools/              # 公共 schema、校验和工具执行
    providers/          # 多供应商推理
    memory/             # 本地会话与研究持久化
    adapters/           # Python 能力传输、ComputeGateway 的本地/Go 实现
  web/                  # 本地 HTTP/SSE 适配；复用同一对话核心
  workers/              # 数据画像、规范化、结果证据及本地计算适配
../src/models/          # 唯一 THETA 算法源码；STM 数值修复与测试在此
../trainning/worker/    # 现有计算协议与管线，直接复用
```

CLI 单向依赖 Agent 核心；数据能力和计算任务通过 JSON/HTTP 边界调用。迁入了原对话循环、终端和供应商实现、SQLite 会话、数据读取/画像和 hash 验证结果读取；旧 application service、固定训练 FSM 与嵌套 THETA 快照未复制；网页依赖独立放在根目录 `frontend/`，不进入核心包。原 Hypha 推理接口抽成兼容的结构化端口，启动无需克隆 Hypha。

## 本地计算

默认 `LocalComputeGateway` 使用独立后台 Python 进程，直接导入根目录 `trainning/worker` 的 `ExecutionSpec` 和 `ThetaPipeline`。仅新增本地任务持久化、单次领取和传输适配，不实现第二套数值计算算法。状态在 `compute.sqlite`；每个任务隔离输入、工作目录和产物，退出 CLI 后继续运行。

`THETA_PYTHON` 指定控制进程及未单独配置的 worker Python；未设置时优先使用 `agent/.venv`，再使用 `python3`。现在可分别指定传统模型、神经模型和结果环境，详见 [worker 环境与上云迁移](docs/worker-environments.md)。`./theta doctor --json` 展示三个环境的实际解释器、Python 版本、依赖指纹、是否继承系统包，以及 LDA 基础依赖探测，不输出密钥。配置路径不等于依赖就绪，readiness 也不等于真实训练验收；当前开发机原有 `.venv` 继承系统包，属于共享兼容环境。

Agent 自己选择如何讨论、分析和提出方案。训练、结果读取与基础解读、结合业务上下文的综合解读分别请求用户确认；没有“先批准方案、再进入下一阶段”的强制工作流。确认绑定数据 hash、参数、目标端点、模型配置和本次额度，修改其中任何一项使批准失效；重复查询同一任务不会启动新训练。原始数据保留，训练从托管副本转换为引擎要求的 UTF-8 CSV：`text` 取选定文本列，`label` 取标签列，`year`/`timestamp` 取同一时间列，协变量按顺序写成 `cov_0`、`cov_1`……，原列名不会进入引擎。

支持 CSV/TSV/TXT/Markdown/JSON/JSONL/NDJSON/XLSX/XLS/Parquet/PDF/DOCX（Excel/Parquet 需对应 reader 依赖；PDF/DOCX 分别按非空页/段落形成文本记录）：分隔符自动嗅探，编码依次尝试 UTF-8（含 BOM）、UTF-16、GB18030/GBK，都无法识别时用 latin-1 保底并明确告警；列名做 NFC 归一化并去掉 BOM 与控制字符；多工作表 Excel 默认取第一张非空表。转换前后都会校验托管副本 hash，副本被改动即拒绝训练。本地导入上限 200 MiB，训练规范化上限一百万行，不能把画像样本当作完整训练数据。

方案展示和授权前，Agent 将 `language=zh/en` 规范化为 `chinese/english`；枚举参数会与根目录预处理和训练入口的实际声明校验，不导入或执行算法。文本语言由引擎自动检测，`language` 主要控制图表显示。`runtime_check` 只代表依赖和缓存就绪，不保证参数正确或训练成功。

失败的本地任务通过 `run_status` 返回 `job.diagnostics`，从该任务的日志中提取有长度限制、经过凭据脱敏的错误证据与执行阶段，已有失败记录也适用。诊断不会重跑训练，不提供任意文件读取或完整日志转发；日志缺失时明确返回无法诊断。修正方案后仍需用户对新计算确认。

训练完成后，Agent 请求结果整理与基础解读的独立确认。`results_read` 读取已验证的主题词、指标、原生表格与图表清单；矩阵只提供有界原值摘录，不在 Agent 中重新实现统计或绘图。表格与矩阵超出摘录范围时明确标注，完整文件通过报告交付。

`report` 视图通过 `workers/result_report.py` 适配调用根目录 `src/models/visualization/run_visualization.py` 的完整原生入口。显式绑定训练工作区词表、BOW、theta/beta；经输入 hash 和逐行文本校验后才接入原始时间、渠道标签。输出包括全部可用原生图、原生 CSV、原始 NPY/模型文件和 HTML 索引；写入独立报告目录，不改变训练结果树。缺少时序、训练历史、多 K 实验等证据时记录跳过原因，不补造数据。原可视化里的随机词义/词分布演化与随机 KL 曲线已停止输出，避免作为研究证据。Agent 不维护另一套简化图表。

`results_synthesize` 为“指定问题 + 已完成任务 + 当前研究 Markdown 版本”申请另一次确认。批准后单独调用 `agent.theta.result-interpretation`，结合原始文本摘录、原生表格、原始矩阵和已保存业务目标回答研究问题。要求引用文本行号与产物、区分观察与推断，并逐类覆盖图表证据和缺失项；不默认比较模型。当前最多提供前40行原始文本（36条验收数据全部覆盖），超过时明确说明不是代表性抽样。独立 Markdown 保存到 `.theta_agent/interpretations/`，不覆盖最初数据理解。用户可以拒绝、修改问题或切换实验。

`/reports` 查找当前会话的完整报告，`/open 序号` 在默认浏览器打开 HTML；恢复会话后仍可使用。重新请求生成报告需确认，相同结果和代码版本可复用已校验报告。图像像素未交给视觉模型；空间几何和颜色不能从文件名推断。原生主题表一基编号、矩阵零基索引；原始行与矩阵只有验证对齐后才可关联。目前完整报告适配本地计算，Go 控制面仍需补齐结果 manifest。

本地适配默认使用 CPU，可通过计划 `device="cuda:0"` 显式选择已就绪的 CUDA，HF 模型下载保持离线。THETA `zero_shot` 的提案默认沿用 `EMBEDDING_PROVIDER`（未配置时为 local），并在保存前明确写入计划；用户明确选择 local 时保留该选择。可显式选择 `params.embedding_provider="cloud"`：Agent 读取主环境文件配置，在确认卡中列出接收端、embedding 模型、全文/派生词表的数据范围与 `externalRequestLimit` 上限（默认 100 次，最多 1000 次）。只有确认后才发送请求，实际 HTTP 尝试（包括失败）计入共享额度；额度用完、配置变化、重定向或取消任务都会阻止新请求。微调模式仍要求本地 Qwen；其他神经模型需要配置本地权重。没有通用模型下载器，缺少权重会如实报告。THETA 云模式不依赖本地 SBERT 或 Qwen 库/权重；CTM/BERTopic 的本地 SBERT 要求保留。GPU 与其他模型的全量训练尚未完成硬件验收。

会话、研究、数据和产物默认保存在启动目录的 `.theta_agent/`；用 `THETA_AGENT_HOME` 固定位置，可跨目录恢复。明确停止训练通过后台取消标记执行。异常退出造成的未确认任务不会自动重复启动；心跳丢失时显示状态不确定，需要核对。单机适配不提供分布式调度和容器级资源限额。

## Go 服务与后续扩容

设置 `THETA_COMPUTE_URL` 切换到已有 Go 控制面，任务状态完全由 Go 管理，本地计算数据库不参与远端任务调度。`THETA_COMPUTE_BINDINGS` 指向宿主维护的私有 JSON，绑定已登记数据版本、规范化列和模型 runtime：

```json
{
  "userId": "已认证用户 UUID",
  "projectId": "项目 UUID",
  "datasets": {
    "dataset-本地内容hash": { "ref": "已登记数据引用", "sha256": "原始数据hash", "textColumn": "content" }
  },
  "models": { "lda": { "modelId": 1, "runtimeId": 1 } }
}
```

远端对象须由宿主按同样列映射规范化并在 Go 业务库登记；适配器不会让 LLM 自填身份或直接写业务库。远端服务须置于已认证的业务边界后；`THETA_COMPUTE_TOKEN` 仅传递凭据，不会替 Go 自动实现鉴权。提交授权同时绑定宿主数据/runtime 映射与凭据指纹；配置变化需重新确认。现有 Go 协议尚不能执行云 embedding 预算校验，远端云 embedding 方案保持拒绝，本地 embedding 参数会显式传给远端，避免继承服务端云配置。现有提交 API 不接受 Agent 指定的运行时限，远端时限取决于已登记 runtime，确认卡会说明这一点。

当前 Go API 已提供 `POST /api/v1/datasets`。当 Agent 管理的数据集尚未绑定到控制面时，适配器会先上传原始文件、核验 SHA-256 并登记到用户项目，再提交训练任务；静态部署绑定仍继续使用并严格校验。任务提交 API 仍没有请求幂等键，因此适配器不自动重试结果不确定的任务 POST。更新后的 `/result` 与 `/model/download` 会返回对象索引、归档 SHA-256、长度及规范化输入 hash。用户确认结果读取后，Agent 下载并安全解包 `result/`、`workspace/` 与 `input/data.csv`，生成本地 `theta.result-report.v2`；只有逐行文本和矩阵行号校验通过时才允许关联原始文本。旧任务或旧 Worker 缺少完整完整性字段时会拒绝深入关联，不会猜测。Worker 横向副本数不改变 Agent API：一个副本串行消费，多副本用于并行吞吐。

## Business API（合作方托管计算）

设置 `THETA_BUSINESS_API_URL`（例如 `https://theta.code-soul.com`）切换到公开 Business API；`THETA_BUSINESS_IDENTIFIER`、`THETA_BUSINESS_PASSWORD` 与 `THETA_BUSINESS_BINDINGS` 放在本机私有配置，不进入模型上下文或浏览器。Node 进程按 [开源验证记录](../docs/opensource-release-validation.md) 的规则充当浏览器：内存保存 Session Cookie 与 CSRF Token，写请求带允许的 Origin，`csrf_failed` 时通过 `/api/v1/auth/me` 刷新且只重试一次，401 只对读请求重新登录。

绑定文件提供项目与模型映射。公开接口没有模型/运行时列表，`model_id` 必须由部署方给出，不能猜测：

```json
{
  "projectId": "已存在项目 UUID",
  "projectTitle": "THETA Agent",
  "allowProjectCreate": false,
  "models": { "lda": { "modelId": 1, "runtimeId": 1 } },
  "datasets": { "本地 datasetRef": { "datasetRef": "已登记引用", "sha256": "内容 hash" } }
}
```

未登记的数据集会以 multipart 上传托管副本，核对远端 `sha256` 与本地一致后才创建任务。Agent 会把文本/时间/标签/协变量列映射编码为 worker 自有的 `input.*` 参数；worker 下载后统一转换为引擎 CSV，并在训练前剥离这些控制字段。公开协议现可传输 `model.*`、`trainer.*`、`fit.*`、`prepare.*`、`main.*`、`config.*`、`pipeline.*`、`word2vec.*`、`embedding.*` 扩展参数，实际接收端仍由 worker 根据当前引擎源码严格校验；未知命名空间或无接收端的参数会失败，不会静默丢弃。`device` 与带计费额度的云 embedding 仍不在当前 Business API 契约内；该适配器只有在合作方结果接口也提供归档完整性和规范化输入证明后才能启用原始行深入解释。提交沿用同一防重复机制，结果不确定时不自动重发。

```bash
npm run probe:business                                   # 无凭据公开接口探测
npm run acceptance:business -- --confirm-live-writes      # 登录、项目、数据集 hash（需私有配置）
```

接入结论、契约证据与剩余阻塞项见 [开源验证记录](../docs/opensource-release-validation.md)。

## 外部 Agent 调用

`theta tools list` 输出 `theta.tools.v1` 协议、工具 schema 和 `hostOnly` 标记。`theta tools call <name> --session <id>` 从 stdin 接收 `{ "input": {...}, "userMessage": "实际用户消息" }`，输出带 `requestId`、成功/错误与证据的 JSON。

先通过 `session_create` 创建会话，再使用宿主 `dataset_attach` 导入文件。对话与外部工具使用相同校验；`checkpoint_approve` 和 `checkpoint_reject` 是可信宿主操作，**不能注册为 LLM 可调用工具**，且须携带已经向用户展示的 `expectedContentHash`；批准还需要真实确认消息。批准后宿主直接恢复挂起的具体工具动作，不依赖模型再猜一次要执行什么。

## 验证

```bash
pnpm typecheck
pnpm test
pnpm test:workers
pnpm smoke:training   # 显式运行小型 CPU LDA，使用仓库 fixture，不调用 LLM
node scripts/check-worker-contract.mjs    # 只读核对 worker 边界 API 是否漂移
node scripts/acceptance-conversion.mjs   # 显式运行两次真实 CPU 训练：Excel（中文列名）与 JSONL 数据源
# 以下会调用已配置的对话模型；计算 gateway 被替换为模拟，不产生 embedding/训练调用：
node scripts/acceptance-conversation.mjs
# 真实训练需要明确选择此开关：
node scripts/acceptance-local.mjs .theta_agent/acceptance/local-20260905 --confirm-local-training
# 数值回归只检查梯度，不启动计算任务：
.venv/bin/python -m unittest discover -s ../src/models/tests
```

`test:workers` 与 Agent 使用相同的解释器选择规则：优先 `THETA_PYTHON`，其次 `agent/.venv`，最后才使用 `python3`，避免系统旧 Python 与实际运行环境不一致。

旧版会话与运行数据完整保留在旧工作树，未自动转换其旧训练状态。会话存储格式可以复用，但不能直接把旧 FSM run 当作新版研究记录；数据迁移需要独立验证。详细设计见 [迁移设计](../docs/architecture/agent-migration.md)。

每次训练结束，终端自动显示 worker 返回的原始结果目录并保留在终端记录中，无需等待模型回复或生成报告；重复心跳不会重复打印该目录提示。路径保持完整，便于复制。结果整理会保留原始训练目录。若旧任务只有部分矩阵、缺少 BERTopic 文档分配等必需证据，报告标记为 `incomplete`，交付已有文件与诊断页，不补造标签、不自动重训、不进入深入解读；`/reports` 和 `/open 编号` 同样可访问诊断入口。训练异常必须非零退出，BERTopic 完整交付还需文档分配和导出清单；旧版误报 completed 的任务以附加警告说明，不改写原始矩阵。

## 按动作授权，而非工作流

- Agent 自身的 DeepSeek/OpenAI 等对话推理按用户偏好直接可用，不逐轮请求确认。
- 读取配置、列模型能力、本地数据画像与准备检查不发外部付费请求，可以自主使用。
- 实际计算、云 embedding、结果整理和综合解读等需确认动作返回 `needsUser` 后，对话循环立即暂停当前工具批次。拒绝、修改、侧面提问均可继续对话。
- 训练方案默认最长运行 **12 小时（43200 秒）**；Agent 可依据数据规模和模型建议更长的有限上限并说明理由，确认卡以小时/分钟展示。用户指定的时限优先，已批准或运行中的任务不自动延长；修改方案后重新确认。本地 worker 按获批上限执行，远端仍受现有 runtime 协议限制。
- 确认卡支持同条消息拒绝并反馈，例如 `拒绝，先抽样再用 LDA`、`/deny 不使用云 embedding`。宿主先撤销旧授权并持久保存原话，再让 Agent 处理反馈；没有理由也可以直接拒绝。
- 授权保存在宿主记录中，15 分钟内可启动一次对应任务；参数、数据、目标和配置必须完全匹配。凭据不提供给 LLM。已开始任务按其时间/请求额度继续执行。
- 默认计算 gateway 与 Python 启动边界都检查授权，不能通过直接调用适配器绕过工具层提示。暂停/批准/拒绝是通用能力，不是研究步骤状态机。
- 计算子进程只得到其获准能力所需的凭据；主环境中的其他服务密钥不作为默认计算权限。已知 embedding HTTP 入口有预算校验，这不是对任意插件代码的操作系统沙箱。
- 新增外部 API/下载工具时必须接入 `EffectApprovals` 并在实际 I/O 边界验证授权；不允许靠提示词或工具自行返回“已批准”跳过检查。未注册能力保持不可执行。

验收同时包含确定性契约测试、真实对话模型配合模拟计算、用户确认后的真实 CPU LDA/STM。云 embedding 使用模拟请求测试权限与预算，未宣称完成真实收费云训练。详见 [开源验证记录](../docs/opensource-release-validation.md)。

## 持久知识与完整参数接入

`/knowledge` 查看资料；Agent 用 knowledge_list/search/read 按需读取 [完整原文](knowledge/documents/theta-models-parameters-2026-09-05.md) 和 [实际执行映射](knowledge/documents/execution-coverage.md)。目录、标签和适用问题维护在 [index.json](knowledge/index.json)，本地检索不使用 embedding。

现有 12 个模型均可由 Agent 提案：LDA、HDP、STM、BTM、ETM、CTM、DTM、NVDM、GSM、ProdLDA、BERTopic、THETA。裸参数对应统一入口；`model.*`、`trainer.*`、`fit.*`、`prepare.*`、`main.*`、`config.*`、`pipeline.*`、`word2vec.*`、`embedding.*` 对应明确接收端，支持数值、布尔、可空先验、神经网络层宽数组。先用 models_inspect 查适用范围；不支持的组合在确认前拒绝。标签/时间/协变量由计划映射，任意路径和矩阵不能绕过数据绑定。本地与更新后的分布式 worker 共用同一源码派生参数契约；部署旧 worker 的远端必须升级后才能获得这些能力。

ETM、NVDM、GSM、ProdLDA 复用已有 BaselineTrainer。NVDM 的 theta 是潜变量而非概率：保留原始负数，仅使用适用的原生可视化，不输出概率占比或 pyLDAvis；结果解释会注明。完整收录不意味着所有声明都有效：HDP 的 T、ETM 被训练器覆盖的 kl_weight、兼容占位和分布式启动器参数均在映射表注明，不伪装可调。当前扩展已做接口与路由验证，未运行全部模型真实训练、CUDA 或付费云调用。
