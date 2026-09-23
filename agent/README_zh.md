# THETA Agent

[English](README.md) | **中文**

用自然语言讨论研究目标、导入数据、选择模型、确认训练和解释结果。CLI 与网页共用同一个本地 Agent 核心；普通使用不需要账号、MySQL、Redis 或 Go 服务。

## 安装与启动

准备 Node.js 22.13+、pnpm 和 Python 计算环境。主题模型依赖见[计算引擎安装](../doc/getting-started/installation.zh.md)；统计分析需要[统计环境](docs/free-analysis.md)。

```bash
cd agent
pnpm install --frozen-lockfile
pnpm build
# 仅首次配置时复制；保留已有的私人配置。
cp .env.example .env.local
pnpm start
```

在私有 `agent/.env.local` 中填写自己的模型地址、名称和 API Key。支持 DeepSeek、OpenAI、MiniMax、GLM 及自定义 Chat Completions 兼容服务。配置优先级为进程环境、Agent 配置、仓库配置；`THETA_ENV_FILE` 可指定唯一配置文件。不要提交密钥。

也可在仓库根目录运行 `./theta`。Windows 可运行 `node agent/cli/bin/theta.mjs`。网页使用 `./theta-web start`，见[网页指南](../frontend/README_zh.md)；无需配置开发环境的用户可选择[桌面应用](../docs/desktop.zh.md)。

## 对话与数据

描述研究问题或拖入文件路径。Agent 会讨论分析单位与证据需求，再按目标选择方法。文本分析可组合描述性统计与主题建模，也可只执行其中一部分。

| 命令 | 用途 |
| --- | --- |
| `/model`、`/models` | 选择对话模型 |
| `/attach /path/to/data.csv` | 导入数据，可在路径后附加说明 |
| `/new`、`/sessions`、`/resume` | 创建、查看和恢复会话 |
| `/context`、`/contexts` | 查看和复用研究上下文 |
| `/mode free` | 切换自由统计分析 |
| `/reports`、`/open 序号` | 查看和打开报告 |
| `/knowledge` | 查看本地知识目录 |
| `/trace` | 查看最近一轮执行记录 |
| `/deny`、`/help`、`/exit` | 拒绝操作、查看帮助、退出 |

支持 CSV、TSV、TXT、Markdown、JSON、JSONL、NDJSON、Excel、Parquet、PDF、DOCX。单次本地导入上限为 200 MiB，训练规范化上限为一百万行。原文件保留，训练使用经过内容校验的托管副本；文本、时间、标签和协变量由所选列映射。

数据理解可向用户配置的对话模型发送有限文本摘录。常见邮箱、手机号和凭据样式会被遮盖，但不构成完整匿名化；敏感数据应在导入前处理。

## 计算、确认与结果

训练、云端嵌入、结果整理和深入解读需要各自的确认。卡片列出数据、模型、参数、运行上限及外部请求范围；修改方案后需要重新确认。状态查询不会重复提交训练。

本地计算默认使用 CPU；已配置 CUDA 的源码环境可显式选择 GPU。`THETA_PYTHON` 指定 Python，未设置时优先使用 `agent/.venv`，其次为 `python3`。运行 `./theta doctor --json` 检查环境。

THETA 零样本分析支持本地或云端嵌入。云端请求受已确认的服务地址、文本范围与预算约束；微调、CTM 和 BERTopic 使用兼容本地模型。缺失模型不会自动下载。绘图语言只控制图表展示，不翻译原文。

训练使用独立后台进程。退出 CLI 或停止生成不等于取消训练，取消需要明确操作。使用相同 `THETA_AGENT_HOME` 可恢复会话、数据和任务；默认保存在 `.theta_agent/`。

结果整理复用原生绘图入口，交付已有矩阵、表格、图表和 HTML 报告。缺少时间、训练历史等证据时会说明，不补造数据。深入解读保存独立报告，并区分观察与推断。原始训练结果不会被覆盖。

## 扩展与参考

- [自由分析](docs/free-analysis.md)：统计、计量、预测、生存与优化工具。
- [绘图技能](docs/data-viz-skill.md)：22 个可修改的 Python 图表模板。
- [本地嵌入](docs/local-embedding.md)、[运行环境](docs/worker-environments.md)。
- [训练进度](docs/training-progress.md)、[确认卡](docs/web-confirmation-cards.md)。
- [计算接口契约](docs/worker-interface-contract.md)、[HTTP API](docs/THETA_AGENT_API.md)。

外部 Agent 可用 `theta tools list` 查看工具，通过 `theta tools call <name> --session <id>` 调用。宿主负责展示确认卡并批准或拒绝；批准接口不能注册为模型可直接调用的工具。
