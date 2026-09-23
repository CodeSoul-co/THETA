# THETA 网页对话

前端唯一源码目录为仓库根目录的 `frontend/`。安装、开发与构建均从这里执行。本仓库只维护本地工作台，不再使用旧托管网站或远端后备地址。

对话模式直接连接 `agent/web/server.ts`，与 CLI 共用 `ConversationAgent`、工具、模型配置和 SQLite 存储，没有接入旧固定训练 FSM。两种模式只在最终结果页汇合，共用 `ResearchResultView` 和右侧咨询助手。

## 本地启动

先按 [Agent 说明](../agent/README.md) 准备 Node.js ≥22.13、Agent 依赖、私有 `agent/.env.local` 和 Python 计算环境。网页依赖只需首次安装：

```bash
npm --prefix frontend ci
```

之后从仓库根目录使用统一入口（macOS / Linux，需要 `ps`）：

```bash
./theta-web start     # 编译并检查后，后台启动全部三个服务
./theta-web status    # 查看进程、服务、日志与数据目录
./theta-web restart   # 更新代码后使用；先通过编译/预检再重启
./theta-web stop      # 停网页与 API，不删除数据，不取消后台训练
./theta-web check     # 只编译 Agent 和核对 Worker 契约，不启动服务/训练
```

打开 <http://127.0.0.1:4320/workbench?mode=conversation>，顶部按钮切换“对话 / 手动”。入口从脚本位置定位当前仓库，也可在其他目录用绝对路径执行。关闭启动终端不会停止服务；电脑重启后需要重新 `start`，未安装开机自启。

| 服务 | 地址 | 源码 / 数据目录 |
| --- | --- | --- |
| 唯一前端 | `127.0.0.1:4320` | `frontend/` 的 Next 开发服务器，读取当前源码 |
| 对话 API | `127.0.0.1:4318` | 本次编译的 `agent/dist/web/server.js`，数据在 `.theta_agent/` |
| 手动 API | `127.0.0.1:4321` | 本次编译的 `agent/dist/web/manual-server.js`，数据在 `.local/manual-workbench/` |

后台管理记录和日志在 `.local/workbench/`：`supervisor.log`、`agent.log`、`manual.log`、`frontend.log`。重复 `start` 不重复启动；端口被旧服务占用时会报错，**不会按端口杀进程或接管未知服务**。核对并停止旧服务后再运行。任何受管服务异常退出时，管理进程会停止其余网页服务并记录原因，不会无限重启掩盖错误；修正原因后重新 `start`。

入口固定本机地址、以上数据目录、同源代理和开发鉴权，避免旧环境变量把页面连接到另一套服务。供应商、云嵌入、权重和各 Python runtime 私有配置仍遵循 Agent 的环境文件优先级；不覆盖 `.env.local`，不打印密钥。Python 默认为 `agent/.venv/bin/python`，也可用私有配置中的 `THETA_PYTHON` 绝对路径。Worker 直接运行当前仓库 `agent/workers/`、`trainning/worker/` 和 `src/` 的 Python 源码；本地流程不启动 `build/` 中历史 Go 二进制，不依赖旧 `theta_project`、MySQL、Redis 或 OSS。预检不调用推理/云嵌入、不下载模型、不训练。

手动服务直接调用本仓库计算能力，不创建 Agent 对话记录。本机开发默认免登录；**此入口不是生产部署命令**，不要用 `next start` 替代而期望保留开发免登录功能。

保留 `.theta_agent/` 和 `.local/manual-workbench/` 整个目录（含 SQLite、上传文件、compute、reports），里面存有此前 ProdLDA/DTM 等真实结果。启动/重启不复制项目、清库或重新提交已完成训练。浏览器继续使用 `http://127.0.0.1:4320`，才能恢复该来源保存的表单、页签及咨询草稿；`localhost` 是不同来源。重启可能中断正在生成的回复或上传，应等它们完成；已提交后台训练独立运行，查询恢复不会重复训练。

只开发单个服务时，仍可分别运行 `npm --prefix agent run web:api`、`npm --prefix agent run web:manual`、`npm --prefix frontend run dev`，不要与统一入口同时占用端口。单独运行时才需自行配置前端 `.env.local`（参考 `.env.example`，不要覆盖已有配置）。

生产模式不能启动这个免登录服务，前端生产路由也不会使用它。远程 Host/Origin 被拒绝；设置 `THETA_LOCAL_AUTH_ENABLED=false` 可关闭本地路由。请保持 `NEXT_PUBLIC_LOCAL_NO_AUTH=false`，不要使用浏览器端假令牌。模型设置共用 `agent/.env.local`，修改后重启 API。手动服务默认只运行本地模型，不会隐式调用付费云端嵌入。

手动上传需收到有效文件编号后才显示配置面板；服务错误不能跳过或伪装为“已完成”。本地模型权重或依赖未就绪时会明确拒绝训练，不产生假的任务。

分析配置中的“绘图语言”只影响图表标题、坐标轴和图例，不翻译原文，也不选择分词器。文本按文字体系自动分段，保留同一行里的混合语言；内置中、英、德、法、西、意、葡、俄、日、韩语及通用停用词表。中文使用 jieba，其他文字采用对应文字段或 Unicode 单词边界；日文分词仍为基础分词，不等同专用形态分析器。

“文本处理与停用词”可上传 UTF-8 TXT（每行一个词，`#` 注释，最多 512 KB / 20000 个词），自定义表替换本次分析的内置表，点击“恢复默认”即可恢复自动加载。导出下载直接读取 `src/models/resources/stopwords/` 中实际使用的全部内置词表，不维护前端副本。自定义词表按内容写入不可变训练参数 `text.stopwords`，历史任务不会随之后上传的词表变化。未上传时无需操作。

API 默认只监听 `127.0.0.1:4318`，存储默认使用仓库根目录 `.theta_agent/`（可通过 `THETA_AGENT_HOME` 修改）。网页经同源 `/api/v3` 代理访问。上传文件后直接让 Agent 理解数据；训练、结果整理和深入解读分别由当前确认卡授权。停止生成会中断本轮推理，后台计算需另行明确取消。

## 工作台模式边界

- **对话模式**由项目下的独立对话记录组织研究；Agent 可以在确认机制约束下准备训练、查询进度、读取结果和执行获批解读。
- **手动模式**由项目中心的项目卡片组织数据、参数、训练状态与结果，不创建或恢复项目对话记录。
- 两种模式的结果页共用右侧只读咨询面板：通过 Agent 无状态 `/api/v3/advisory` 回答项目问题，或读取已引用图表的 CSV/JSON 数据；不发送图片像素。接口不创建训练对话对象，浏览器按项目/任务分别持久化咨询文本。
- 咨询接口没有训练工具，不会创建、修改或取消训练；训练操作请返回原工作台。结果页保留模型、页签和筛选状态，并同步训练完成后发布的图表；未确认“整理结果”时提示返回对话确认。

## 验证与范围

界面固定使用简体中文。`lib/presentation.ts` 统一管理系统状态、训练阶段、参数、图表及数据表的展示名称；原始文件名、接口字段、模型标识和用户输入不重命名。新增结果类型时应补充中文名称及对应测试，不在各组件重复维护词典。

```bash
npm --prefix agent test
pnpm --dir frontend build
```

卡片设计与专项测试见 [确认卡交互说明](../agent/docs/web-confirmation-cards.md)。浏览器验收见 [开源验证记录](../docs/opensource-release-validation.md)。当前为本机单用户开发入口，不提供生产多租户隔离。
