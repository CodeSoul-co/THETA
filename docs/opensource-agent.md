# 开源 Agent、CLI 与工作台

开源发布包含 `agent/` 的设计、CLI、HTTP 后端、统计 worker、内置技能，以及 `frontend/` 现有工作台、`src/models/` 计算引擎和其所需的 `trainning/worker/` Python 适配器。普通本地使用不需要商业账号服务、Go 控制面、MySQL 或 Redis。

## 启动

安装 Node.js ≥22.13、pnpm 和 Python 3.11–3.13。先在两个目录分别安装依赖：

```sh
cd agent
pnpm install --frozen-lockfile
npm run build
cp .env.example .env.local
# 在 .env.local 填写自己的对话模型配置；不要提交这个文件。
```

随后安装前端依赖并在仓库根目录启动：

```sh
npm --prefix frontend ci
./theta-web start
./theta-web status
```

打开 `http://127.0.0.1:4320/workbench?mode=conversation`，顶部可切换对话和手动工作台。`./theta-web restart` 更新编译并重启，`./theta-web stop` 停止服务但保留数据。启动器同时运行前端 4320、对话 API 4318 和手动 API 4321。开源版不登录、不生成账号；手动数据接口仅在本机开发模式开放，账号接口继续禁用。

统计环境按 [自由分析指南](../agent/docs/free-analysis.md) 安装；主题计算按原引擎要求安装 Python 依赖与模型。缺少环境或 API key 时会说明具体问题。完整启动说明见 [前端文档](../frontend/README.md)。

只使用 CLI：在仓库根目录运行 `./theta`；`./theta --help`、`./theta doctor --json`、`./theta tools list` 可查看使用方式、运行环境与工具。CLI 与网页共享 Agent 核心，`THETA_AGENT_HOME` 指向同一目录时共享本地研究记录。

## 版本边界

`frontend/lib/edition.json` 是随发行版提交的构建配置。开源快照值为 `opensource`；dev/main 保持 `hosted`，免登录不会由浏览器参数、localStorage 或环境误操作启用。开源 AuthProvider 保持 `user=null`，工作台依据发行版开放访问，而不是伪造已登录身份。账号相关代理在开源版禁用。

这是个人本地工作台，没有租户隔离或公开多用户服务承诺。后台仅绑定 loopback；生产构建需要明确设置 `THETA_AGENT_API_URL`。发布远端多用户服务需要部署者自行实现身份和数据隔离。

项目、对话、任务回执和分析计划存于 Agent home；报告、复现脚本和技能工程保存在其子目录。刷新页面不取消已受理任务；服务中断后先核查记录与已有结果，避免重复计算。详见 [自由模式与恢复](../agent/docs/free-analysis.md)、[确认卡](../agent/docs/web-confirmation-cards.md)。

## 内置绘图技能

已内置 [Data Viz Skill](https://github.com/AdamsukS/data-viz-skill)，含固定来源、MIT 许可证、模板代码和预览，见 [推荐说明、工具与运行方法](../agent/docs/data-viz-skill.md)。模板工程准备与真正渲染分开记账，示例数据不会被称为用户结果。

## 发布方式

开源分支从公开历史上接收允许目录的代码快照；不合并商业分支历史。私有配置、本地数据库、上传数据、模型权重、部署凭据和内部验收记录不进入快照。两种发行版共享业务代码，只对明确列出的发行版配置和说明作差异处理。回到 dev 通过集成测试后再合并 main。
