# THETA 桌面应用

[English](desktop.md) | **中文**

桌面版复用 CLI Agent、网页工作台与 Python 计算引擎，提供对话和手动两种模式。安装包内置 Python 3.12 与 CPU 计算依赖，用户无需安装 Python、Node.js 或 Conda。模型权重不在安装包内。

## 使用

首版构建目标为 **macOS Apple Silicon（arm64）** 和 **Windows x64**。从 [GitHub Releases](https://github.com/CodeSoul-co/THETA/releases/tag/desktop-v0.3.1) 下载：Mac 使用 `THETA-0.3.1-mac-arm64.dmg`，Windows 使用 `THETA-0.3.1-win-x64.exe`。Mac 打开 DMG 后将 THETA 拖入「应用程序」；Windows 运行 EXE 安装程序。当前为未正式签名／公证的预览版，系统可能显示安全提示。暂不提供 Intel Mac 或 Windows ARM 原生版本。

发行页提供 `SHA256SUMS.txt`，用于核对下载文件完整性。macOS 可运行 `shasum -a 256 文件名.dmg`；Windows PowerShell 可运行 `Get-FileHash 文件名.exe -Algorithm SHA256`，与校验文件对应行比较。

应用启动后直接进入工作台。打开左下角「设置」：

- **模型 API**：填写对话模型的供应商、Base URL、模型名称和 API Key。
- **Embedding 模型 → 本地模型**：默认建议 `Qwen/Qwen3-Embedding-0.6B`。用户可填写名称并选择兼容模型的完整本地目录（包含 config.json、分词器和权重）；名称仅用于标识，实际加载所选目录。CTM / BERTopic 可另选 Sentence Transformers 兼容模型。
- **Embedding 模型 → 云端 Embedding API**：默认 GLM / 智谱，Base URL 为 `https://open.bigmodel.cn/api/paas/v4`，模型为 `embedding-3`。可修改地址、模型和向量维度，并单独保存 Embedding API Key；也可使用 OpenAI 兼容服务。

可以稍后再配置模型。LDA 等传统算法不需要下载神经网络权重；本地 Embedding 权重通过设置中的链接自行下载。云端 Embedding 当前适用于 THETA zero-shot，执行具体任务时仍需确认发送文本及请求预算；THETA 微调和 CTM / BERTopic 使用本地模型。

保存配置后新任务立即生效，不需要重启工作台。正在运行的训练使用启动时的配置。密钥由 Electron safeStorage 使用操作系统能力加密，保存在应用数据目录，页面不回显密钥。

安装包不预置 API Key，首次使用由用户自行填写。打包前会拒绝私有环境文件、用户设置和已知本地密钥，并核对布偶猫 Logo；应用启动时也不会继承开发机的模型密钥环境变量。重新安装后仍显示的已保存配置来自本机应用数据目录，不是安装包内置配置。

## 数据与日志

应用菜单可打开数据和日志目录，也可重新启动本地服务。

- macOS：`~/Library/Application Support/THETA`
- Windows：`%APPDATA%/THETA`

配置、项目、上传文件、结果和缓存与安装目录分离。重新打包、升级或卸载不会主动删除这些数据。桌面版使用独立数据目录，不自动迁移源码版的 `.theta_agent` 或 `.local/manual-workbench`。

本地服务只监听 loopback，使用每次启动生成的访问令牌，默认工作台端口 14320（冲突时自动选择空闲端口）。退出应用关闭其自身服务；已提交的训练遵循原有 Worker 生命周期，不会重复提交。

## 从源码构建

构建机需要 Node.js 22.13+、pnpm 和 uv。必须在目标操作系统和架构上准备 Python 依赖，不能把 macOS 的 runtime 复制到 Windows。

```sh
pnpm --dir agent install --frozen-lockfile
npm --prefix frontend ci
npm --prefix desktop ci
npm --prefix desktop run prepare:python
npm --prefix desktop run prepare:runtime
npm --prefix desktop test
npm --prefix desktop run smoke
npm --prefix desktop run start
npm --prefix desktop run dist
```

`prepare:python` 下载独立 Python，生成／使用平台依赖锁并安装全部 CPU 计算依赖；macOS 会检查和修复原生库路径。不会下载模型权重。安装环境需要数 GB 磁盘空间，构建机还需预留打包临时空间。

`prepare:runtime` 编译 Agent 与 Next.js standalone，并只复制运行需要的引擎、配置模板及技能。不会复制本地私密环境文件、上传数据、训练结果或模型目录。

输出在 `desktop/release`：macOS `.app`、`.dmg` 和 `.zip`，Windows NSIS `.exe` 安装程序。`desktop/runtime` 和 `desktop/release` 不提交到 Git。

`.github/workflows/desktop.yml` 支持手动选择平台构建；`desktop-v*` 标签构建两种平台。工作流验证源码运行和打包后的应用，再上传安装包。标签构建全部通过后，会发布带 SHA-256 校验文件的 GitHub 预览版 Release；手动构建只上传 Artifacts。应用不自动更新，下载新版安装程序升级即可。

