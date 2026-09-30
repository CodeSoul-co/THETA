# 使用方式

[English](https://github.com/CodeSoul-co/THETA/blob/main/doc/getting-started/interfaces.md) | **中文**

无需配置开发环境可使用桌面应用；希望在本机浏览器中操作可使用网页工作台；习惯终端可使用命令行 Agent。三种入口共用分析引擎，项目保存在本机。

## 桌面应用

从 [GitHub 发布页](https://github.com/CodeSoul-co/THETA/releases)下载安装包。Apple Silicon Mac 使用 DMG，Windows x64 使用 EXE。Mac 打开 DMG 后将 THETA 拖入「应用程序」；Windows 运行安装程序。

安装包内置 Python 与 CPU 计算依赖，无需另装 Python、Node.js 或 Conda，不包含模型权重和用户密钥。当前为未正式签名的预览版，系统可能显示安全提示。

进入 THETA 后，在设置中填写对话模型地址、模型名称与密钥。本地嵌入可选择兼容模型目录，默认建议 Qwen3-Embedding-0.6B；云端嵌入可填写服务地址和密钥，默认预设为 GLM embedding-3。THETA 零样本模式、CTM 和 BERTopic 支持云端嵌入。THETA 微调需要兼容的本地权重。

启动无需输入钥匙串密码。密钥通过本机用户所有的加密文件保存；从旧版钥匙串存储升级时，需重新输入一次密钥。

项目和结果保存在本机。Mac 使用 `~/Library/Application Support/THETA`。Windows 新安装优先使用安装目录内可写的 `THETA-data` 文件夹，必要时回退到 `%APPDATA%/THETA`。更新保留已有数据位置。

## 技能管理

在对话模式中，点击「新建项目」上方的「技能管理」。可导入文件夹、ZIP、tar.gz 或 SKILL.md，也可填写 GitHub 仓库或技能文件夹地址下载。支持启用、停用、导出和删除已导入的技能。内置绘图模板和 Agent 按用户要求下载的技能显示在同一列表中。Agent 使用前会读取已启用技能的说明；导入不会执行脚本或安装依赖。

## 网页工作台

源码版需安装 Node.js 22.13+、pnpm 和 [Python 计算依赖](installation.zh.md)。仅首次配置时从示例创建私有的 `agent/.env.local`，填写模型地址和密钥；保留已有配置。

在仓库根目录执行：

```sh
pnpm --dir agent install --frozen-lockfile
npm --prefix frontend ci
./theta-web start
```

打开 `http://127.0.0.1:4320/workbench`，在顶部选择对话或手动模式。使用 `./theta-web status` 查看服务状态，使用 `./theta-web stop` 停止服务。

启动器运行本地服务，项目保存在 `.theta_agent/` 和 `.local/manual-workbench/`。更新源码时保留这些目录。

## 命令行 Agent

完成源码配置后，在仓库根目录运行 `./theta`。Windows 使用 `node agent/cli/bin/theta.mjs`。描述研究问题并附加数据文件，检查模型配置，确认方案后开始计算。

使用 `/help` 查看命令，`/sessions` 查找会话，`/resume` 继续会话。配置与命令详情见[命令行指南](https://github.com/CodeSoul-co/THETA/blob/main/agent/README_zh.md)。

## 工作流技能

兼容的 Agent 可读取 `skills/theta-workflow/SKILL.md`，引导数据准备、模型选择、执行确认和结果解释。详见[工作流说明](https://github.com/CodeSoul-co/THETA/tree/main/skills/theta-workflow)。
