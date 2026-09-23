# 本地使用指南

[English](opensource-agent.md) | **中文**

THETA 提供网页工作台、CLI Agent 和桌面应用。本机使用无需商业账号、Go 控制面、MySQL 或 Redis。项目和结果保存在本机，用户可自行配置云端对话模型与 Embedding API。

## 桌面应用

从 [GitHub Releases](https://github.com/CodeSoul-co/THETA/releases/tag/desktop-v0.3.0) 下载 Mac DMG 或 Windows EXE。Python 与 CPU 计算依赖已内置；模型权重按需下载，密钥由用户在设置中填写。详见[桌面指南](desktop.zh.md)。

## 源码版

准备 Node.js 22.13+、pnpm 与 Python 计算环境，依赖见[Agent 指南](../agent/README_zh.md)和[引擎安装](../doc/getting-started/installation.zh.md)。在仓库根目录执行：

```sh
pnpm --dir agent install --frozen-lockfile
npm --prefix agent run build
npm --prefix frontend ci
# 仅首次配置时复制，保留已有配置。
cp agent/.env.example agent/.env.local
```

在私有 `agent/.env.local` 中填写自己的对话模型配置，然后选择入口：

```sh
./theta
./theta-web start
```

网页地址为 `http://127.0.0.1:4320/workbench?mode=conversation`，顶部切换对话和手动模式。使用 `./theta-web status` 查看状态、`restart` 更新并重启、`stop` 停止服务。CLI 的 Windows 入口为 `node agent/cli/bin/theta.mjs`。

## 数据与运行范围

后台服务只监听本机地址。源码版使用 `.theta_agent/` 和 `.local/manual-workbench/`；桌面版使用操作系统的 THETA 应用数据目录。保留这些目录即可保留对应项目、上传数据和结果。

刷新页面不会取消已经受理的计算；服务中断后先检查任务记录，避免重复提交。实际训练、云端嵌入和结果解读通过明确的确认操作启动。模型权重、个人数据和密钥不随安装包分发。

详细说明：[CLI Agent](../agent/README_zh.md)、[网页工作台](../frontend/README_zh.md)、[桌面应用](desktop.zh.md)。
