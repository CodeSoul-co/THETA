# THETA 0.3.0 · Desktop Preview

THETA 现支持 **Web 工作台、CLI Agent、Mac / Windows 桌面端**，另提供可供其他 Agent 使用的 THETA Workflow Skill。

## 下载安装

| 系统 | 文件 | 安装方法 |
| --- | --- | --- |
| macOS Apple Silicon（M 系列） | `THETA-0.3.0-mac-arm64.dmg` | 打开 DMG，将 THETA 拖入「应用程序」 |
| Windows x64 | `THETA-0.3.0-win-x64.exe` | 运行安装程序 |

macOS ZIP 是备用应用压缩包，常规安装请使用 DMG。`SHA256SUMS.txt` 提供安装资源的 SHA-256 校验值。源码压缩包不包含桌面运行环境。

## 本次版本

- 桌面安装包内置 Python 3.12 与 CPU 计算依赖，无需另装 Python、Node.js 或 Conda。
- 应用名称统一为 THETA，使用布偶猫 Logo，启动后直接进入工作台。
- Web、CLI Agent 和桌面端使用同一套本地 Agent 与计算引擎，支持对话和手动分析。
- 修复手动模式 Excel 数据列预览被代理拦截的问题，并验证上传、列预览及预处理状态的完整链路。
- 修复对话模式及手动模式右侧咨询输入框的重复焦点边框。
- 工作台统一连接本地服务，移除旧线上后端回退和托管部署入口。

## 模型与数据

进入设置后填写自己的对话模型 API 地址、模型名称和 API Key。Embedding 支持自选本地兼容模型目录，默认建议 Qwen3-Embedding-0.6B；也支持自行配置云端 API，默认预设为 GLM embedding-3。

安装包不包含用户密钥、私人数据或模型权重。LDA 等传统模型无需神经网络权重；需要本地 Embedding 的模型请另行下载并配置。云端 Embedding 目前支持 THETA zero-shot，发送文本前需在任务中确认。

项目和结果保存在本机。升级安装保留应用数据；桌面端与源码版的数据目录各自独立。

## 其他使用方式

- **Web**：完成源码环境配置后运行 `./theta-web start`，访问 `http://127.0.0.1:4320/workbench`。
- **CLI Agent**：完成 Agent 环境配置后运行 `./theta`。
- **Skill**：使用仓库中的 `skills/theta-workflow/SKILL.md`。

详细步骤见仓库 [README](https://github.com/CodeSoul-co/THETA#choose-how-to-use-theta)、[中文说明](https://github.com/CodeSoul-co/THETA/blob/main/README_zh.md)和[桌面指南](https://github.com/CodeSoul-co/THETA/blob/main/docs/desktop.md)。

## 预览版范围

本版本未正式签名／公证，操作系统可能显示安全提示。提供 macOS arm64 与 Windows x64，不提供 Intel Mac 或 Windows ARM 原生安装包，也不启用自动更新。构建流程验证两种平台的源码运行与打包后应用，不调用付费模型 API。

---

**English:** Download the DMG for Apple Silicon Macs or the EXE installer for Windows x64. Python 3.12 and CPU dependencies are included; model weights and user API keys are not. Configure your own model endpoints in Settings. The repository also provides a local Web workbench (`./theta-web start`), CLI Agent (`./theta`), and an optional workflow Skill. This unsigned desktop preview includes local Excel preview fixes and removes duplicate focus outlines in both chat composers. See `SHA256SUMS.txt` for file integrity checks.
