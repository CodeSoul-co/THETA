# THETA 0.3.5 · 桌面预览版

[English](https://github.com/CodeSoul-co/THETA/blob/main/docs/releases/desktop-v0.3.5.md) | **中文**

THETA 提供本地 **Web 工作台、CLI Agent 和桌面应用**，并可通过工作流技能接入其他 Agent。

## 下载与安装

| 系统 | 下载 | 安装方法 |
| --- | --- | --- |
| macOS Apple Silicon（M 系列） | [DMG 安装包](https://github.com/CodeSoul-co/THETA/releases/download/desktop-v0.3.5/THETA-0.3.5-mac-arm64.dmg) | 打开 DMG，将 THETA 拖入「应用程序」 |
| Windows x64 | [EXE 安装程序](https://github.com/CodeSoul-co/THETA/releases/download/desktop-v0.3.5/THETA-0.3.5-win-x64.exe) | 运行安装程序 |

Mac ZIP 是备用应用压缩包，常规安装请使用 DMG。[SHA256SUMS.txt](https://github.com/CodeSoul-co/THETA/releases/download/desktop-v0.3.5/SHA256SUMS.txt) 提供各安装资源的校验值。源码压缩包不包含桌面运行环境。

安装包内置 Python 3.12 与计算依赖，无需另装 Python、Node.js 或 Conda。应用名称为 THETA，使用布偶猫图标。

## 本次更新

Windows 默认使用兼容的 NVIDIA CUDA 显卡加速支持的本地模型和嵌入计算。安装包内置支持 CUDA 12.8 的 PyTorch，无显卡电脑仍可使用 CPU。无需另装 Python 或 CUDA Toolkit；显卡加速需要兼容的 NVIDIA 驱动。本版本中的 AMD、Intel 显卡使用 CPU，macOS 继续使用 CPU。

训练前实际检查 GPU 分配与运算，不可用则直接选择 CPU。自动模式下遇到 CUDA 错误或显存不足时，会保留失败日志、清理本任务临时结果，并在 CPU 上重新执行一次；原有时间上限、取消请求及云端请求预算继续生效。数据错误和超时不会自动重试。

执行日志显示所用设备，并在回退到 CPU 重新执行时提醒用户。LDA、BTM、HDP、STM 继续使用 CPU；显式指定的 CPU 或 CUDA 设备会被保留。

GPU 依赖使 Windows 安装包体积增大。升级前请退出 THETA：Mac 在「应用程序」中替换旧版，Windows 运行新版 EXE。项目与配置会保留，更新需要手动下载安装。

## 模型与数据

进入设置后填写自己的对话模型地址、模型名称和 API Key，点击「完成」即可保存。已保存的密钥以星号显示。本地嵌入使用用户选择的兼容模型目录，默认建议 Qwen3-Embedding-0.6B；云端嵌入支持自行配置 API，默认预设为 GLM embedding-3。

安装包不含用户密钥、私人数据或模型权重。LDA 等传统模型无需神经网络权重，本地嵌入模型需要另行下载。云端嵌入目前支持 THETA 零样本模式，需确认发送文本的范围与请求预算。

项目和结果保存在本机。升级安装保留应用数据；桌面版与源码版的数据目录各自独立。

## 其他入口

- **Web 工作台**：完成源码配置后运行 `./theta-web start`，访问 `http://127.0.0.1:4320/workbench`。
- **CLI Agent**：完成 Agent 配置后运行 `./theta`。
- **工作流技能**：在兼容的 Agent 中使用 `skills/theta-workflow/SKILL.md`。

详细步骤见[中文首页](https://github.com/CodeSoul-co/THETA/blob/main/README_zh.md)和[桌面指南](https://github.com/CodeSoul-co/THETA/blob/main/docs/desktop.zh.md)。

## 预览版要求

本版本未正式签名或公证，操作系统可能显示安全提示。安装包支持 macOS arm64 与 Windows x64，不提供 Intel Mac 或 Windows ARM 原生版本。更新需要手动下载安装。
