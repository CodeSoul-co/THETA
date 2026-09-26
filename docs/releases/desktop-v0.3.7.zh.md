# THETA 0.3.7

[English](https://github.com/CodeSoul-co/THETA/blob/main/docs/releases/desktop-v0.3.7.md) | **中文**

## 下载

| 平台 | 安装包 |
| --- | --- |
| macOS Apple Silicon（M 系列） | [DMG 安装包](https://github.com/CodeSoul-co/THETA/releases/download/desktop-v0.3.7/THETA-0.3.7-mac-arm64.dmg) |
| Windows x64 | [EXE 安装程序](https://github.com/CodeSoul-co/THETA/releases/download/desktop-v0.3.7/THETA-0.3.7-win-x64.exe) |

已内置 Python 和计算依赖，模型权重按需另行下载，密钥由用户在设置中填写。覆盖安装前请先退出 THETA，原有项目和配置会保留。

## 修复内容

- 删除手动项目后释放名称，刷新后不会从保留的数据与结果重新出现；仅在服务确认删除后显示成功。原始数据与结果仍保留在本机。
- 手动与对话模式在文件添加、上传成功后提供明确提示。
- 兼容 Windows 的文件名排序规则，修复手动结果转入对话时被误判为校验失败的问题，同时保留文件完整性校验。
- Windows 训练导出模型与读取结果时支持长路径及中文目录，避免在模型保存阶段因路径长度失败。
- 支持先取消全部模型再选择其他模型；未选择模型时继续操作会提示，不会启动任务，重新选择后正常显示模型参数。

网页与桌面共用本次更新的工作台和本地服务。源码用户更新仓库后重启网页工作台，桌面用户需要安装本版才能收到修复。

本版仍为未正式签名的预览版。Mac ZIP 为备用应用压缩包；安装资源均提供 [SHA-256 校验文件](https://github.com/CodeSoul-co/THETA/releases/download/desktop-v0.3.7/SHA256SUMS.txt)。

使用说明见[帮助文档](https://codesoul-co.github.io/THETA/zh/)。问题请反馈至 [duanzhenke@code-soul.com](mailto:duanzhenke@code-soul.com)。
