# THETA 0.3.9

[English](https://github.com/CodeSoul-co/THETA/releases/tag/desktop-v0.3.9) | **中文**

## 下载

| 平台 | 安装包 |
| --- | --- |
| 苹果芯片 Mac | [下载磁盘映像](https://github.com/CodeSoul-co/THETA/releases/download/desktop-v0.3.9/THETA-0.3.9-mac-arm64.dmg) |
| Windows x64 | [下载安装程序](https://github.com/CodeSoul-co/THETA/releases/download/desktop-v0.3.9/THETA-0.3.9-win-x64.exe) |

## 数据集划分

- 数据选择卡新增自定义划分开关：可调整训练、验证、测试比例，选择随机或顺序划分，也可分别上传三份数据，各自设置格式与数据列。
- 关闭自定义划分时，训练与验证默认按 7:3 划分，测试使用全量数据；结果页明确提示测试包含训练数据。
- 词表与模型仅使用训练记录拟合；全量、训练、验证和测试结果分类展示，提供各组指标、平均主题权重和逐条文档结果下载。
- 网页和桌面端共用这套流程，命令行智能体也支持可复现的比例划分方案。

安装包内置 Python 与计算依赖；模型权重按需下载，密钥由用户自行填写。0.3.8 起支持应用内检查更新，项目与配置保留。当前 Mac 预览版尚未正式签名，下载并校验磁盘映像后替换安装；Windows 支持确认后重启安装。

[帮助文档](https://codesoul-co.github.io/THETA/) · [问题反馈](mailto:duanzhenke@code-soul.com)
