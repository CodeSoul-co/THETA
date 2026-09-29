# THETA 0.3.13

[English](https://github.com/CodeSoul-co/THETA/releases/tag/desktop-v0.3.13) | **中文**

## 下载

| 平台 | 安装包 |
| --- | --- |
| 苹果芯片 Mac | [下载磁盘映像](https://github.com/CodeSoul-co/THETA/releases/download/desktop-v0.3.13/THETA-0.3.13-mac-arm64.dmg) |
| Windows x64 | [下载安装程序](https://github.com/CodeSoul-co/THETA/releases/download/desktop-v0.3.13/THETA-0.3.13-win-x64.exe) |

## 多模型独立训练

同时选择多个模型或主题数时，每个任务分别检查配置并启动。某一项失败不会阻断其他有效模型，成功结果与失败模型的具体原因会分别展示。所有任务结束后，总进度完成。手动模式与智能体工作流均采用此行为。

## 本地与云端嵌入

THETA 零样本、CTM 和 BERTopic 共用本地或云端嵌入配置，训练设置会采用已保存的使用方式。选择云端后不再要求无关的本地模型权重；本地编码失败会明确报错，不再忽略异常继续运行。将文本发送到云端仍需确认当前任务。THETA 微调使用本地权重。

## 文件夹与上传文件管理

选择文件夹时递归读取支持的文件并保留相对来源路径。已上传文件可勾选合并分析，也可逐个移除。表格合并支持指定正文列，文档合集保留全部提取记录。移除源文件会同步使相关派生合集失效，已有训练结果保留。取消文件大小与数量的固定限制，数据较多时仅提示；上传采用流式写入，避免服务端在内存中缓存整个文件。

安装包内置 Python 和计算依赖，不包含模型权重及个人密钥。升级会保留已有项目和设置。发布资源包含 Mac 自动更新压缩包、Windows 安装程序及更新索引，可供应用内升级使用。当前仍为未经正式签名或公证的预览版。

[帮助文档](https://codesoul-co.github.io/THETA/) · [问题反馈](mailto:duanzhenke@code-soul.com)
