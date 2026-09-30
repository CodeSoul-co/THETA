# THETA 0.3.14

[English](https://github.com/CodeSoul-co/THETA/blob/main/docs/releases/desktop-v0.3.14.md) | **中文**

| 平台 | 安装包 |
| --- | --- |
| macOS Apple Silicon | [下载 DMG](https://github.com/CodeSoul-co/THETA/releases/download/desktop-v0.3.14/THETA-0.3.14-mac-arm64.dmg) |
| Windows x64 | [下载 EXE](https://github.com/CodeSoul-co/THETA/releases/download/desktop-v0.3.14/THETA-0.3.14-win-x64.exe) |

启动不再要求钥匙串密码。API Key 使用本机用户专属加密存储；从旧版钥匙串存储升级时，需要重新输入一次 API Key。Windows 使用普通用户权限运行，安装程序不再提供权限提升。未签名应用的系统提示仍由操作系统决定，当前预览版未正式签名或公证。

训练提交、进度、取消和结果读取统一通过 worker HTTP API。桌面 worker 服务与界面、Agent 服务分别运行；Web 和 CLI 使用同一协议。预留 HTTPS 远程 worker 地址与令牌配置；远程部署还需提供 worker 可访问的数据与结果存储。

对话模式左侧增加「技能管理」。支持导入文件夹、ZIP、tar.gz 或 SKILL.md，从 GitHub 仓库或技能文件夹下载安装，启用／停用、导出和删除导入的技能。内置 Data Viz 科学绘图模板也显示在同一列表中。Agent 按用户要求下载的技能写入同一持久目录，并显示在管理界面。导入技能不会自动执行脚本或安装依赖。

大数据评估按批统计主题涉及词的实际出现次数，困惑度计算避免复制完整文档与词表概率矩阵。评估日志显示正在计算的具体指标，不再停留在最后一轮训练进度。评估继续使用全量实际记录，不抽样替代。

安装包内置 Python 和计算依赖，不包含用户 API Key 或模型权重。升级保留已有项目。

[帮助文档](https://codesoul-co.github.io/THETA/) · [反馈问题](mailto:duanzhenke@code-soul.com)
