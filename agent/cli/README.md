# CLI 应用边界

本目录独立承载所有 CLI 专用代码，不设 `src/` 子目录。

| 目录 | 内容 |
| --- | --- |
| `bin/` | 启动入口、参数解析、进程退出码 |
| `shell/` | 持续输入、斜杠命令、历史与会话选择、中断交互 |
| `ui/` | 终端布局、文本渲染、进度、工具调用展示、确认卡片 |
| `commands/` | `tools list/call` 等非交互 CLI 命令适配 |
| `shell/*.test.ts`、`ui/*.test.ts` | 与实现相邻的输入输出、命令交互和界面测试 |

CLI 调用 `../src/` 中的核心接口，消费其文本、工具、进度和确认事件；确认动作交回宿主授权边界。不要把模型推理、工具 schema、会话数据库或训练调度复制到这里。

原 `agent-shell.ts`、`terminal-view.ts`、`tools-cli.ts` 及其界面/交互测试已迁入本目录；`conversation-agent.ts`、`session-store.ts`、`tool-catalog.ts` 属于共享核心。`bin/theta.mjs` 是可执行入口，仓库根目录 `./theta` 与 `agent/` 内的 `pnpm start` 均可启动。
