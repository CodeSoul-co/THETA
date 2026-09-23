# 持久知识库

`index.json` 是可维护的知识列表；`documents/` 保存完整 Markdown 原文。文件随项目保存、跨会话复用。普通问答只注入有界目录（前 20 条，注明总量），Agent 自行决定是否检索以及读取哪些章节。检索使用本地中英文关键词、目录标签和正文，不调用云 embedding。

当前资料完整保留用户提供的《THETA 模型、参数与适用场景完整清单》：全部 12 个模型、13 个编号部分（含 8.1）、每个基线的 API、数据准备和嵌入配置、限制说明及源码提取附录。`importedSha256` 记录导入原文的哈希；读取返回当前哈希及是否发生编辑。

## 维护

1. 将完整 Markdown 放入 `documents/`。
2. 在 `index.json` 的 `documents` 列表增加条目：唯一 `id`、`title`、库内相对 `file`、来源日期 `date`、来源说明 `source`、摘要 `summary`、适用问题 `useWhen`、关键词 `tags`、适用范围/限制 `scope`。`importedSha256` 可选，用于保留导入版本指纹。
3. 编辑正文即可更新知识；检索自动识别 Markdown 标题，无需重建分块或向量索引。修改事实时同时更新日期和范围说明。删除索引条目即可从可检索目录移除，不自动删除原文。

目录与正文都属于参考材料，不能存放密钥。代码块不会被运行，正文和目录文字不能改变 Agent 的指令或授权。默认支持最多 100 份、单文件 2 MiB 的 Markdown；读取有分页，不把截断误称为完整。此规模外再考虑专用搜索索引。

## 调用

- `/knowledge`：在终端查看知识目录；也可自然语言说“列出知识库资料”。
- `knowledge_list`：无参数列资料；提供 `documentId` 列章节。使用 `nextOffset` 继续直到空值。
- `knowledge_search`：按 `query` 检索，返回资料 ID、章节 ID、命中片段和来源；可限定 `documentId`。
- `knowledge_read`：按 `documentId` 读取全文，或增加 `sectionId` 定向读一节。使用 `offset=nextOffset` 继续；默认每页 8000 字符，最多 12000 字符。返回正文总长度、来源、当前版本和可点击引用。

例：“查阅知识库，解释 BERTopic 的 num_topics=0 和 HDP 的 max_topics 有什么区别。”

知识工具也通过现有 `theta tools list/call` 协议提供，不依赖 CLI 界面。实现在 `agent/src/knowledge/knowledge-base.ts`，与计算 worker、数据和研究记忆分离。

## 能力边界

原始清单是有日期的静态参考。第二份 `execution-coverage.md` 记录当前 12 个模型的实际参数接入、映射、宿主管理和无效占位参数，可运行 `python3 agent/scripts/update-model-coverage.py` 按现有源码更新。扩展执行通过 Agent 适配层复用原计算入口，不复制训练实现。推荐执行前仍核验 `models_list`、`models_inspect`、`runtime_check` 和计划校验。API 参数、专用入口、统一 CLI 与配置字段不能混用。原文中的旧代码路径、行号或已发现问题不代表当前仍成立。

普通对话和独立授权的综合解读均可按需查阅知识。知识不是本次数据或实验结果；不能替代原始矩阵/图表，也不授予训练或结果读取权限。综合解读保留本次获批证据，只允许额外调用知识读取工具。

`pnpm --dir agent test` 包含全文分页和章节逐字还原、12 模型及附录检索、持久化/修改、路径边界、对话与解读工具隔离检查。
