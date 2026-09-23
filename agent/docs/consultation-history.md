# 右侧咨询历史

[English](consultation-history.en.md) | **中文**

右侧项目助手与左侧训练对话独立。咨询内容不再仅保存在浏览器：由 Agent 服务在
`THETA_AGENT_HOME/research.sqlite` 的 `consultations` 和 `consultation_selection` 表保存。
本地默认位置为仓库根目录 `.theta_agent/research.sqlite`，部署时须将该目录挂载为持久卷。

每条记录保存提问、回复、图表引用及原始绘图数据、标题、置顶标记、删除时间和更新时间。
选择的咨询也存库。输入框未发送草稿仍保存在浏览器，按咨询编号隔离。

## 接口

以下前缀部署为 `/api/v1/agent`，本地兼容 `/api/v3`。全部经过现有认证与写操作 CSRF 检查。
服务端从认证信息获取用户编号，不接受客户端指定 owner；同名 scope 不允许跨用户读写。
`scope` 是咨询的项目命名空间（如 `manual:<项目编号>` 或 `results:<结果范围>`），不授予训练或数据访问权限。

| 方法 | 路径 | 请求/行为 |
| --- | --- | --- |
| GET | `/consultations?scope=…` | 返回 `{ok,data:{activeId,threads}}`，包括软删除记录；置顶优先，其次按更新时间倒序 |
| POST | `/consultations?scope=…` | `{id?}`，新建并选中；相同 id 幂等，已删除 id 不会被重建 |
| PATCH | `/consultations/{id}?scope=…` | `{pinned?:boolean,select?:boolean,deleted?:boolean}`；`deleted:false` 恢复 |
| DELETE | `/consultations/{id}?scope=…` | 软删除，不清除正文；当前条被删后返回另一条可用记录，全部删除则 activeId 为空 |
| POST | `/consultations/import?scope=…` | `{threads,activeId?}`；仅插入尚不存在的编号，整批事务迁移，原本地备份保留 |
| POST | `/advisory` | 原请求增加 `consultation:{scope,id,requestId}`，使用数据库近期消息上下文，并由服务端保存本次提问/回复 |

同一 `requestId` 和请求内容重复发送不会再次调用模型；内容改变返回 409，处理中返回 202。
新问题在上一条回复未完成时返回 409。无 `consultation` 的图表解读请求维持无状态行为。
请求限制：提问 16000 字符；最多 4 张图、每图 4 个 CSV/JSON 数据源；迁移最多 1000 条咨询、
单条最多 2000 条消息，整批最多 32 MiB。超限返回错误，不静默截断已有消息。

## 恢复与删除

- 前端首次打开对应项目时自动导入旧本地历史；数据库已有记录（包括删除标记）不会被覆盖。
- 用户提问在模型调用前落库；切换页面或关闭浏览器不影响服务端保存最终回复。
- 服务重启时，未完成回复转成明确的中断提示，不伪造成功，也不自动重发付费模型请求。
- 删除、置顶发生在等待回复期间时，后续回复不会撤销它们。已删除记录可从右栏“已删除”恢复。
- 数据库不可用时显示错误，不能把本地缓存伪装成已同步；未发送草稿保留。
