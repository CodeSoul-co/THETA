# 对话中的代码预处理

[English](conversation-preprocessing.en.md) | **中文**

Agent 可在数据理解之后、训练配置之前调用 `dataset_preprocess`。这是受限 pandas 代码解释器，不是任意 Python 主机执行权限；不依赖 Docker，也不开放 shell、文件读写、联网、安装包、import、循环、lambda 或任意回调。文件内容只作为数据，不作为指令或代码。

## Agent 工具 / Worker 契约

工具参数：`{datasetRef?, purpose, code, textColumn, timeColumn?}`。宿主将已授权数据对象、`home`、`uploadDir`、会话进度 ID 补入 `dataset.preprocess` Worker 请求；模型不能提供主机路径。

```python
df["正文"] = df["正文"].fillna("").astype("str").str.strip()
df = df[df["正文"].str.len() > 0]
df["时间"] = pd.to_datetime(df["发布时间"], errors="coerce").dt.strftime("%Y-%m-%d")
```

每条语句是 `df = ...` 或 `df["列名"] = ...`；支持列选择、布尔筛选、比较及 Series 相加。

- DataFrame：`dropna`、`drop_duplicates`、`fillna`、`rename`、`reset_index`、`sort_values`、`copy`、`assign`。
- Series：`fillna`、`astype`、`isin`、`notna`、`isna`、`clip`、`where`、`replace`。
- 字符串：`strip/lstrip/rstrip`、`lower/upper`、`replace`、`slice`、`normalize`、`contains`、`len`。
- 日期：`pd.to_datetime`、`.dt.strftime`、`.dt.floor`；数值：`pd.to_numeric`。

所有现有上传格式经统一读取器加载。单次最多 20 万行、500 列、80 条赋值、16000 字符代码，运行沿用 Worker 超时，不静默截断为画像样本。空结果、正文全部为空、指定列不存在、重复列名、时间列无法解析都会失败，不切换训练输入。语义性合并、删行、去重必须有用户目标依据，有歧义需询问，不编造时间或标签。

成功回执含 `sourceDatasetRef`、`datasetRef`、`inputRows`、`outputRows`、`removedRows`、列名、正文空值数、数据画像及产物：`预处理数据.csv`、`预处理代码.py`、`预处理校验.json`。宿主保留原数据，新建研究记录并绑定派生版本，后续 `training_configure` 使用新版本及其列；旧研究、旧训练不变。存在待确认卡/训练时不允许切换，处理成功也不代表训练获批。

代码、校验报告、派生 CSV 通过当前会话的标准受控产物下载链接交付；代码在安装了本仓库依赖的环境中可复现：在 `agent/` 下运行 `python 预处理代码.py 原始文件路径 输出.csv`。

## 用户消息计数与引用

`GET /api/v3/runs/{runId}`、会话列表和现有 SSE 快照返回 `userMessageCount`，从持久化记录统计用户真实输入（包含明确发送后失败的轮次），排除助手、工具、后台监控和确认卡点击。`messageCount` 是兼容别名，现在也表示用户发送数。重复 requestId 不增加计数；旧记录缺来源标记时按已存角色和已知确认格式兼容统计。

`datasetRefs` 为会话持久数据绑定；输入框引用单独保存，首次发送后收起或手动移除不会删除项目数据。后续消息沿用已绑定数据，无需重复引用。OpenAPI 已同步以上字段。
