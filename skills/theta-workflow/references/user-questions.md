# THETA User Questions

Use this reference when the skill needs to ask the user for missing task details. Match the user's language. Ask only what is needed for the next decision; do not ask for secrets in chat.

## 中文询问模板

### 仓库缺失确认

```text
我没有找到可用的 THETA 仓库。使用 THETA Skill 时，第一步需要先准备仓库，然后才能配置环境、检查数据或生成训练命令。

我准备执行以下操作：

操作：
克隆 THETA 仓库到本地。

命令：
git clone https://github.com/CodeSoul-co/THETA.git

影响的文件或目录：
当前工作目录下会新增 THETA 文件夹。

可能的资源消耗：
网络下载和少量磁盘空间。

是否会覆盖已有文件：
否；如果目标目录已存在，我会先停下来确认。

请确认是否继续。
```

### 初始任务确认

```text
我需要先确认几个信息：

1. 数据文件路径是什么？
2. 文本列叫什么？例如 text、content、正文、abstract。
3. 数据是中文、英文，还是中英混合？
4. 是否有时间列？例如 year、date、timestamp。
5. 是否有省份、机构、类别、来源等协变量？
6. 你的目标更偏向快速跑通、主题质量最好、可解释性最好、时间演化，还是论文级模型对比？
```

### 资源与成本确认

```text
还需要确认资源边界：

1. 是否可以使用云端 embedding API？
2. 文本数据是否允许发送到外部 API？
3. 是否有本地 GPU？
4. 你希望先跑小规模验证，还是直接跑完整实验？
```

### 缺少数据路径

```text
我还需要数据文件路径才能继续做只读预检。请提供 CSV、TSV、JSONL 或 Excel 文件路径，或者数据目录路径。
```

### 缺少文本列

```text
我可以先读取列名来判断文本列；如果你已经知道，请告诉我文本列名，例如 text、content、正文 或 abstract。
```

### 模式选择

```text
你希望使用哪种 THETA 模式？

1. zero_shot：可以使用云端 embedding，适合快速跑通，但可能产生 API 费用。
2. supervised：需要本地模型，适合有标签或需要 finetune 的流程。
3. unsupervised：需要本地模型，适合本地训练和研究实验。
```

### 输出目标

```text
最终你希望得到什么输出？

1. 只要跑通并看到主题词。
2. 需要模型指标和可视化。
3. 需要多模型对比。
4. 需要论文可用的实验结论和报告。
```

## English Question Templates

### Missing Repository Confirmation

```text
I did not find a usable THETA repository. When the THETA Skill is active, the first step is to prepare the repository before configuring the environment, inspecting data, or generating training commands.

I am preparing to perform the following action:

Action:
Clone the THETA repository locally.

Command:
git clone https://github.com/CodeSoul-co/THETA.git

Affected files or directories:
A new THETA folder will be created under the current working directory.

Potential resource usage:
Network download and a small amount of disk space.

Will this overwrite existing files:
No. If the target directory already exists, I will stop and confirm first.

Please confirm whether to continue.
```

### Initial Task Intake

```text
I need to confirm a few details first:

1. What is the dataset path?
2. What is the text column name? For example: text, content, body, abstract.
3. Is the dataset Chinese, English, or mixed?
4. Does it include a time column? For example: year, date, timestamp.
5. Does it include covariates such as province, organization, category, source, or department?
6. Is your goal a quick run, best topic quality, best interpretability, topic evolution, or paper-grade model comparison?
```

### Resource and Cost Boundaries

```text
I also need to confirm the resource boundaries:

1. Can cloud embedding APIs be used?
2. Is it acceptable for text data to be sent to an external API?
3. Is a local GPU available?
4. Do you want a small smoke test first, or a full experiment directly?
```

### Missing Dataset Path

```text
I need the dataset path before I can continue with read-only preflight. Please provide a CSV, TSV, JSONL, Excel file path, or a dataset directory path.
```

### Missing Text Column

```text
I can inspect the column names first and infer the text column. If you already know it, please provide the text column name, such as text, content, body, or abstract.
```

### Mode Selection

```text
Which THETA mode do you want to use?

1. zero_shot: can use cloud embedding, suitable for a quick run, but may incur API cost.
2. supervised: requires local models, suitable for labeled data or finetune-capable workflows.
3. unsupervised: requires local models, suitable for local training and research experiments.
```

### Output Goal

```text
What final output do you want?

1. Just run successfully and inspect topic words.
2. Include model metrics and visualizations.
3. Include multi-model comparison.
4. Produce paper-ready experimental conclusions and a report.
```
