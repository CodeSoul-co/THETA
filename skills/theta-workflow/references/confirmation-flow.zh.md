# THETA Skill 使用流程：中文确认版

## 第 0 步：识别是否应该使用 THETA Skill

当用户说到下面这些内容时，agent 应判断可以启用 THETA Skill：

```text
THETA
主题建模
topic modeling
LDA / STM / DTM / CTM / BERTopic
政策文本主题分析
社交媒体文本主题分析
主题演化
自动建模
文本聚类 / 主题发现
```

这一阶段只是判断是否启用 skill，不需要用户确认。

## 第 1 步：检查仓库是否存在

如果启动了 THETA Skill，agent 的第一步应该是检查当前工作区或用户提供的位置是否已经有可用的 THETA 仓库。

只读检查不需要确认：

- THETA 仓库是否存在。
- `README.md` 是否存在。
- `scripts/` 目录是否存在。
- `src/models/run_pipeline.py` 是否存在。
- `.env.example` 是否存在。

如果仓库不存在，第一项需要执行的动作是克隆仓库，而不是配置环境、检查数据或生成训练命令。

克隆命令：

```bash
git clone https://github.com/CodeSoul-co/THETA.git
```

这会在本地创建新文件夹，必须确认：

```text
我没有找到本地 THETA 仓库。

我准备执行以下操作：

操作：
克隆 THETA 仓库到本地。

命令：
git clone https://github.com/CodeSoul-co/THETA.git

影响：
会创建一个新的 THETA 文件夹，不会修改已有文件。

请确认是否继续。
```

确认并克隆完成后，才能继续进入环境检查、`.env` 配置、数据检查和模型流程。

## 第 2 步：理解用户任务

先问必要信息，不要一次问太多：

1. 数据文件在哪里？
2. 文本列叫什么？例如 `text`、`content`、`正文`、`abstract`。
3. 数据语言是中文还是英文？
4. 是否有时间列？例如 `year`、`date`、`timestamp`。
5. 是否有协变量？例如 `province`、`policy_type`、`source`、`department`。
6. 用户目标是什么：快速跑通、主题质量最好、可解释性最好、时间演化、论文实验对比。

这一阶段只是提问和理解需求，不需要额外确认。

示例：

```text
我需要先确认几个信息：

1. 你的数据文件路径是什么？
2. 文本列叫什么？
3. 数据是中文还是英文？
4. 是否有时间列？
5. 是否有省份、机构、类别等协变量？
6. 你更想快速跑通，还是做论文级模型对比？
```

## 第 3 步：再次确认仓库结构

进入后续流程前，可以再次只读检查：

- THETA 仓库是否存在。
- `README.md` 是否存在。
- `scripts/` 目录是否存在。
- `src/models/run_pipeline.py` 是否存在。
- `.env.example` 是否存在。

只读检查不需要用户确认。

如果此时仍然没有仓库，需要回到第 1 步，确认后克隆：

```bash
git clone https://github.com/CodeSoul-co/THETA.git
```

这会在本地创建新文件夹，必须确认：

```text
我没有找到本地 THETA 仓库。

我准备执行以下操作：

操作：
克隆 THETA 仓库到本地。

命令：
git clone https://github.com/CodeSoul-co/THETA.git

影响：
会创建一个新的 THETA 文件夹，不会修改已有文件。

请确认是否继续。
```

## 第 4 步：检查环境

可以读取环境情况，不需要确认：

- 当前 Python 版本。
- 是否存在 conda。
- 是否存在 theta 环境。
- 是否存在 requirements 文件。
- 是否存在 `.env`。
- 是否存在模型路径配置。

创建环境或安装依赖必须确认：

```bash
conda create -n theta python=3.10 -y
bash scripts/env_setup.sh
pip install -r requirements.txt
```

确认话术：

```text
THETA 需要 Python 3.10 环境。

我准备执行以下操作：

操作：
创建 conda 环境并安装 THETA 依赖。

命令：
conda create -n theta python=3.10 -y
conda activate theta
bash scripts/env_setup.sh

影响：
会创建新的 conda 环境 theta，并安装依赖包。

请确认是否继续。
```

## 第 5 步：检查 `.env` 配置

读取 `.env.example` 并判断需要哪些配置，不需要确认。

创建或修改 `.env` 必须确认：

```bash
cp .env.example .env
```

或写入：

```text
EMBEDDING_PROVIDER=cloud
OPENAI_API_KEY=...
QWEN_MODEL_0_6B=...
```

agent 不能在回复里直接展示、打印、保存用户 API Key。

确认话术：

```text
THETA 需要配置 `.env` 文件。

我准备执行以下操作：

操作：
根据 `.env.example` 创建 `.env` 配置文件。

命令：
cp .env.example .env

影响：
会新建 `.env` 文件。如果文件已经存在，我不会直接覆盖，会先提醒你。

注意：
如果使用云端 embedding，需要你自己把 API Key 放进 `.env`，我不会在回复中显示或保存你的密钥。

请确认是否继续。
```

## 第 6 步：选择 embedding 模式

### 5.1 云端 embedding

适合普通用户，配置简单，但可能产生 API 费用。

```text
EMBEDDING_PROVIDER=cloud
EMBEDDING_CLOUD_PROVIDER=openai
```

因为可能调用付费 API，必须确认：

```text
你当前选择的是云端 embedding。

这可能会调用外部 API，并根据你的账户设置产生费用。

我准备使用以下配置：

EMBEDDING_PROVIDER=cloud
EMBEDDING_CLOUD_PROVIDER=openai

请确认是否继续。
```

### 5.2 本地 embedding

适合研究用户，但需要本地模型。

```text
QWEN_MODEL_0_6B=./models/qwen3_embedding_0.6B
SBERT_MODEL_PATH=./models/sbert
```

检查路径不需要确认；下载模型必须确认：

```text
当前本地模型路径不存在。

我准备下载或配置本地 embedding 模型。

操作：
下载或设置 Qwen / SBERT 模型路径。

影响：
可能占用较多磁盘空间，并且下载时间较长。

请确认是否继续。
```

### 5.3 高性能本地大模型

4B / 8B embedding 模型需要更多 GPU、显存和存储，必须确认：

```text
你选择的是更大的本地 embedding 模型。

这可能需要较高 GPU 显存和较大磁盘空间。

我建议先使用 0.6B 模型跑通流程，再决定是否切换到 4B 或 8B。

是否仍然继续使用大模型？
```

## 第 7 步：检查数据集

只读检查不需要确认：

- 行数。
- 列名。
- 文本列是否存在。
- 是否有空文本。
- 是否有重复文本。
- 文本平均长度。
- 是否有时间列。
- 是否有协变量列。

示例：

```text
我检查了数据集：

总行数：8420
检测到文本列：content
检测到时间列：year
没有发现严重缺失
重复文本数量：132
平均文本长度：486 字符

这个数据适合做政策文本主题演化分析。
```

## 第 8 步：数据格式转换

THETA 通常期望文本列为 `text`，时间列为 `timestamp`，协变量列使用 `cov_` 前缀。

如果原始数据是：

```text
content
year
province
policy_type
```

可建议转换成：

```text
text
timestamp
cov_province
cov_policy_type
```

涉及写文件就必须确认。默认不要修改原始数据，而是创建新文件，例如 `policy_data.theta_ready.csv`。

确认话术：

```text
你的数据列名和 THETA 期望格式不完全一致。

我建议创建一个转换后的副本：

原始文件：
policy_data.csv

新文件：
policy_data.theta_ready.csv

转换规则：
content → text
year → timestamp
province → cov_province
policy_type → cov_policy_type

原始文件不会被修改。

请确认是否创建转换文件。
```

## 第 9 步：推荐模型

根据数据和目标推荐模型，不需要确认：

- 不知道主题数：推荐 HDP / BERTopic。
- 短文本：推荐 BTM / BERTopic。
- 有时间列：推荐 DTM + THETA。
- 有协变量：推荐 STM + THETA。
- 高质量语义主题：推荐 THETA。
- 论文实验对比：推荐 LDA + CTM + BERTopic + ProdLDA + THETA。

示例：

```text
根据你的数据情况，我建议：

主模型：
THETA

对比模型：
LDA、CTM、BERTopic

如果你重点关注时间演化，可以额外加入 DTM。

理由：
你的数据是政策文本，并且包含 year 时间列；THETA 适合作为高质量语义主题模型，DTM 适合展示主题随时间变化。
```

## 第 10 步：推荐参数

建议参数不需要确认：

```text
我建议第一轮参数如下：

num_topics = 20
vocab_size = 5000
epochs = 100
batch_size = 64
learning_rate = 0.002

如果第一轮主题重叠严重，可以把 K 调小到 15。
如果主题太粗，可以把 K 调大到 30。
```

数据规模参考：

- 小数据集：K = 8 / 12 / 15。
- 中等数据集：K = 15 / 20 / 30。
- 大数据集：K = 30 / 50 / 80。

## 第 11 步：生成运行命令

生成命令给用户看不需要确认，真正执行命令必须确认。

示例命令必须根据当前仓库真实 CLI 调整：

```bash
python src/models/run_pipeline.py \
  --dataset policy_data \
  --models theta,lda,ctm,bertopic \
  --num_topics 20 \
  --language zh \
  --epochs 100 \
  --vocab_size 5000
```

## 第 12 步：执行模型训练

训练可能占用 GPU、耗时、产生 API 费用、写入结果文件或运行多个模型，必须确认。

确认话术：

```text
我准备开始运行模型。

操作：
运行 THETA、LDA、CTM、BERTopic 四个模型。

命令：
python src/models/run_pipeline.py \
  --dataset policy_data \
  --models theta,lda,ctm,bertopic \
  --num_topics 20 \
  --language zh \
  --epochs 100 \
  --vocab_size 5000

输出目录：
result/policy_data/

资源影响：
可能占用本地计算资源。如果启用了 cloud embedding，可能产生 API 费用。

是否会覆盖旧结果：
不会，默认创建新的实验目录。

请确认是否执行。
```

只有用户明确确认后才能继续。

## 第 13 步：读取运行结果

读取已有结果不需要确认：

- `metrics.json`
- `topic_words.json`
- `topic_table.csv`
- `topic_network.png`
- `topic_similarity.png`
- `training_history.json`

解释时覆盖：

- 主题质量怎么样。
- topic coherence 是否高。
- topic diversity 是否高。
- exclusivity 是否高。
- 主题之间是否重叠。
- 是否需要调整 K。
- 是否需要换模型。

示例：

```text
我读取了本轮结果。

整体来看：
THETA 的主题多样性较好，但 Topic 7 和 Topic 12 有明显重叠。
NPMI 中等，说明主题语义连贯性还可以继续提升。
Exclusivity 偏低，说明部分主题边界不够清晰。

建议下一轮尝试：
K 从 20 降到 15；
同时增加自定义停用词表。
```

## 第 14 步：调参建议

建议本身不需要确认：

- 主题太碎：降低 `num_topics`。
- 主题太粗：提高 `num_topics`。
- 主题词重复严重：加强停用词、降低 K、提高 exclusivity。
- coherence 低：调整 `vocab_size`、增加预处理、换 THETA/CTM。
- 训练不稳定：降低 `learning_rate`，增加 patience。

## 第 15 步：执行调参实验

建议不需要确认；真的跑第二轮、第三轮必须确认。

```text
我建议下一轮做两个实验：

实验 A：
K = 15

实验 B：
K = 30

当前实验：
K = 20

目的：
比较不同主题数下的 coherence、diversity 和 exclusivity。

这会额外启动 2 次模型运行。

请确认是否继续。
```

## 第 16 步：生成最终报告

只在对话里总结不需要确认。

写入文件必须确认，例如：

- `report.md`
- `experiment_summary.docx`
- `topic_modeling_result.tex`

确认话术：

```text
我可以把本轮实验结果整理成报告文件：

输出文件：
theta_experiment_report.md

内容包括：
模型配置
数据说明
评估指标
主题解释
调参建议
论文写作口径

请确认是否生成该文件。
```

## 第 17 步：清理文件或删除结果

任何删除行为都必须确认：

- 删除缓存。
- 删除旧结果。
- 删除模型文件。
- 删除中间数据。
- 重置 workspace。

确认话术：

```text
我发现有旧的实验结果目录：

result/policy_data/exp_001
result/policy_data/exp_002

如果清理它们，将无法从当前目录恢复这些结果。

请确认是否删除。
```

## 确认表

| 步骤 | 操作 | 是否需要确认 |
| --- | --- | --- |
| 识别是否启用 skill | 判断任务是否属于 THETA | 不需要 |
| 检查仓库是否存在 | 读取目录结构 | 不需要 |
| 仓库不存在时克隆仓库 | `git clone` | 需要 |
| 理解任务 | 询问数据、语言、目标 | 不需要 |
| 检查仓库 | 读取目录结构 | 不需要 |
| 克隆仓库 | `git clone` | 需要 |
| 检查环境 | 查看 Python / conda / `.env` | 不需要 |
| 创建环境 | `conda create` | 需要 |
| 安装依赖 | `pip install` / `env_setup.sh` | 需要 |
| 读取 `.env.example` | 查看配置模板 | 不需要 |
| 修改 `.env` | 写入配置 | 需要 |
| 检查 API Key 是否存在 | 判断变量是否配置 | 不需要 |
| 使用云 API | 可能产生费用 | 需要 |
| 检查数据 | 读取列名、行数、缺失 | 不需要 |
| 转换数据 | 创建新 CSV / JSONL | 需要 |
| 推荐模型 | LDA / DTM / STM / THETA | 不需要 |
| 推荐参数 | K、epoch、batch size | 不需要 |
| 生成命令 | 给用户看命令 | 不需要 |
| 执行模型 | 运行训练代码 | 需要 |
| 读取结果 | 读 metrics / topics | 不需要 |
| 调参建议 | 分析下一轮怎么改 | 不需要 |
| 执行调参 | 重新训练 | 需要 |
| 对话中总结报告 | 直接回复总结 | 不需要 |
| 写入报告文件 | 生成 md / docx / tex | 需要 |
| 删除结果 | 清理文件夹 | 需要 |
