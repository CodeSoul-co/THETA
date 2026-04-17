# Shell 脚本参考手册

所有脚本均为**非交互式**（纯命令行参数），适用于 DLC/批处理环境，无需标准输入。

## 脚本概览

| 脚本 | 描述 |
|--------|-------------|
| `01_setup.sh` | 安装依赖并从 HuggingFace 下载数据 |
| `02_clean_data.sh` | 清洗原始文本数据（分词、停用词去除、词形还原） |
| `02_generate_embeddings.sh` | 生成通义千问嵌入（03 的子脚本，用于失败恢复） |
| `03_prepare_data.sh` | 一站式数据准备：为所有 12 个模型生成词袋 + 嵌入 |
| `04_train_theta.sh` | 训练 THETA 模型（训练 + 评估 + 可视化一体化） |
| `05_train_baseline.sh` | 训练 11 个基线模型与 THETA 进行比较 |
| `06_visualize.sh` | 为已训练模型生成可视化 |
| `07_evaluate.sh` | 使用 7 个统一指标进行独立评估 |
| `08_compare_models.sh` | 跨模型指标比较表 |
| `09_download_from_hf.sh` | 从 HuggingFace 下载预训练数据 |
| `10_quick_start_english.sh` | 英文数据集快速入门 |
| `11_quick_start_chinese.sh` | 中文数据集快速入门 |
| `12_train_multi_gpu.sh` | 使用 DistributedDataParallel 进行多 GPU 训练 |
| `13_test_agent.sh` | 测试 LLM 智能体连接和功能 |
| `14_start_agent_api.sh` | 启动智能体 API 服务器（FastAPI） |

---

## A) 数据清洗 — `02_clean_data.sh`

逐行文本清洗，用户指定列选择。两种模式：
- **CSV 模式**：用户指定 `--text_column`（需清洗的列）和 `--label_columns`（保留原样的列）
- **目录模式**：将 docx/txt 文件转换为单个清洗后的 CSV 文件

**支持的语言**：`english`、`chinese`、`german`、`spanish`

```bash
# 1. 预览列（CSV 格式推荐的第一步）
bash scripts/02_clean_data.sh \
    --input data/FCPB/complaints_text_only.csv --preview

# 2. 仅清洗文本列
bash scripts/02_clean_data.sh \
    --input data/FCPB/complaints_text_only.csv \
    --language english \
    --text_column 'Consumer complaint narrative'

# 3. 清洗文本 + 保留标签列
bash scripts/02_clean_data.sh \
    --input data/hatespeech/hatespeech_text_only.csv \
    --language english \
    --text_column cleaned_content --label_columns Label

# 4. 保留所有列，仅清洗文本列
bash scripts/02_clean_data.sh \
    --input raw.csv --language english \
    --text_column text --keep_all

# 5. 目录模式（docx/txt → CSV）
bash scripts/02_clean_data.sh \
    --input data/edu_data/ --language chinese
```

| 参数 | 必需 | 描述 | 默认值 |
|-----------|----------|-------------|---------|
| `--input` | 是 | 输入的 CSV 文件或目录（docx/txt） | - |
| `--language` | 是（预览模式不适用） | 数据语言：english, chinese, german, spanish | - |
| `--text_column` | 是（CSV 模式） | 需要清洗的文本列名 | - |
| `--label_columns` | | 需要原样保留的标签/元数据列，逗号分隔 | - |
| `--keep_all` | | 保留所有原始列（仅清洗文本列） | false |
| `--preview` | | 显示 CSV 列和示例行后退出 | false |
| `--output` | | 输出 CSV 路径 | 自动生成 |
| `--min_words` | | 清洗后每个文档的最小词数 | 3 |

**输出**：`data/{数据集}/{数据集}_cleaned.csv`

---

## B) 数据准备 — `03_prepare_data.sh`

为所有 12 个模型提供一站式数据准备。生成词袋矩阵和特定模型的嵌入。

**各模型的数据需求**：

| 模型 | 类型 | 所需数据 |
|-------|------|-------------|
| lda, hdp, btm | 传统模型 | 仅词袋 |
| stm | 传统模型 | 词袋 + 协变量（文档元数据） |
| nvdm, gsm, prodlda | 神经模型 | 仅词袋 |
| etm | 神经模型 | 词袋 + Word2Vec |
| ctm | 神经模型 | 词袋 + SBERT |
| dtm | 神经模型 | 词袋 + SBERT + 时间切片 |
| bertopic | 神经模型 | SBERT + 原始文本 |
| theta | THETA 模型 | 词袋 + 通义千问嵌入 |

> **注意**：模型 1-7（仅需词袋）共享相同的数据实验。准备一次，训练所有。

```bash
# ---- 基线模型 ----

# 仅需词袋的模型（lda, hdp, btm, nvdm, gsm, prodlda 共享此数据）
bash scripts/03_prepare_data.sh \
    --dataset edu_data --model lda --vocab_size 3500 --language chinese

# CTM（词袋 + SBERT 嵌入）
bash scripts/03_prepare_data.sh \
    --dataset edu_data --model ctm --vocab_size 3500 --language chinese

# ETM（词袋 + Word2Vec 嵌入）
bash scripts/03_prepare_data.sh \
    --dataset edu_data --model etm --vocab_size 3500 --language chinese

# DTM（词袋 + SBERT + 时间切片，需要时间列）
bash scripts/03_prepare_data.sh \
    --dataset edu_data --model dtm --vocab_size 3500 --language chinese --time_column year

# BERTopic（SBERT + 原始文本）
bash scripts/03_prepare_data.sh \
    --dataset edu_data --model bertopic --vocab_size 3500 --language chinese

# ---- THETA 模型 ----

# 零样本（最快，无需训练）
bash scripts/03_prepare_data.sh \
    --dataset edu_data --model theta --model_size 0.6B --mode zero_shot \
    --vocab_size 3500 --language chinese

# 无监督（LoRA 微调通义千问嵌入）
bash scripts/03_prepare_data.sh \
    --dataset edu_data --model theta --model_size 0.6B --mode unsupervised \
    --vocab_size 3500 --language chinese

# 有监督（需要标签列）
bash scripts/03_prepare_data.sh \
    --dataset edu_data --model theta --model_size 0.6B --mode supervised \
    --vocab_size 3500 --language chinese

# ---- 高级选项 ----

# 仅生成词袋（跳过嵌入生成）
bash scripts/03_prepare_data.sh --dataset mydata --model theta --bow-only --vocab_size 5000

# 检查数据文件是否已存在
bash scripts/03_prepare_data.sh --dataset mydata --model theta --check-only

# 自定义词汇表大小
bash scripts/03_prepare_data.sh --dataset mydata \
    --model theta --model_size 0.6B --mode zero_shot \
    --vocab_size 10000 --batch_size 64 --gpu 0
```

| 参数 | 必需 | 描述 | 默认值 |
|-----------|----------|-------------|---------|
| `--dataset` | 是 | 数据集名称 | - |
| `--model` | 是 | 目标模型：lda, hdp, stm（需要协变量）, btm, nvdm, gsm, prodlda, ctm, etm, dtm, bertopic, theta | - |
| `--model_size` | | 通义千问模型规模（仅 theta）：0.6B, 4B, 8B | 0.6B |
| `--mode` | | 嵌入模式（仅 theta）：zero_shot, unsupervised, supervised | zero_shot |
| `--vocab_size` | | 词汇表大小 | 5000 |
| `--batch_size` | | 嵌入生成批大小 | 32 |
| `--gpu` | | GPU 设备 ID | 0 |
| `--language` | | 数据语言：english, chinese（控制分词） | english |
| `--bow-only` | | 仅生成词袋，跳过嵌入 | false |
| `--check-only` | | 仅检查文件是否存在 | false |
| `--time_column` | | 时间列名（仅 DTM） | year |
| `--label_column` | | 标签列（仅 theta 有监督模式） | - |
| `--emb_epochs` | | 嵌入微调轮数（仅 theta） | 10 |
| `--emb_batch_size` | | 嵌入微调批大小（仅 theta） | 8 |
| `--exp_name` | | 实验名称标签 | 自动生成 |

**嵌入恢复** — 如果嵌入生成失败（如内存不足），可仅重新运行嵌入步骤：

```bash
bash scripts/02_generate_embeddings.sh \
    --dataset edu_data --mode zero_shot --model_size 0.6B \
    --batch_size 4 --exp_dir result/0.6B/edu_data/data/exp_xxx
```

---

## C) THETA 模型训练 — `04_train_theta.sh`

训练 THETA 模型，集成了训练 + 评估 + 可视化。

```bash
# ---- 基本用法 ----

# 零样本模式（最简单的命令）
bash scripts/04_train_theta.sh \
    --dataset edu_data --model_size 0.6B --mode zero_shot --num_topics 20

# 无监督模式
bash scripts/04_train_theta.sh \
    --dataset edu_data --model_size 0.6B --mode unsupervised --num_topics 20

# 有监督模式（需要标签列）
bash scripts/04_train_theta.sh \
    --dataset hatespeech --model_size 0.6B --mode supervised --num_topics 20

# 使用更大模型以获得更好质量
bash scripts/04_train_theta.sh \
    --dataset hatespeech --model_size 4B --mode zero_shot --num_topics 20

# ---- 完整参数 ----

bash scripts/04_train_theta.sh \
    --dataset edu_data --model_size 0.6B --mode zero_shot \
    --num_topics 20 --epochs 100 --batch_size 64 \
    --hidden_dim 512 --learning_rate 0.002 \
    --kl_start 0.0 --kl_end 1.0 --kl_warmup 50 \
    --patience 10 --gpu 0 --language zh

# 自定义 KL 退火
bash scripts/04_train_theta.sh \
    --dataset hatespeech --model_size 0.6B --mode zero_shot \
    --num_topics 20 --epochs 200 \
    --kl_start 0.1 --kl_end 0.8 --kl_warmup 40

# ---- 指定数据实验 ----

bash scripts/04_train_theta.sh \
    --dataset edu_data --model_size 0.6B --mode zero_shot \
    --data_exp exp_20260208_151906_vocab3500_theta_0.6B_zero_shot \
    --num_topics 20 --epochs 50 --language zh

# ---- 跳过选项 ----

# 跳过可视化（仅训练 + 评估，更快）
bash scripts/04_train_theta.sh \
    --dataset edu_data --model_size 0.6B --mode zero_shot \
    --num_topics 20 --skip-viz

# 跳过训练（评估 + 可视化现有模型）
bash scripts/04_train_theta.sh \
    --dataset edu_data --model_size 0.6B --mode zero_shot \
    --skip-train --language zh
```

| 参数 | 必需 | 描述 | 默认值 |
|-----------|----------|-------------|---------|
| `--dataset` | 是 | 数据集名称 | - |
| `--model_size` | | 通义千问模型规模：0.6B, 4B, 8B | 0.6B |
| `--mode` | | 嵌入模式：zero_shot, unsupervised, supervised | zero_shot |
| `--num_topics` | | 主题数量 K | 20 |
| `--epochs` | | 训练轮数 | 100 |
| `--batch_size` | | 训练批大小 | 64 |
| `--hidden_dim` | | 编码器隐藏层维度 | 512 |
| `--learning_rate` | | 学习率 | 0.002 |
| `--kl_start` | | KL 退火起始权重 | 0.0 |
| `--kl_end` | | KL 退火结束权重 | 1.0 |
| `--kl_warmup` | | KL 退火轮数 | 50 |
| `--patience` | | 早停耐心值 | 10 |
| `--gpu` | | GPU 设备 ID | 0 |
| `--language` | | 可视化语言：en, zh | en |
| `--skip-train` | | 跳过训练，仅评估 | false |
| `--skip-viz` | | 跳过可视化 | false |
| `--data_exp` | | 数据实验 ID | 自动最新 |
| `--exp_name` | | 实验名称标签 | 自动生成 |

---

## D) 基线模型训练 — `05_train_baseline.sh`

训练 11 个基线主题模型，与 THETA 进行比较。

### 支持的模型

| 模型 | 类型 | 描述 | 特定模型参数 |
|-------|------|-------------|---------------------------|
| **lda** | 传统模型 | 潜在狄利克雷分配 | `--max_iter` |
| **hdp** | 传统模型 | 层次狄利克雷过程（自动确定主题数） | `--max_topics`, `--alpha` |
| **stm** | 传统模型 | 结构主题模型（**需要协变量**） | `--max_iter` |
| **btm** | 传统模型 | 双词主题模型（最适合短文本） | `--n_iter`, `--alpha`, `--beta` |
| **nvdm** | 神经模型 | 神经变分文档模型 | `--epochs`, `--dropout` |
| **gsm** | 神经模型 | 高斯 Softmax 模型 | `--epochs`, `--dropout` |
| **prodlda** | 神经模型 | 专家乘积 LDA | `--epochs`, `--dropout` |
| **ctm** | 神经模型 | 上下文主题模型（需要 SBERT） | `--epochs`, `--inference_type` |
| **etm** | 神经模型 | 嵌入主题模型（需要 Word2Vec） | `--epochs` |
| **dtm** | 神经模型 | 动态主题模型（需要时间戳） | `--epochs` |
| **bertopic** | 神经模型 | 基于 BERT 的主题模型（自动确定主题数） | - |

### 各模型的完整示例

```bash
# ============================================================
# 1. LDA — 潜在狄利克雷分配
# ============================================================

bash scripts/05_train_baseline.sh \
    --dataset edu_data --models lda --num_topics 20

bash scripts/05_train_baseline.sh \
    --dataset edu_data --models lda \
    --num_topics 20 --max_iter 200 \
    --gpu 0 --language zh --with-viz \
    --data_exp exp_20260208_153424_vocab3500_lda \
    --exp_name lda_full

# ============================================================
# 2. HDP — 层次狄利克雷过程
#    注意：自动确定主题数，--num_topics 被忽略
# ============================================================

bash scripts/05_train_baseline.sh \
    --dataset edu_data --models hdp

bash scripts/05_train_baseline.sh \
    --dataset edu_data --models hdp \
    --max_topics 150 --alpha 1.0 \
    --gpu 0 --language zh --with-viz \
    --data_exp exp_20260208_153424_vocab3500_lda \
    --exp_name hdp_full

# ============================================================
# 3. STM — 结构主题模型
#    需要协变量 — 如果数据集没有元数据则自动跳过
# ============================================================
#
# 使用 STM 的方法：
#   1. 确保清洗后的 CSV 有元数据列（如 year, source, category）
#   2. 在 ETM/config.py → DATASET_CONFIGS 中注册协变量
#   3. 准备数据（与其他仅词袋模型相同）
#   4. 训练 STM

bash scripts/05_train_baseline.sh \
    --dataset my_dataset_with_covariates --models stm --num_topics 20

bash scripts/05_train_baseline.sh \
    --dataset my_dataset_with_covariates --models stm \
    --num_topics 20 --max_iter 200 \
    --gpu 0 --language en --with-viz \
    --data_exp exp_20260208_153424_vocab3500_lda \
    --exp_name stm_full

# ============================================================
# 4. BTM — 双词主题模型（最适合短文本）
# ============================================================

bash scripts/05_train_baseline.sh \
    --dataset edu_data --models btm --num_topics 20

bash scripts/05_train_baseline.sh \
    --dataset edu_data --models btm \
    --num_topics 20 --n_iter 100 --alpha 1.0 --beta 0.01 \
    --gpu 0 --language zh --with-viz \
    --data_exp exp_20260208_153424_vocab3500_lda \
    --exp_name btm_full

# ============================================================
# 5. NVDM / 6. GSM / 7. ProdLDA — 仅词袋神经模型
# ============================================================

bash scripts/05_train_baseline.sh \
    --dataset edu_data --models nvdm --num_topics 20

bash scripts/05_train_baseline.sh \
    --dataset edu_data --models nvdm \
    --num_topics 20 --epochs 200 --batch_size 128 \
    --hidden_dim 512 --learning_rate 0.002 --dropout 0.2 \
    --gpu 0 --language zh --with-viz \
    --data_exp exp_20260208_153424_vocab3500_lda \
    --exp_name nvdm_full

# （将 nvdm 替换为 gsm 或 prodlda 可训练对应模型）

# ============================================================
# 8. CTM — 上下文主题模型
#    需要 SBERT 数据实验（使用 --model ctm 准备）
# ============================================================

bash scripts/05_train_baseline.sh \
    --dataset edu_data --models ctm --num_topics 20

bash scripts/05_train_baseline.sh \
    --dataset edu_data --models ctm \
    --num_topics 20 --epochs 100 --inference_type zeroshot \
    --batch_size 64 --hidden_dim 512 --learning_rate 0.002 \
    --gpu 0 --language zh --with-viz \
    --data_exp exp_20260208_154645_vocab3500_ctm \
    --exp_name ctm_zeroshot

# 组合推理（同时使用词袋和 SBERT）
bash scripts/05_train_baseline.sh \
    --dataset edu_data --models ctm \
    --num_topics 20 --epochs 100 --inference_type combined \
    --gpu 0 --language zh --with-viz

# ============================================================
# 9. ETM — 嵌入主题模型（词袋 + Word2Vec）
# ============================================================

bash scripts/05_train_baseline.sh \
    --dataset edu_data --models etm --num_topics 20

bash scripts/05_train_baseline.sh \
    --dataset edu_data --models etm \
    --num_topics 20 --epochs 200 --batch_size 64 \
    --hidden_dim 512 --learning_rate 0.002 \
    --gpu 0 --language zh --with-viz \
    --data_exp exp_20260208_153424_vocab3500_lda \
    --exp_name etm_full

# ============================================================
# 10. DTM — 动态主题模型（需要时间戳）
# ============================================================

bash scripts/05_train_baseline.sh \
    --dataset edu_data --models dtm --num_topics 20

bash scripts/05_train_baseline.sh \
    --dataset edu_data --models dtm \
    --num_topics 20 --epochs 200 --batch_size 64 \
    --hidden_dim 512 --learning_rate 0.002 \
    --gpu 0 --language zh --with-viz \
    --data_exp exp_20260208_171413_vocab3500_dtm \
    --exp_name dtm_full

# ============================================================
# 11. BERTopic — 自动确定主题数
# ============================================================

bash scripts/05_train_baseline.sh \
    --dataset edu_data --models bertopic

bash scripts/05_train_baseline.sh \
    --dataset edu_data --models bertopic \
    --gpu 0 --language zh --with-viz \
    --data_exp exp_20260208_154645_vocab3500_ctm \
    --exp_name bertopic_full

# ============================================================
# 批量训练（同时训练多个模型）
# ============================================================

# 训练所有仅词袋模型（共享同一个数据实验）
bash scripts/05_train_baseline.sh \
    --dataset edu_data \
    --models lda,hdp,btm,nvdm,gsm,prodlda \
    --num_topics 20 --epochs 100 \
    --data_exp exp_20260208_153424_vocab3500_lda

# 训练 CTM + BERTopic（共享 SBERT 数据实验）
bash scripts/05_train_baseline.sh \
    --dataset edu_data --models ctm,bertopic \
    --num_topics 20 --epochs 100 \
    --data_exp exp_20260208_154645_vocab3500_ctm

# 跳过训练，仅评估和可视化现有模型
bash scripts/05_train_baseline.sh \
    --dataset edu_data --models lda --num_topics 20 --skip-train

# 启用可视化（默认禁用）
bash scripts/05_train_baseline.sh \
    --dataset edu_data --models lda --num_topics 20 \
    --with-viz --language zh
```

> **重要提示**：
> - BTM 使用吉布斯采样，在长文档上非常慢（每个文档最多采样 50 个词）。最适合短文本。
> - HDP 和 BERTopic 自动确定主题数；对这些模型，`--num_topics` 被忽略。
> - STM 需要文档级别的协变量。如果您的数据在 `DATASET_CONFIGS` 中没有 `covariate_columns`，STM 将自动跳过。
> - DTM 需要包含 `time_slices.json` 的数据实验（使用 `--model dtm` 准备）。
> - CTM 和 BERTopic 需要包含 SBERT 嵌入的数据实验。

### 参数参考

**通用参数**：

| 参数 | 必需 | 描述 | 默认值 |
|-----------|----------|-------------|---------|
| `--dataset` | 是 | 数据集名称 | - |
| `--models` | 是 | 模型列表（逗号分隔） | - |
| `--num_topics` | | 主题数量（hdp/bertopic 忽略此参数） | 20 |
| `--vocab_size` | | 词汇表大小 | 5000 |
| `--epochs` | | 训练轮数（神经模型） | 100 |
| `--batch_size` | | 批大小 | 64 |
| `--hidden_dim` | | 隐藏层维度 | 512 |
| `--learning_rate` | | 学习率 | 0.002 |
| `--gpu` | | GPU 设备 ID | 0 |
| `--language` | | 可视化语言：en, zh | en |
| `--skip-train` | | 跳过训练 | false |
| `--with-viz` | | 启用可视化 | false |
| `--data_exp` | | 数据实验 ID | 自动最新 |
| `--exp_name` | | 实验名称标签 | 自动生成 |

**特定模型参数**：

| 参数 | 适用模型 | 描述 | 默认值 |
|-----------|-------------------|-------------|---------|
| `--max_iter` | lda, stm | 最大迭代次数（EM 算法） | 100 |
| `--max_topics` | hdp | 最大主题数 | 150 |
| `--n_iter` | btm | 吉布斯采样迭代次数 | 100 |
| `--alpha` | hdp, btm | Alpha 先验 | 1.0 |
| `--beta` | btm | Beta 先验 | 0.01 |
| `--inference_type` | ctm | 推理类型：zeroshot, combined | zeroshot |
| `--dropout` | 神经模型（nvdm, gsm, prodlda, ctm, etm, dtm） | Dropout 率 | 0.2 |

---

## E) 可视化 — `06_visualize.sh`

为已训练模型生成可视化，无需重新训练。

```bash
# THETA 模型可视化
bash scripts/06_visualize.sh \
    --dataset edu_data --model_size 0.6B --mode zero_shot --language zh

# 英文图表 + 高 DPI（用于论文）
bash scripts/06_visualize.sh \
    --dataset edu_data --model_size 0.6B --mode zero_shot --language en --dpi 600

# 基线模型可视化
bash scripts/06_visualize.sh \
    --baseline --dataset edu_data --model lda --num_topics 20 --language zh

# HDP（自动主题数，使用训练得到的实际 K 值）
bash scripts/06_visualize.sh \
    --baseline --dataset edu_data --model hdp --num_topics 150 --language zh

# DTM（包含主题演化图表）
bash scripts/06_visualize.sh \
    --baseline --dataset edu_data --model dtm --num_topics 20 --language zh

# 明确指定模型实验
bash scripts/06_visualize.sh \
    --baseline --dataset edu_data --model ctm --model_exp exp_20260208_xxx --language zh
```

| 参数 | 描述 | 默认值 |
|-----------|-------------|---------|
| `--dataset` | 数据集名称（必需） | — |
| `--baseline` | 基线模型模式 | false |
| `--model` | 基线模型名称 | — |
| `--model_exp` | 模型实验 ID（如未指定，自动选择最新） | 自动最新 |
| `--model_size` | THETA 模型规模 | 0.6B |
| `--mode` | THETA 模式 | zero_shot |
| `--language` | 可视化语言：en, zh | en |
| `--dpi` | 图像 DPI | 300 |

**生成的图表**（20+ 种类型）：

| 图表 | 描述 | 文件名 |
|-------|-------------|----------|
| 主题表 | 每个主题的前几个词 | topic_table.png |
| 主题网络 | 主题间相似性网络 | topic_network.png |
| 文档聚类 | UMAP 文档分布 | doc_topic_umap.png |
| 聚类热图 | 主题-文档热图 | cluster_heatmap.png |
| 主题比例 | 每个主题的文档比例 | topic_proportion.png |
| 训练损失 | 损失曲线 | training_loss.png |
| 评估指标 | 7 指标雷达图 | metrics.png |
| 主题连贯性 | 每个主题的 NPMI | topic_coherence.png |
| 主题专有性 | 每个主题的专有性 | topic_exclusivity.png |
| 词云 | 所有主题的词云 | topic_wordclouds.png |
| 主题相似性 | 主题间余弦相似度 | topic_similarity.png |
| pyLDAvis | 交互式主题探索器 | pyldavis_interactive.html |
| 每个主题的词 | 每个主题的词权重 | topics/topic_N/word_importance.png |

---

## F) 评估 — `07_evaluate.sh`

使用 7 个统一指标进行独立评估。

```bash
# 评估基线模型
bash scripts/07_evaluate.sh --dataset edu_data --model lda --num_topics 20
bash scripts/07_evaluate.sh --dataset edu_data --model hdp --num_topics 150
bash scripts/07_evaluate.sh --dataset edu_data --model ctm --num_topics 20
bash scripts/07_evaluate.sh --dataset edu_data --model etm --num_topics 20
bash scripts/07_evaluate.sh --dataset edu_data --model dtm --num_topics 20
bash scripts/07_evaluate.sh --dataset edu_data --model bertopic --num_topics 20

# 自定义词汇表大小
bash scripts/07_evaluate.sh --dataset edu_data --model lda --num_topics 20 --vocab_size 3500

# 评估 THETA 模型
bash scripts/07_evaluate.sh --dataset edu_data --model theta --model_size 0.6B --mode zero_shot
bash scripts/07_evaluate.sh --dataset edu_data --model theta --model_size 0.6B --mode unsupervised
bash scripts/07_evaluate.sh --dataset edu_data --model theta --model_size 4B --mode supervised
```

| 参数 | 描述 | 默认值 |
|-----------|-------------|---------|
| `--dataset` | 数据集名称（必需） | — |
| `--model` | 模型名称（必需）：lda, hdp, stm, btm, nvdm, gsm, prodlda, ctm, etm, dtm, bertopic, theta | — |
| `--num_topics` | 主题数量 | 20 |
| `--vocab_size` | 词汇表大小 | 5000 |
| `--model_size` | THETA 模型规模：0.6B, 4B, 8B | 0.6B |
| `--mode` | THETA 模式：zero_shot, unsupervised, supervised | zero_shot |

**评估指标（7 个指标）**：

| 指标 | 全称 | 方向 | 描述 |
|--------|-----------|-----------|-------------|
| **TD** | 主题多样性 | ↑ 越高越好 | 各主题间独特词的比例 |
| **iRBO** | 逆排名偏重重叠 | ↑ 越高越好 | 基于排名的主题多样性 |
| **NPMI** | 标准化点互信息 | ↑ 越高越好 | 标准化的点互信息连贯性 |
| **C_V** | C_V 连贯性 | ↑ 越高越好 | 基于滑动窗口的连贯性 |
| **UMass** | UMass 连贯性 | → 越接近 0 越好 | 基于文档共现的连贯性 |
| **Exclusivity** | 主题专有性 | ↑ 越高越好 | 词对其主题的专有程度 |
| **PPL** | 困惑度 | ↓ 越低越好 | 模型拟合度（越低泛化越好） |

---

## G) 模型比较 — `08_compare_models.sh`

跨模型指标比较表。

```bash
# 比较所有基线模型
bash scripts/08_compare_models.sh \
    --dataset edu_data \
    --models lda,hdp,btm,nvdm,gsm,prodlda,ctm,etm,dtm,bertopic \
    --num_topics 20

# 仅比较传统模型
bash scripts/08_compare_models.sh \
    --dataset edu_data --models lda,hdp,btm --num_topics 20

# 仅比较神经模型
bash scripts/08_compare_models.sh \
    --dataset edu_data --models nvdm,gsm,prodlda,ctm,etm,dtm --num_topics 20

# 导出到 CSV
bash scripts/08_compare_models.sh \
    --dataset edu_data --models lda,hdp,nvdm,gsm,prodlda,ctm,etm,dtm \
    --num_topics 20 --output comparison.csv
```

**示例输出**：
```
================================================================================
模型比较：edu_data (K=20)
================================================================================

模型              TD     iRBO     NPMI      C_V    UMass  Exclusivity        PPL
--------------------------------------------------------------------------------
lda            0.8500   0.7200   0.0512   0.4231  -2.1234       0.6543     123.45
prodlda        0.9200   0.8100   0.0634   0.4567  -1.8765       0.7234      98.76
ctm            0.8800   0.7800   0.0589   0.4412  -1.9876       0.6987     105.32
--------------------------------------------------------------------------------

最佳模型：
  - 最佳 TD（主题多样性）：prodlda (0.9200)
  - 最佳 NPMI（连贯性）：  prodlda (0.0634)
  - 最佳 PPL（困惑度）：   prodlda (98.76)
```

| 参数 | 描述 | 默认值 |
|-----------|-------------|---------|
| `--dataset` | 数据集名称（必需） | — |
| `--models` | 逗号分隔的模型列表（必需） | — |
| `--num_topics` | 主题数量 | 20 |
| `--output` | 输出 CSV 文件路径 | 仅终端显示 |

---

## H) 多 GPU 训练 — `12_train_multi_gpu.sh`

THETA 支持使用 PyTorch DistributedDataParallel（DDP）进行多 GPU 训练。

```bash
# 使用 2 个 GPU 训练
bash scripts/12_train_multi_gpu.sh --dataset hatespeech --num_gpus 2 --num_topics 20

# 完整参数
bash scripts/12_train_multi_gpu.sh --dataset hatespeech \
    --num_gpus 4 --model_size 0.6B --mode zero_shot \
    --num_topics 25 --epochs 150 --batch_size 64 \
    --hidden_dim 768 --learning_rate 0.001

# 自定义主端口（用于多个并发任务）
bash scripts/12_train_multi_gpu.sh --dataset socialTwitter \
    --num_gpus 2 --master_port 29501

# 或直接使用 torchrun
torchrun --nproc_per_node=2 --master_port=29500 \
    ETM/main.py train \
    --dataset hatespeech --mode zero_shot --num_topics 20 --epochs 100
```

---

## I) 智能体 API — `14_start_agent_api.sh`

启动 AI 智能体 API 服务器，用于交互式分析和问答。

```bash
# 启动智能体 API（默认端口 8000）
bash scripts/14_start_agent_api.sh --port 8000

# 测试智能体连接
bash scripts/13_test_agent.sh
```

API 端点：`POST /chat`、`POST /api/chat/v2`、`POST /api/interpret/metrics`、`POST /api/interpret/topics`、`POST /api/vision/analyze`。完整文档请见 `agent/docs/API_REFERENCE.md`。

---

## J) 批量处理示例

```bash
# 在多个数据集上训练 THETA
for dataset in hatespeech mental_health socialTwitter; do
    bash scripts/04_train_theta.sh --dataset $dataset \
        --model_size 0.6B --mode zero_shot --num_topics 20
done

# 比较不同主题数量
for k in 10 15 20 25 30; do
    bash scripts/04_train_theta.sh --dataset hatespeech \
        --model_size 0.6B --mode zero_shot --num_topics $k
done

# 为所有训练过的基线模型生成可视化
for model in lda etm ctm prodlda; do
    bash scripts/06_visualize.sh --baseline --dataset hatespeech \
        --model $model --num_topics 20 --language en
done
```

---

## K) 端到端示例：edu_data

以下展示了使用 `edu_data`（823 份中文教育政策文档）从数据清洗到模型比较的完整流程。

```bash
# 1. 设置环境
bash scripts/01_setup.sh

# 2. 数据清洗
bash scripts/02_clean_data.sh --input ./data/edu_data/ --language chinese

# 3. 数据准备 — 基线模型
bash scripts/03_prepare_data.sh \
    --dataset edu_data --model lda --vocab_size 3500 --language chinese
bash scripts/03_prepare_data.sh \
    --dataset edu_data --model ctm --vocab_size 3500 --language chinese
bash scripts/03_prepare_data.sh \
    --dataset edu_data --model dtm --vocab_size 3500 --language chinese --time_column year

# 4. 数据准备 — THETA 模型
bash scripts/03_prepare_data.sh \
    --dataset edu_data --model theta --model_size 0.6B --mode zero_shot \
    --vocab_size 3500 --language chinese

# 5. 训练基线模型
bash scripts/05_train_baseline.sh \
    --dataset edu_data --models lda,hdp,btm,nvdm,gsm,prodlda \
    --num_topics 20 --epochs 100

# 6. 训练 THETA 模型
bash scripts/04_train_theta.sh \
    --dataset edu_data --model_size 0.6B --mode zero_shot \
    --num_topics 20 --epochs 100 --language zh

# 7. 模型比较
bash scripts/08_compare_models.sh \
    --dataset edu_data \
    --models lda,hdp,btm,nvdm,gsm,prodlda,ctm,etm \
    --num_topics 20
```
