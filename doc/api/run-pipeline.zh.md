# run_pipeline.py

**[English](run-pipeline.md)** | **[中文](run-pipeline.zh.md)**

---

统一的训练、评估和可视化流程。

---

## 基本用法

```bash
python run_pipeline.py --dataset 数据集名称 --models 模型列表 [选项]
```

---

## 必需参数

| 参数 | 类型 | 描述 |
|-----------|------|-------------|
| `--dataset` | 字符串 | 数据集名称 |
| `--models` | 字符串 | 逗号分隔的模型列表：`theta,lda,hdp,stm,btm,etm,ctm,dtm,nvdm,gsm,prodlda,bertopic` |

---

## 通用参数

所有或大多数模型共享的参数。标记 `*` 的参数仅适用于神经网络模型。

| 参数              | 类型  | 默认值 | 范围       | 描述                                                    |
| ----------------- | ----- | ------ | ---------- | ------------------------------------------------------- |
| `--num_topics`    | int   | 20     | 5–100      | 主题数 K（HDP 为上限；BERTopic 可选）                   |
| `--vocab_size`    | int   | 5000   | 1000–20000 | 词表大小                                                |
| `--epochs` *      | int   | 100    | 10–500     | 训练轮数                                                |
| `--batch_size` *  | int   | 64     | 8–512      | 批大小                                                  |
| `--learning_rate` * | float | 0.002  | 1e-5–0.1   | 学习率                                                  |
| `--dropout` *     | float | 0.2    | 0–0.9      | 编码器 Dropout 率                                       |
| `--hidden_dim` *  | int   | 512    | 128–2048   | 每层隐藏单元数（NVDM/GSM/ProdLDA 默认 256）             |
| `--num_layers` *  | int   | 2      | 1–5        | 编码器隐藏层数                                          |
| `--patience` *    | int   | 10     | 1–50       | 早停耐心轮数                                            |

---

## 各模型额外参数

### THETA

除通用参数外的额外参数：

| 参数           | 类型  | 默认值      | 范围                                      | 描述           |
| -------------- | ----- | ----------- | ----------------------------------------- | -------------- |
| `--model_size` | str   | `0.6B`      | `0.6B` / `4B` / `8B`                      | Qwen 模型规格  |
| `--mode`       | str   | `zero_shot` | `zero_shot` / `supervised` / `unsupervised` | 嵌入模式       |
| `--kl_start`   | float | 0.0         | 0–1                                       | KL 退火起始权重 |
| `--kl_end`     | float | 1.0         | 0–1                                       | KL 退火终止权重 |
| `--kl_warmup`  | int   | 50          | 0–epochs                                  | KL 预热轮数    |
| `--language`   | str   | `zh`        | `en` / `zh`                               | 可视化语言     |

### LDA

| 参数         | 类型  | 默认值   | 范围   | 描述                  |
| ------------ | ----- | -------- | ------ | --------------------- |
| `--max_iter` | int   | 100      | 10–500 | 最大 EM 迭代次数      |
| `--alpha`    | float | 1/K（自动） | >0     | 文档-主题狄利克雷先验 |

### HDP

| 参数           | 类型  | 默认值 | 范围   | 描述                           |
| -------------- | ----- | ------ | ------ | ------------------------------ |
| `--max_topics` | int   | 150    | 50–300 | 主题数上限（替代 `--num_topics`） |
| `--alpha`      | float | 1.0    | >0     | 文档级集中参数                 |

### STM

| 参数         | 类型 | 默认值 | 范围   | 描述             |
| ------------ | ---- | ------ | ------ | ---------------- |
| `--max_iter` | int  | 100    | 10–500 | 最大 EM 迭代次数 |

### BTM

| 参数       | 类型  | 默认值 | 范围   | 描述                           |
| ---------- | ----- | ------ | ------ | ------------------------------ |
| `--n_iter` | int   | 100    | 10–500 | Gibbs 采样迭代次数（替代 `--epochs`） |
| `--alpha`  | float | 1.0    | >0     | 主题分布狄利克雷先验           |
| `--beta`   | float | 0.01   | >0     | 词分布狄利克雷先验             |

### ETM

| 参数              | 类型 | 默认值 | 范围    | 描述                   |
| ----------------- | ---- | ------ | ------- | ---------------------- |
| `--embedding_dim` | int  | 300    | 50–1024 | 词嵌入维度（Word2Vec） |

### CTM

| 参数               | 类型 | 默认值     | 范围                   | 描述                                            |
| ------------------ | ---- | ---------- | ---------------------- | ----------------------------------------------- |
| `--inference_type` | str  | `zeroshot` | `zeroshot` / `combined` | 推理模式：仅 SBERT 或 SBERT + BOW               |
| `--hidden_dim`     | int  | 100        | 32–1024                | 覆盖通用默认值（512 → 100）                     |

### DTM

| 参数              | 类型 | 默认值 | 范围    | 描述           |
| ----------------- | ---- | ------ | ------- | -------------- |
| `--embedding_dim` | int  | 300    | 50–1024 | 词嵌入维度     |

> **注意**：DTM 不使用 `--num_layers`、`--dropout` 或 `--patience`。  
> **数据要求**：DTM 需要数据包含 `timestamp` 列，训练前需运行 `python prepare_data.py --dataset your_data --model dtm`

### NVDM / GSM / ProdLDA

无额外参数 — 所有设置由通用参数覆盖。  
> **注意**：这些模型的 `--hidden_dim` 默认为 256。

### BERTopic

| 参数                 | 类型 | 默认值 | 范围    | 描述                                      |
| -------------------- | ---- | ------ | ------- | ----------------------------------------- |
| `--min_cluster_size` | int  | 10     | 2–100   | HDBSCAN 最小簇大小，控制主题粒度          |
| `--min_samples`      | int  | None   | 1–100   | HDBSCAN min_samples（默认同 min_cluster_size） |
| `--top_n_words`      | int  | 10     | 1–30    | 每个主题展示的词数                        |
| `--n_neighbors`      | int  | 15     | 2–100   | UMAP 近邻数                               |
| `--n_components`     | int  | 5      | 2–50    | UMAP 降维后的维度数                       |
| `--random_state`     | int  | 42     | 任意整数 | UMAP 随机种子，用于结果可复现             |

> **注意**：BERTopic 不使用 `--epochs`、`--batch_size`、`--learning_rate` 或其他神经训练参数。  
> `--num_topics` 可选；设为 `None` 可自动检测主题数。

---

## 早停

| 参数 | 类型 | 默认值 | 范围 | 描述 |
|-----------|------|---------|-------|-------------|
| `--patience` | 整数 | `10` | 1-50 | 早停前等待的轮数 |
| `--no_early_stopping` | 标志 | False | 不适用 | 禁用早停 |

## 硬件配置

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `--gpu` | 整数 | `0` | GPU设备ID |

## 输出配置

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `--language` | 字符串 | `en` | 可视化语言：`en` 或 `zh` |

## 流程控制

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `--skip-train` | 标志 | False | 跳过训练，仅评估 |
| `--skip-eval` | 标志 | False | 跳过评估 |
| `--skip-viz` | 标志 | False | 跳过可视化 |
| `--check-only` | 标志 | False | 仅检查数据文件 |
| `--prepare` | 标志 | False | 训练前运行预处理 |

---

## 示例

**基本THETA训练：**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --epochs 100 \
    --gpu 0
```

**多个基线模型：**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models lda,etm,ctm \
    --num_topics 20 \
    --epochs 100 \
    --gpu 0
```

**自定义超参数：**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 30 \
    --epochs 150 \
    --batch_size 32 \
    --hidden_dim 768 \
    --learning_rate 0.001 \
    --kl_start 0.0 \
    --kl_end 1.0 \
    --kl_warmup 80 \
    --patience 15 \
    --gpu 0
```

**评估现有模型：**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --skip-train \
    --gpu 0
```

---

## 输出文件

**THETA模型：**
```
./result/{model_size}/{dataset}/{mode}/
├── checkpoints/
│   ├── best_model.pt
│   └── training_history.json
├── metrics/
│   └── evaluation_results.json
└── visualizations/
    ├── topic_words_bars.png
    ├── topic_similarity.png
    ├── doc_topic_umap.png
    ├── topic_wordclouds.png
    ├── metrics.png
    └── pyldavis.html
```

**基线模型：**
```
./result/baseline/{dataset}/{model}/K{num_topics}/
├── checkpoints/
├── metrics/
└── visualizations/
```

---

## 返回码

| 退出码 | 含义 |
|-----------|---------|
| 0 | 成功 |
| 1 | 一般错误 |
| 2 | 文件未找到 |
| 3 | 无效参数 |
| 4 | CUDA内存不足 |
| 5 | 收敛失败 |