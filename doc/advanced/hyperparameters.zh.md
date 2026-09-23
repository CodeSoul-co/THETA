# 超参数调优

**[English](hyperparameters.md)** | **[中文](hyperparameters.zh.md)**

---

优化THETA超参数的系统指南。

---

## 参数参考

### 通用参数

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

### 各模型额外参数

#### THETA

除通用参数外的额外参数：

| 参数           | 类型  | 默认值      | 范围                                      | 描述           |
| -------------- | ----- | ----------- | ----------------------------------------- | -------------- |
| `--model_size` | str   | `0.6B`      | `0.6B` / `4B` / `8B`                      | Qwen 模型规格  |
| `--mode`       | str   | `zero_shot` | `zero_shot` / `supervised` / `unsupervised` | 嵌入模式       |
| `--kl_start`   | float | 0.0         | 0–1                                       | KL 退火起始权重 |
| `--kl_end`     | float | 1.0         | 0–1                                       | KL 退火终止权重 |
| `--kl_warmup`  | int   | 50          | 0–epochs                                  | KL 预热轮数    |
| `--language`   | str   | `zh`        | `en` / `zh`                               | 可视化语言     |

#### LDA

| 参数         | 类型  | 默认值   | 范围   | 描述                  |
| ------------ | ----- | -------- | ------ | --------------------- |
| `--max_iter` | int   | 100      | 10–500 | 最大 EM 迭代次数      |
| `--alpha`    | float | 1/K（自动） | >0     | 文档-主题狄利克雷先验 |

#### HDP

| 参数           | 类型  | 默认值 | 范围   | 描述                           |
| -------------- | ----- | ------ | ------ | ------------------------------ |
| `--max_topics` | int   | 150    | 50–300 | 主题数上限（替代 `--num_topics`） |
| `--alpha`      | float | 1.0    | >0     | 文档级集中参数                 |

#### STM

| 参数         | 类型 | 默认值 | 范围   | 描述             |
| ------------ | ---- | ------ | ------ | ---------------- |
| `--max_iter` | int  | 100    | 10–500 | 最大 EM 迭代次数 |

#### BTM

| 参数       | 类型  | 默认值 | 范围   | 描述                           |
| ---------- | ----- | ------ | ------ | ------------------------------ |
| `--n_iter` | int   | 100    | 10–500 | Gibbs 采样迭代次数（替代 `--epochs`） |
| `--alpha`  | float | 1.0    | >0     | 主题分布狄利克雷先验           |
| `--beta`   | float | 0.01   | >0     | 词分布狄利克雷先验             |

#### ETM

| 参数              | 类型 | 默认值 | 范围    | 描述                   |
| ----------------- | ---- | ------ | ------- | ---------------------- |
| `--embedding_dim` | int  | 300    | 50–1024 | 词嵌入维度（Word2Vec） |

#### CTM

| 参数               | 类型 | 默认值     | 范围                   | 描述                                            |
| ------------------ | ---- | ---------- | ---------------------- | ----------------------------------------------- |
| `--inference_type` | str  | `zeroshot` | `zeroshot` / `combined` | 推理模式：仅 SBERT 或 SBERT + BOW               |
| `--hidden_dim`     | int  | 100        | 32–1024                | 覆盖通用默认值（512 → 100）                     |

#### DTM

| 参数              | 类型 | 默认值 | 范围    | 描述           |
| ----------------- | ---- | ------ | ------- | -------------- |
| `--embedding_dim` | int  | 300    | 50–1024 | 词嵌入维度     |

> **注意**：DTM 不使用 `--num_layers`、`--dropout` 或 `--patience`。  
> **数据要求**：DTM 需要数据包含 `timestamp` 列，训练前需运行 `python prepare_data.py --dataset your_data --model dtm`

#### NVDM / GSM / ProdLDA

无额外参数 — 所有设置由通用参数覆盖。  
> **注意**：这些模型的 `--hidden_dim` 默认为 256。

#### BERTopic

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

## 学习率调度

**保守方法（训练不稳定时）：**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --learning_rate 0.0005 \
    --epochs 150 \
    --gpu 0
```

**标准方法：**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --learning_rate 0.002 \
    --epochs 100 \
    --gpu 0
```

**激进方法（收敛慢时）：**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 0.6B \
    --mode zero_shot \
    --num_topics 20 \
    --learning_rate 0.01 \
    --epochs 80 \
    --gpu 0
```

监控训练损失曲线以确定是否需要调整。

---

## 批大小优化

| 批大小 | 优势 | 劣势 |
|-----------|-----------|---------------|
| 32 | 内存占用低，探索性好 | 更新噪声大，收敛慢 |
| 64 | 平衡（默认） | — |
| 128 | 更新稳定，轮次更快 | 内存占用高，可能过拟合 |

---

## KL退火策略

**无退火（立即完全KL）：**
`--kl_start 1.0 --kl_end 1.0 --kl_warmup 0`
风险：后验坍塌，主题质量差

**标准退火（推荐）：**
`--kl_start 0.0 --kl_end 1.0 --kl_warmup 50`

**慢速退火（复杂数据）：**
`--kl_start 0.0 --kl_end 1.0 --kl_warmup 80`

**部分退火（微调时）：**
`--kl_start 0.2 --kl_end 0.8 --kl_warmup 40`

---

## 隐藏层维度调优

| 隐藏层维度 | 适用场景 |
|-----------|----------|
| 256 | 小型数据集或内存受限 |
| 512 | 大多数应用的默认选择 |
| 1024 | 显存允许的大型复杂数据集 |

---

## 早停配置

| 耐心值 | 行为 |
|----------|----------|
| 5 | 验证损失平台期时快速停止 |
| 10 | 默认设置 |
| 20 | 停止前允许更长时间训练 |
| 禁用（`--no_early_stopping`） | 训练所有指定轮次 |

---

## 词汇表大小选择

| 语料库规模 | 词汇表大小 | 覆盖率 |
|------------|----------------|----------|
| < 1千篇文档 | 2000-3000 | ~85% |
| 1千-1万篇文档 | 5000 | ~90% |
| 1万-10万篇文档 | 8000-10000 | ~92% |
| > 10万篇文档 | 10000-15000 | ~95% |

---

## 使用不同模型规模

### 扩展策略

**开发流程：**
1. 从0.6B模型开始
2. 优化超参数
3. 扩展到4B用于生产
4. 如需最终结果使用8B

**快速比较：**
```bash
for size in 0.6B 4B 8B; do
    python run_pipeline.py \
        --dataset my_dataset \
        --models theta \
        --model_size $size \
        --mode zero_shot \
        --num_topics 20 \
        --gpu 0
done
```

### 质量与成本分析

**0.6B → 4B：**
- 主题多样性：+3-5%
- 连贯性（NPMI）：+10-15%
- 训练时间：+60-80%

**4B → 8B：**
- 主题多样性：+1-2%
- 连贯性（NPMI）：+5-8%
- 训练时间：+80-100%

收益递减表明4B通常是生产环境的最佳选择。

---

## 网格搜索

系统性超参数探索：

```bash
#!/bin/bash
topics=(15 20 25 30)
learning_rates=(0.001 0.002 0.005)
hidden_dims=(256 512 768)

for K in "${topics[@]}"; do
    for lr in "${learning_rates[@]}"; do
        for hd in "${hidden_dims[@]}"; do
            echo "训练 K=$K, lr=$lr, hd=$hd"
            
            python run_pipeline.py \
                --dataset my_dataset \
                --models theta \
                --model_size 0.6B \
                --mode zero_shot \
                --num_topics $K \
                --learning_rate $lr \
                --hidden_dim $hd \
                --epochs 100 \
                --batch_size 64 \
                --gpu 0

            mkdir -p results_grid/K${K}_lr${lr}_hd${hd}
            cp -r result/0.6B/my_dataset/zero_shot/* results_grid/K${K}_lr${lr}_hd${hd}/
        done
    done
done
```

---

## 批量处理多个数据集

```bash
#!/bin/bash
datasets=("news" "reviews" "papers" "social")

for dataset in "${datasets[@]}"; do
    echo "处理 $dataset..."
    
    python prepare_data.py \
        --dataset $dataset \
        --model theta \
        --model_size 0.6B \
        --mode zero_shot \
        --vocab_size 5000 \
        --gpu 0
    
    python run_pipeline.py \
        --dataset $dataset \
        --models theta \
        --model_size 0.6B \
        --mode zero_shot \
        --num_topics 20 \
        --gpu 0
done
```

---

## 多GPU并行处理

```bash
# 终端1
CUDA_VISIBLE_DEVICES=0 python run_pipeline.py \
    --dataset dataset1 --models theta --gpu 0 &

# 终端2  
CUDA_VISIBLE_DEVICES=1 python run_pipeline.py \
    --dataset dataset2 --models theta --gpu 0 &

# 终端3
CUDA_VISIBLE_DEVICES=2 python run_pipeline.py \
    --dataset dataset3 --models theta --gpu 0 &
```

每个进程使用不同的GPU。