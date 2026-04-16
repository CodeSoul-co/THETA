# run_pipeline.py

Unified training, evaluation, and visualization pipeline.

---

## Basic Usage

```bash
python run_pipeline.py --dataset DATASET --models MODELS [OPTIONS]
```

---

## Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `--dataset` | string | Dataset name |
| `--models` | string | Comma-separated model list: `theta,lda,hdp,stm,btm,etm,ctm,dtm,nvdm,gsm,prodlda,bertopic` |

---

## Common Parameters

Shared across all or most models. Parameters marked `*` apply to neural network–based models only.

| Parameter         | Type  | Default | Range      | Description                                           |
| ----------------- | ----- | ------- | ---------- | ----------------------------------------------------- |
| `--num_topics`    | int   | 20      | 5–100      | Number of topics K (upper bound for HDP; optional for BERTopic) |
| `--vocab_size`    | int   | 5000    | 1000–20000 | Vocabulary size                                       |
| `--epochs` *      | int   | 100     | 10–500     | Training epochs                                       |
| `--batch_size` *  | int   | 64      | 8–512      | Mini-batch size                                       |
| `--learning_rate` * | float | 0.002   | 1e-5–0.1   | Learning rate                                         |
| `--dropout` *     | float | 0.2     | 0–0.9      | Encoder dropout rate                                  |
| `--hidden_dim` *  | int   | 512     | 128–2048   | Hidden units per layer (NVDM/GSM/ProdLDA default: 256) |
| `--num_layers` *  | int   | 2       | 1–5        | Number of encoder hidden layers                       |
| `--patience` *    | int   | 10      | 1–50       | Early stopping patience                               |

---

## Model-Specific Additional Parameters

### THETA

Additional parameters beyond common defaults:

| Parameter      | Type  | Default     | Range                               | Description             |
| -------------- | ----- | ----------- | ----------------------------------- | ----------------------- |
| `--model_size` | str   | `0.6B`      | `0.6B` / `4B` / `8B`                | Qwen model size         |
| `--mode`       | str   | `zero_shot` | `zero_shot` / `supervised` / `unsupervised` | Embedding mode          |
| `--kl_start`   | float | 0.0         | 0–1                                 | KL annealing start weight |
| `--kl_end`     | float | 1.0         | 0–1                                 | KL annealing end weight |
| `--kl_warmup`  | int   | 50          | 0–epochs                            | KL warmup epochs        |
| `--language`   | str   | `zh`        | `en` / `zh`                         | Visualization language  |

### LDA

| Parameter    | Type  | Default  | Range  | Description                    |
| ------------ | ----- | -------- | ------ | ------------------------------ |
| `--max_iter` | int   | 100      | 10–500 | Maximum EM iterations          |
| `--alpha`    | float | 1/K (auto) | >0     | Document-topic Dirichlet prior |

### HDP

| Parameter      | Type  | Default | Range  | Description                            |
| -------------- | ----- | ------- | ------ | -------------------------------------- |
| `--max_topics` | int   | 150     | 50–300 | Upper bound on number of topics (replaces `--num_topics`) |
| `--alpha`      | float | 1.0     | >0     | Document-level concentration parameter |

### STM

| Parameter    | Type | Default | Range  | Description           |
| ------------ | ---- | ------- | ------ | --------------------- |
| `--max_iter` | int  | 100     | 10–500 | Maximum EM iterations |

### BTM

| Parameter  | Type  | Default | Range  | Description                            |
| ---------- | ----- | ------- | ------ | -------------------------------------- |
| `--n_iter` | int   | 100     | 10–500 | Gibbs sampling iterations (replaces `--epochs`) |
| `--alpha`  | float | 1.0     | >0     | Topic distribution Dirichlet prior     |
| `--beta`   | float | 0.01    | >0     | Word distribution Dirichlet prior      |

### ETM

| Parameter         | Type | Default | Range   | Description                    |
| ----------------- | ---- | ------- | ------- | ------------------------------ |
| `--embedding_dim` | int  | 300     | 50–1024 | Word embedding dimension (Word2Vec) |

### CTM

| Parameter          | Type | Default    | Range                   | Description                                         |
| ------------------ | ---- | ---------- | ----------------------- | --------------------------------------------------- |
| `--inference_type` | str  | `zeroshot` | `zeroshot` / `combined` | Inference mode: SBERT only or SBERT + BOW           |
| `--hidden_dim`     | int  | 100        | 32–1024                 | Overrides common default (512 → 100)                |

### DTM

| Parameter         | Type | Default | Range   | Description          |
| ----------------- | ---- | ------- | ------- | -------------------- |
| `--embedding_dim` | int  | 300     | 50–1024 | Word embedding dimension |

> **Note**: DTM does not use `--num_layers`, `--dropout`, or `--patience`.  
> **Data requirement**: DTM requires a `timestamp` column. Run `python prepare_data.py --dataset your_data --model dtm` before training.

### NVDM / GSM / ProdLDA

No additional parameters — all settings covered by common defaults.  
> **Note**: `--hidden_dim` defaults to 256 for these models.

### BERTopic

| Parameter            | Type | Default | Range    | Description                                      |
| -------------------- | ---- | ------- | -------- | ------------------------------------------------ |
| `--min_cluster_size` | int  | 10      | 2–100    | HDBSCAN minimum cluster size; controls topic granularity |
| `--min_samples`      | int  | None    | 1–100    | HDBSCAN min_samples (defaults to min_cluster_size) |
| `--top_n_words`      | int  | 10      | 1–30     | Top words displayed per topic                    |
| `--n_neighbors`      | int  | 15      | 2–100    | UMAP number of neighbors                         |
| `--n_components`     | int  | 5       | 2–50     | UMAP reduced dimensions                          |
| `--random_state`     | int  | 42      | any int  | UMAP random seed for reproducibility             |

> **Note**: BERTopic does not use `--epochs`, `--batch_size`, `--learning_rate`, or other neural training parameters.  
> `--num_topics` is optional; set to `None` for auto-detection.

---

## Pipeline Control Flags

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `--kl_start` | float | `0.0` | 0.0-1.0 | Initial KL divergence weight |
| `--kl_end` | float | `1.0` | 0.0-1.0 | Final KL divergence weight |
| `--kl_warmup` | int | `50` | 0-200 | Number of warmup epochs for KL annealing |

## Early Stopping

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `--patience` | int | `10` | 1-50 | Epochs to wait before early stopping |
| `--no_early_stopping` | flag | False | N/A | Disable early stopping |

## Hardware Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--gpu` | int | `0` | GPU device ID |

## Output Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--language` | string | `en` | Visualization language: `en` or `zh` |

## Pipeline Control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--skip-train` | flag | False | Skip training, evaluate only |
| `--skip-eval` | flag | False | Skip evaluation |
| `--skip-viz` | flag | False | Skip visualization |
| `--check-only` | flag | False | Check data files only |
| `--prepare` | flag | False | Run preprocessing before training |

---

## Examples

**Basic THETA training:**
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

**Multiple baseline models:**
```bash
python run_pipeline.py \
    --dataset my_dataset \
    --models lda,etm,ctm \
    --num_topics 20 \
    --epochs 100 \
    --gpu 0
```

**Custom hyperparameters:**
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

**Evaluate existing model:**
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

## Output Files

**THETA models:**
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

**Baseline models:**
```
./result/baseline/{dataset}/{model}/K{num_topics}/
├── checkpoints/
├── metrics/
└── visualizations/
```

---

## Return Codes

| Exit Code | Meaning |
|-----------|---------|
| 0 | Success |
| 1 | General error |
| 2 | File not found |
| 3 | Invalid parameters |
| 4 | CUDA out of memory |
| 5 | Convergence failure |
