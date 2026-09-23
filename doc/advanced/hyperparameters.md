# Hyperparameter Tuning

Systematic guide to optimizing THETA hyperparameters.

---

## Parameter Reference

### Common Parameters

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

### Model-Specific Parameters

#### THETA

Additional parameters beyond common defaults:

| Parameter      | Type  | Default     | Range                               | Description             |
| -------------- | ----- | ----------- | ----------------------------------- | ----------------------- |
| `--model_size` | str   | `0.6B`      | `0.6B` / `4B` / `8B`                | Qwen model size         |
| `--mode`       | str   | `zero_shot` | `zero_shot` / `supervised` / `unsupervised` | Embedding mode          |
| `--kl_start`   | float | 0.0         | 0–1                                 | KL annealing start weight |
| `--kl_end`     | float | 1.0         | 0–1                                 | KL annealing end weight |
| `--kl_warmup`  | int   | 50          | 0–epochs                            | KL warmup epochs        |
| `--language`   | str   | `zh`        | `en` / `zh`                         | Visualization language  |

#### LDA

| Parameter    | Type  | Default  | Range  | Description                    |
| ------------ | ----- | -------- | ------ | ------------------------------ |
| `--max_iter` | int   | 100      | 10–500 | Maximum EM iterations          |
| `--alpha`    | float | 1/K (auto) | >0     | Document-topic Dirichlet prior |

#### HDP

| Parameter      | Type  | Default | Range  | Description                            |
| -------------- | ----- | ------- | ------ | -------------------------------------- |
| `--max_topics` | int   | 150     | 50–300 | Upper bound on number of topics (replaces `--num_topics`) |
| `--alpha`      | float | 1.0     | >0     | Document-level concentration parameter |

#### STM

| Parameter    | Type | Default | Range  | Description           |
| ------------ | ---- | ------- | ------ | --------------------- |
| `--max_iter` | int  | 100     | 10–500 | Maximum EM iterations |

#### BTM

| Parameter  | Type  | Default | Range  | Description                            |
| ---------- | ----- | ------- | ------ | -------------------------------------- |
| `--n_iter` | int   | 100     | 10–500 | Gibbs sampling iterations (replaces `--epochs`) |
| `--alpha`  | float | 1.0     | >0     | Topic distribution Dirichlet prior     |
| `--beta`   | float | 0.01    | >0     | Word distribution Dirichlet prior      |

#### ETM

| Parameter         | Type | Default | Range   | Description                    |
| ----------------- | ---- | ------- | ------- | ------------------------------ |
| `--embedding_dim` | int  | 300     | 50–1024 | Word embedding dimension (Word2Vec) |

#### CTM

| Parameter          | Type | Default    | Range                   | Description                                         |
| ------------------ | ---- | ---------- | ----------------------- | --------------------------------------------------- |
| `--inference_type` | str  | `zeroshot` | `zeroshot` / `combined` | Inference mode: SBERT only or SBERT + BOW           |
| `--hidden_dim`     | int  | 100        | 32–1024                 | Overrides common default (512 → 100)                |

#### DTM

| Parameter         | Type | Default | Range   | Description          |
| ----------------- | ---- | ------- | ------- | -------------------- |
| `--embedding_dim` | int  | 300     | 50–1024 | Word embedding dimension |

> **Note**: DTM does not use `--num_layers`, `--dropout`, or `--patience`.  
> **Data requirement**: DTM requires a `timestamp` column. Run `python prepare_data.py --dataset your_data --model dtm` before training.

#### NVDM / GSM / ProdLDA

No additional parameters — all settings covered by common defaults.  
> **Note**: `--hidden_dim` defaults to 256 for these models.

#### BERTopic

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

## Tuning Strategies

### Learning Rate Scheduling

**Conservative approach (unstable training):**
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

**Standard approach:**
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

**Aggressive approach (slow convergence):**
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

Monitor training loss curves to determine if adjustment is needed.

---

## Batch Size Optimization

| Batch Size | Advantages | Disadvantages |
|-----------|-----------|---------------|
| 32 | Lower memory, better exploration | Noisy updates, slower convergence |
| 64 | Balanced (default) | — |
| 128 | Stable updates, faster epochs | Higher memory, may overfit |

---

## KL Annealing Strategies

**No annealing (immediate full KL):**
`--kl_start 1.0 --kl_end 1.0 --kl_warmup 0`
Risk: Posterior collapse, poor topic quality

**Standard annealing (recommended):**
`--kl_start 0.0 --kl_end 1.0 --kl_warmup 50`

**Slow annealing (complex data):**
`--kl_start 0.0 --kl_end 1.0 --kl_warmup 80`

**Partial annealing (fine-tuning):**
`--kl_start 0.2 --kl_end 0.8 --kl_warmup 40`

---

## Hidden Dimension Tuning

| Hidden Dim | Use Case |
|-----------|----------|
| 256 | Small datasets or memory constrained |
| 512 | Default choice for most applications |
| 1024 | Large complex datasets when VRAM permits |

---

## Early Stopping Configuration

| Patience | Behavior |
|----------|----------|
| 5 | Stops quickly if validation loss plateaus |
| 10 | Default setting |
| 20 | Allows longer training before stopping |
| Disabled (`--no_early_stopping`) | Trains for all specified epochs |

---

## Vocabulary Size Selection

| Corpus Size | Vocabulary Size | Coverage |
|------------|----------------|----------|
| < 1K docs | 2000-3000 | ~85% |
| 1K-10K docs | 5000 | ~90% |
| 10K-100K docs | 8000-10000 | ~92% |
| > 100K docs | 10000-15000 | ~95% |

---

## Using Different Model Sizes

### Scaling Strategy

**Development workflow:**
1. Start with 0.6B model
2. Optimize hyperparameters
3. Scale to 4B for production
4. Use 8B for final results if needed

**Quick comparison:**
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

### Quality vs Cost Analysis

**0.6B → 4B:**
- Topic diversity: +3-5%
- Coherence (NPMI): +10-15%
- Training time: +60-80%

**4B → 8B:**
- Topic diversity: +1-2%
- Coherence (NPMI): +5-8%
- Training time: +80-100%

Diminishing returns suggest 4B is often the best choice for production.

---

## Grid Search

Systematic hyperparameter exploration:

```bash
#!/bin/bash
topics=(15 20 25 30)
learning_rates=(0.001 0.002 0.005)
hidden_dims=(256 512 768)

for K in "${topics[@]}"; do
    for lr in "${learning_rates[@]}"; do
        for hd in "${hidden_dims[@]}"; do
            echo "Training K=$K, lr=$lr, hd=$hd"
            
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

## Batch Processing Multiple Datasets

```bash
#!/bin/bash
datasets=("news" "reviews" "papers" "social")

for dataset in "${datasets[@]}"; do
    echo "Processing $dataset..."
    
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

## Parallel Processing on Multiple GPUs

```bash
# Terminal 1
CUDA_VISIBLE_DEVICES=0 python run_pipeline.py \
    --dataset dataset1 --models theta --gpu 0 &

# Terminal 2  
CUDA_VISIBLE_DEVICES=1 python run_pipeline.py \
    --dataset dataset2 --models theta --gpu 0 &

# Terminal 3
CUDA_VISIBLE_DEVICES=2 python run_pipeline.py \
    --dataset dataset3 --models theta --gpu 0 &
```

Each process uses a different GPU.
