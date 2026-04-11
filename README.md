<div align="center">

<img src="assets/THETA.png" width="40%" alt="THETA Logo"/>

<h1>THETA (θ)</h1>

[![Platform](https://img.shields.io/badge/Platform-theta.code--soul.com-blue?style=flat-square)](https://theta.code-soul.com/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-CodeSoulco%2FTHETA-yellow?style=flat-square)](https://huggingface.co/CodeSoulco/THETA)
[![Paper](https://img.shields.io/badge/arXiv-2603.05972-b31b1b.svg)](https://arxiv.org/abs/2603.05972)

**English** | [中文](README_zh.md)

**THETA (θ) is a low-barrier, high-performance LLM-enhanced topic analysis platform for social science research.**

</div>

---

## Table of Contents

1. [Quick Start: 5-Minute Setup](#quick-start-5-minute-setup)
2. [Configuration System: From Hardware to Experiments](#configuration-system-from-hardware-to-experiments)
3. [Running Modes: Beginner vs Expert](#running-modes-beginner-vs-expert)
4. [Output Map: Where Are the Results?](#output-map-where-are-the-results)
5. [Scientific Evaluation Standards](#scientific-evaluation-standards)
6. [Supported Models](#supported-models)
7. [Training Parameters Reference](#training-parameters-reference)
8. [FAQ](#faq)

---

## Quick Start: 5-Minute Setup

### Step 1: Clone Repository

```bash
git clone https://github.com/CodeSoul-co/THETA.git
cd THETA
```

### Step 2: Environment Isolation (Conda)

```bash
conda create -n theta python=3.10 -y
conda activate theta
```

### Step 3: Install Dependencies + Download Models

```bash
bash scripts/env_setup.sh
```

### Step 4: Configure Environment Variables

```bash
# Copy configuration template
cp .env.example .env

# Edit .env file to configure model paths
# At minimum, configure: QWEN_MODEL_0_6B and SBERT_MODEL_PATH
```

### Step 5: Load Environment Variables

```bash
# If you encounter "$'\r': command not found" error, fix Windows line endings first
sed -i 's/\r$//' scripts/env_setup.sh

# Load environment variables to current shell (required for subsequent scripts)
source scripts/env_setup.sh
```

**Model Download Links**:

| Model | Purpose | Download Link |
|-------|---------|---------------|
| **Qwen3-Embedding-0.6B** | THETA document embedding | [ModelScope](https://www.modelscope.cn/models/Qwen/Qwen3-Embedding-0.6B) |
| **all-MiniLM-L6-v2** | CTM/SBERT embedding | [HuggingFace](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |

Place downloaded models in the `models/` directory:

```
models/
├── qwen3_embedding_0.6B/
└── sbert/sentence-transformers/all-MiniLM-L6-v2/
```

---

## Configuration System: From Hardware to Experiments

THETA uses a **layered configuration** architecture for flexible control from hardware paths to experiment parameters.

### Core Configuration File `.env` (Hardware Physical Paths)

Create a `.env` file (refer to `.env.example`) with **required** settings:

```bash
# Required: Qwen embedding model path
QWEN_MODEL_0_6B=./models/qwen3_embedding_0.6B
# QWEN_MODEL_4B=./models/qwen3_embedding_4B
# QWEN_MODEL_8B=./models/qwen3_embedding_8B

# Required: SBERT model path (needed for CTM/BERTopic)
SBERT_MODEL_PATH=./models/sbert/sentence-transformers/all-MiniLM-L6-v2

# Required: Data and result directories
DATA_DIR=./data
WORKSPACE_DIR=./data/workspace
RESULT_DIR=./result
```

### Experiment Parameters `config/default.yaml` (Default Hyperparameters)

This file stores default training parameters for all models:

```yaml
# Common training parameters
training:
  epochs: 100
  batch_size: 64
  learning_rate: 0.002

# THETA-specific parameters
theta:
  num_topics: 20
  hidden_dim: 512
  model_size: 0.6B

# Visualization settings
visualization:
  language: en    # English visualization
  dpi: 150
```

### Priority Rule

Parameter priority order:

```
CLI arguments  >  YAML defaults  >  Code fallback values
```

Example:
- `--num_topics 50` overrides `num_topics: 20` in YAML
- If neither CLI nor YAML specifies a value, code defaults are used

---

## Running Modes: Beginner vs Expert

### Beginner Mode: One-Click Automation (Bash Scripts)

Just prepare your data, and the script will automatically complete **cleaning → preprocessing → training → evaluation → visualization**:

```bash
# One-click training (specify language parameter)
bash scripts/quick_start.sh my_dataset --language chinese
bash scripts/quick_start.sh my_dataset --language english
```

**Data Preparation Requirements**:
- Place raw documents in `data/{dataset}/` directory
- Supported formats: `.txt`, `.csv`, `.docx`, `.pdf`

### Expert Mode: Surgical-Level Tuning (Python CLI)

Directly call Python scripts with precise control over every parameter:

```bash
# Train LDA, override default parameters
python src/models/run_pipeline.py \
    --dataset my_dataset \
    --models lda \
    --num_topics 50 \
    --learning_rate 0.01 \
    --language chinese

# Train THETA with 4B model
python src/models/run_pipeline.py \
    --dataset my_dataset \
    --models theta \
    --model_size 4B \
    --num_topics 30 \
    --epochs 200
```

**Key Module Entry Points**:

| Module | Entry Script | Function |
|--------|--------------|----------|
| Data Cleaning | `src/models/dataclean/main.py` | Text cleaning, tokenization, stopword removal |
| Data Preprocessing | `src/models/prepare_data.py` | Generate BOW matrix and embedding vectors |
| Model Training | `src/models/run_pipeline.py` | Training, evaluation, visualization all-in-one |

---

## Output Map: Where Are the Results?

THETA model and baseline model result paths are **different**, please note the distinction:

### THETA Model Results

```
result/{dataset}/{model_size}/theta/exp_{timestamp}/
├── config.json                     # Experiment configuration
├── metrics.json                    # 7 evaluation metrics
├── data/                           # Preprocessed data
│   ├── bow/                        # BOW matrix
│   │   ├── bow_matrix.npy
│   │   ├── vocab.txt
│   │   ├── vocab.json
│   │   └── vocab_embeddings.npy
│   └── embeddings/                 # Qwen document embeddings
│       ├── embeddings.npy
│       └── metadata.json
├── theta/                          # Model parameters (fixed filenames, no timestamp)
│   ├── theta.npy                   # Document-topic distribution (D × K)
│   ├── beta.npy                    # Topic-word distribution (K × V)
│   ├── topic_embeddings.npy        # Topic embedding vectors
│   ├── topic_words.json            # Topic word list
│   ├── training_history.json       # Training history
│   └── etm_model.pt                # PyTorch model
└── {lang}/                         # Visualization output (zh or en)
    ├── global/                     # Global charts
    │   ├── topic_table.csv
    │   ├── topic_network.png
    │   ├── topic_similarity.png
    │   ├── topic_wordcloud.png
    │   ├── 7_core_metrics.png
    │   └── ...
    └── topic/                      # Topic details
        ├── topic_1/
        │   └── word_importance.png
        └── ...
```

### Baseline Model Results (LDA, CTM, BTM, etc.)

```
result/{dataset}/{user_id}/{model}/exp_{timestamp}/
├── config.json                     # Experiment configuration
├── metrics_k{K}.json               # 7 evaluation metrics
├── {model}/                        # Model parameters
│   ├── theta_k{K}.npy              # Document-topic distribution
│   ├── beta_k{K}.npy               # Topic-word distribution
│   ├── model_k{K}.pkl              # Model file
│   └── topic_words_k{K}.json
├── {lang}/                         # Visualization directory (zh or en)
│   ├── global/                     # Global comparison charts
│   │   ├── topic_network.png
│   │   ├── topic_similarity.png
│   │   └── ...
│   └── topic/                      # Topic details
│       ├── topic_0/
│       │   ├── wordcloud.png
│       │   └── word_distribution.png
│       └── ...
└── README.md                       # Experiment summary
```

### Path Summary

| Model Type | Result Path |
|------------|-------------|
| THETA | `result/{dataset}/{model_size}/theta/exp_{timestamp}/` |
| Baseline Models | `result/{dataset}/{user_id}/{model}/exp_{timestamp}/` |

---

## Scientific Evaluation Standards

THETA enforces **7 Gold Standard Metrics** to ensure evaluation alignment across all models (THETA and 12 baselines):

| Metric | Full Name | Description | Ideal Value |
|--------|-----------|-------------|-------------|
| **TD** | Topic Diversity | Topic diversity, measures uniqueness of topic words | ↑ Higher is better |
| **iRBO** | Inverse Rank-Biased Overlap | Inverse rank-biased overlap, measures inter-topic differences | ↑ Higher is better |
| **NPMI** | Normalized PMI | Normalized pointwise mutual information, measures topic word co-occurrence | ↑ Higher is better |
| **C_V** | C_V Coherence | Sliding window-based coherence | ↑ Higher is better |
| **UMass** | UMass Coherence | Document co-occurrence-based coherence | ↑ Higher is better (negative) |
| **Exclusivity** | Topic Exclusivity | Topic exclusivity, whether words belong to single topics | ↑ Higher is better |
| **PPL** | Perplexity | Perplexity, model fitting ability | ↓ Lower is better |

> **Note**: Significance data is only used for visualization, not included in core evaluation metrics.

---

## Supported Models

### Model Overview

| Model | Type | Description | Auto Topics | Best Use Case |
|-------|------|-------------|-------------|---------------|
| `theta` | Neural | THETA model with Qwen embeddings | No | General purpose, high quality |
| `lda` | Traditional | Latent Dirichlet Allocation | No | Fast baseline, highly interpretable |
| `hdp` | Traditional | Hierarchical Dirichlet Process | **Yes** | Unknown topic count |
| `stm` | Traditional | Structural Topic Model | No | **Requires covariates** |
| `btm` | Traditional | Biterm Topic Model | No | Short texts (tweets, titles) |
| `etm` | Neural | Embedded Topic Model | No | Word embedding integration |
| `ctm` | Neural | Contextualized Topic Model | No | Semantic understanding |
| `dtm` | Neural | Dynamic Topic Model | No | Time series analysis |
| `nvdm` | Neural | Neural Variational Document Model | No | VAE baseline |
| `gsm` | Neural | Gaussian Softmax Model | No | Better topic separation |
| `prodlda` | Neural | Product of Experts LDA | No | State-of-the-art neural LDA |
| `bertopic` | Neural | BERT-based topic modeling | **Yes** | Clustering-based topics |

### Model Selection Guide

```
┌─────────────────────────────────────────────────────────────────┐
│ Do you know the number of topics?                               │
│   ├─ No  → Use HDP or BERTopic (auto-detect topic count)       │
│   └─ Yes → Continue below                                       │
├─────────────────────────────────────────────────────────────────┤
│ What is your text length?                                       │
│   ├─ Short texts (tweets, titles) → Use BTM                    │
│   └─ Normal/Long texts → Continue below                         │
├─────────────────────────────────────────────────────────────────┤
│ Do you have document-level metadata (covariates)?               │
│   ├─ Yes → Use STM (models how metadata affects topics)         │
│   └─ No  → Continue below                                       │
├─────────────────────────────────────────────────────────────────┤
│ Do you have time series data?                                   │
│   ├─ Yes → Use DTM                                              │
│   └─ No  → Continue below                                       │
├─────────────────────────────────────────────────────────────────┤
│ What is your priority?                                          │
│   ├─ Speed      → Use LDA (fastest)                            │
│   ├─ Quality    → Use THETA (best with Qwen embeddings)        │
│   └─ Comparison → Use multiple models: lda,nvdm,prodlda,theta  │
└─────────────────────────────────────────────────────────────────┘
```



### Training Parameters Reference

#### THETA Parameters

| Parameter         | Type  | Default     | Range                                     | Description                      |
| ----------------- | ----- | ----------- | ----------------------------------------- | -------------------------------- |
| `--model_size`    | str   | `0.6B`      | `0.6B`, `4B`, `8B`                        | Qwen model size                  |
| `--mode`          | str   | `zero_shot` | `zero_shot`, `supervised`, `unsupervised` | Embedding mode                   |
| `--num_topics`    | int   | 20          | 5–100                                     | Number of topics K               |
| `--num_layers`    | int   | 2           | 1–5                                       | Number of encoder hidden layers  |
| `--hidden_dim`    | int   | 512         | 128–2048                                  | Neurons per hidden layer         |
| `--epochs`        | int   | 100         | 10–500                                    | Training epochs                  |
| `--batch_size`    | int   | 64          | 8–512                                     | Batch size                       |
| `--learning_rate` | float | 0.002       | 1e-5–0.1                                  | Learning rate                    |
| `--dropout`       | float | 0.2         | 0–0.9                                     | Encoder dropout rate             |
| `--kl_start`      | float | 0.0         | 0–1                                       | KL annealing start weight        |
| `--kl_end`        | float | 1.0         | 0–1                                       | KL annealing end weight          |
| `--kl_warmup`     | int   | 50          | 0–epochs                                  | KL warmup epochs                 |
| `--patience`      | int   | 10          | 1–50                                      | Early stopping patience          |
| `--language`      | str   | `en`        | `en`, `zh`                                | Visualization language           |

#### Baseline Model Parameters

**LDA**

| Parameter      | Type  | Default    | Range      | Description                    |
| -------------- | ----- | ---------- | ---------- | ------------------------------ |
| `--num_topics` | int   | 20         | 5–100      | Number of topics K             |
| `--max_iter`   | int   | 100        | 10–500     | Maximum EM iterations          |
| `--alpha`      | float | auto (1/K) | >0         | Document-topic Dirichlet prior |
| `--vocab_size` | int   | 5000       | 1000–20000 | Vocabulary size                |

**HDP**

| Parameter      | Type  | Default | Range      | Description                            |
| -------------- | ----- | ------- | ---------- | -------------------------------------- |
| `--max_topics` | int   | 150     | 50–300     | Upper bound on topic count             |
| `--alpha`      | float | 1.0     | >0         | Document-level concentration parameter |
| `--vocab_size` | int   | 5000    | 1000–20000 | Vocabulary size                        |

**STM**

| Parameter      | Type | Default | Range      | Description           |
| -------------- | ---- | ------- | ---------- | --------------------- |
| `--num_topics` | int  | 20      | 5–100      | Number of topics K    |
| `--max_iter`   | int  | 100     | 10–500     | Maximum EM iterations |
| `--vocab_size` | int  | 5000    | 1000–20000 | Vocabulary size       |

**BTM**

| Parameter      | Type  | Default | Range      | Description                            |
| -------------- | ----- | ------- | ---------- | -------------------------------------- |
| `--num_topics` | int   | 20      | 5–100      | Number of topics K                     |
| `--n_iter`     | int   | 100     | 10–500     | Gibbs sampling iterations              |
| `--alpha`      | float | 1.0     | >0         | Dirichlet prior for topic distribution |
| `--beta`       | float | 0.01    | >0         | Dirichlet prior for word distribution  |
| `--vocab_size` | int   | 5000    | 1000–20000 | Vocabulary size                        |

**ETM**

| Parameter         | Type  | Default | Range      | Description                         |
| ----------------- | ----- | ------- | ---------- | ----------------------------------- |
| `--num_topics`    | int   | 20      | 5–100      | Number of topics K                  |
| `--num_layers`    | int   | 2       | 1–5        | Number of encoder hidden layers     |
| `--hidden_dim`    | int   | 800     | 128–2048   | Neurons per hidden layer            |
| `--embedding_dim` | int   | 300     | 50–1024    | Word embedding dimension (Word2Vec) |
| `--epochs`        | int   | 100     | 10–500     | Training epochs                     |
| `--batch_size`    | int   | 64      | 8–512      | Batch size                          |
| `--learning_rate` | float | 0.002   | 1e-5–0.1   | Learning rate                       |
| `--dropout`       | float | 0.5     | 0–0.9      | Dropout rate                        |
| `--vocab_size`    | int   | 5000    | 1000–20000 | Vocabulary size                     |

**CTM**

| Parameter          | Type  | Default    | Range                  | Description                                         |
| ------------------ | ----- | ---------- | ---------------------- | --------------------------------------------------- |
| `--num_topics`     | int   | 20         | 5–100                  | Number of topics K                                  |
| `--inference_type` | str   | `zeroshot` | `zeroshot`, `combined` | `zeroshot` (SBERT only) or `combined` (SBERT + BOW) |
| `--num_layers`     | int   | 2          | 1–5                    | Number of encoder hidden layers                     |
| `--hidden_dim`     | int   | 100        | 32–1024                | Neurons per hidden layer                            |
| `--epochs`         | int   | 100        | 10–500                 | Training epochs                                     |
| `--batch_size`     | int   | 64         | 8–512                  | Batch size                                          |
| `--learning_rate`  | float | 0.002      | 1e-5–0.1               | Learning rate                                       |
| `--dropout`        | float | 0.2        | 0–0.9                  | Dropout rate                                        |
| `--vocab_size`     | int   | 5000       | 1000–20000             | Vocabulary size                                     |

**DTM**

| Parameter         | Type  | Default | Range      | Description                     |
| ----------------- | ----- | ------- | ---------- | ------------------------------- |
| `--num_topics`    | int   | 20      | 5–100      | Number of topics K              |
| `--num_layers`    | int   | 2       | 1–5        | Number of encoder hidden layers |
| `--hidden_dim`    | int   | 512     | 128–2048   | Neurons per hidden layer        |
| `--epochs`        | int   | 100     | 10–500     | Training epochs                 |
| `--batch_size`    | int   | 64      | 8–512      | Batch size                      |
| `--learning_rate` | float | 0.002   | 1e-5–0.1   | Learning rate                   |
| `--dropout`       | float | 0.2     | 0–0.9      | Dropout rate                    |
| `--vocab_size`    | int   | 5000    | 1000–20000 | Vocabulary size                 |

**NVDM / GSM / ProdLDA**

| Parameter         | Type  | Default | Range      | Description                     |
| ----------------- | ----- | ------- | ---------- | ------------------------------- |
| `--num_topics`    | int   | 20      | 5–100      | Number of topics K              |
| `--num_layers`    | int   | 2       | 1–5        | Number of encoder hidden layers |
| `--hidden_dim`    | int   | 256     | 128–2048   | Neurons per hidden layer        |
| `--epochs`        | int   | 100     | 10–500     | Training epochs                 |
| `--batch_size`    | int   | 64      | 8–512      | Batch size                      |
| `--learning_rate` | float | 0.002   | 1e-5–0.1   | Learning rate                   |
| `--dropout`       | float | 0.2     | 0–0.9      | Dropout rate                    |
| `--vocab_size`    | int   | 5000    | 1000–20000 | Vocabulary size                 |

**BERTopic**

| Parameter            | Type | Default | Range        | Description                                           |
| -------------------- | ---- | ------- | ------------ | ----------------------------------------------------- |
| `--num_topics`       | int  | auto    | ≥2 or `None` | Target topic count; `None` = automatic detection      |
| `--min_cluster_size` | int  | 10      | 2–100        | Minimum cluster size, controls topic granularity      |
| `--top_n_words`      | int  | 10      | 1–30         | Number of words per topic                             |
| `--n_neighbors`      | int  | 15      | 2–100        | UMAP neighbors, controls local vs global structure    |
| `--n_components`     | int  | 5       | 2–50         | UMAP output dimensionality                            |

---

## FAQ

### Data Requirements

**Q: What is the minimum number of documents required?**

A: **Minimum 5 documents**. Topic models need sufficient documents to learn meaningful topic distributions. Recommendations:
- Small experiments: 50+ documents
- Formal research: 500+ documents
- Large-scale analysis: 5000+ documents

**Q: What data formats are supported?**

A: Supports `.txt`, `.csv`, `.docx`, `.pdf`. CSV files need a `text` column (or specify another column via `--text_column`).

---

### Memory & Performance

**Q: What to do about Out of Memory (OOM)?**

A: When GPU memory is insufficient, adjust in this order:

| Stage | Parameter | Recommended Value |
|-------|-----------|-------------------|
| Embedding generation | `--batch_size` | 4–8 |
| THETA/Neural model training | `--batch_size` | 16–32 |
| Use smaller model | `--model_size` | `0.6B` instead of `4B` |

```bash
# Check GPU usage
nvidia-smi

# Kill zombie processes
kill -9 <PID>
```

**Q: Why is BTM training slow?**

A: BTM uses Gibbs sampling, with computation proportional to `biterm count × iterations`. For large datasets, it may take 30–90 minutes. Reduce iterations with `--n_iter 50` to speed up.

---

### Model Selection

**Q: What's the difference between ETM and DTM?**

A:
- **ETM**: Static topic model, learns fixed topics across the entire corpus
- **DTM**: Dynamic topic model, models topic evolution over time, **requires timestamp column**

**Q: Why was STM skipped?**

A: STM requires **covariates** (document-level metadata). If the dataset doesn't have configured covariates, STM is automatically skipped. Alternatives: use CTM or LDA.

**Q: How to choose the number of topics K?**

| Dataset Size | Recommended K |
|--------------|---------------|
| < 1000 docs | 5–15 |
| 1000–10000 | 10–30 |
| > 10000 | 20–50 |

You can also use `hdp` or `bertopic` to auto-detect topic count as a reference.

---

### Visualization

**Q: What does the `--language` parameter do?**

A: Controls the language of visualization charts:
- `chinese` or `zh`: Chinese chart titles and filenames (e.g., `主题网络图.png`)
- `english` or `en`: English chart titles and filenames (e.g., `topic_network.png`)

Only affects visualization, not model training or evaluation.

---

### Other

**Q: Is this project only for Qwen?**

A: No. Qwen is the default embedding model, but THETA is designed to be model-agnostic. You can adapt other embedding models (e.g., BERT, LLaMA).

**Q: How to add a custom dataset?**

A:
1. Place cleaned CSV in `data/{dataset}/` directory
2. Ensure CSV contains a `text` column
3. Run: `bash scripts/quick_start.sh {dataset} --language english`

---

## Citation

If you find **THETA** useful in your research, please consider citing our paper:

```bibtex
@article{duan2026theta,
  title={THETA: A Textual Hybrid Embedding-based Topic Analysis Framework and AI Scientist Agent for Scalable Computational Social Science},
  author={Codesoul.co},
  journal={arXiv preprint arXiv:2603.05972},
  year={2026},
  doi={10.48550/arXiv.2603.05972}
}
```

---

## Contact

For questions, please contact:
- duanzhenke@code-soul.com
- panjiqun@code-soul.com
- lixin@code-soul.com

---

## License

Apache-2.0
