<div align="center">

<img src="assets/THETA.png" width="40%" alt="THETA Logo"/>

<h1>THETA (θ)</h1>

[![Platform](https://img.shields.io/badge/Platform-theta.code--soul.com-blue?style=flat-square)](https://theta.code-soul.com/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-CodeSoulco%2FTHETA-yellow?style=flat-square)](https://huggingface.co/CodeSoulco/THETA)

**English** | [中文](README_zh.md)

**THETA (θ) is a low-barrier, high-performance agent-native topic analysis platform for social science research.**

</div>

## Four ways to use THETA

| Entry point | Getting started |
| --- | --- |
| CLI | `./theta`; see [Agent setup](agent/README.md) |
| Web | `./theta-web start`; see below |
| Windows / Mac App | Bundled Python; configure models in the workbench Settings. See [desktop build and usage](docs/desktop.md) |
| Skill | Use [THETA Workflow](skills/theta-workflow/SKILL.md) with a compatible agent |

## Start the Web Workbench (Conversation / Manual)

Run from the repository root on macOS / Linux for local development:

```bash
./theta-web start     # Build, check, and start the frontend and both APIs in the background
./theta-web status    # Show service health, logs, and data directories
./theta-web restart   # Rebuild and restart after updating code
./theta-web stop      # Stop web services; keep projects and training results
./theta-web check     # Build and check the Worker without starting training
```

Open **<http://127.0.0.1:4320/workbench?mode=conversation>** and switch between Conversation and Manual at the top. Closing the terminal does not stop the services; run `./theta-web start` again after rebooting your computer.

First-time setup requires **Node.js ≥ 22.13, pnpm, the Python compute environment, and private `agent/.env.local` configuration** ([Agent setup](agent/README.md)). Install dependencies once:

```bash
pnpm --dir agent install --frozen-lockfile
npm --prefix frontend ci
```

The launcher uses `frontend/`, the conversation API on `4318`, and the manual API on `4321`, preserving existing data in `.theta_agent/` and `.local/manual-workbench/`. No legacy services are required. **Do not delete these data directories or overwrite existing `.env.local` files.** Logs are in `.local/workbench/`. Port conflicts are reported without killing unrelated processes.

对话支持[受限代码预处理与会话持久化](agent/docs/conversation-preprocessing.md)：保留原始数据，保存派生数据、代码与校验报告，再由用户确认训练。输入框的文件引用移除后，数据仍保留在项目中；用户发送条数保存在会话记录中。

This is a local development entry point, not an unauthenticated production deployment. See [web setup and troubleshooting](frontend/README.md) for details. The independent CLI remains `./theta`.

对话模式的 **THETA 零样本分析**会先显示“本地／云端嵌入”选择卡，不会自动采用环境中的云端默认值。选定后才检查资源并生成独立的训练确认卡，确认后才执行；云端服务、发送范围与请求上限会在卡片中列明。选择与待确认状态保存在服务端，刷新或切换页面不会丢失。也可在选择卡出现后回复“本地嵌入”或“云端嵌入”。CTM、BERTopic 和 THETA 微调仍使用各自支持的本地嵌入，不提供无效的云端切换。以下计算引擎命令的默认参数与网页对话的显式选择流程不同。

对话模式与手动模式复用同一训练配置弹窗。Agent 预填一个或多个模型及各自参数，用户可增减模型、修改数据列与参数，再点击“确认并开始训练”。确认前不会训练；确认后弹窗收起，多个模型按顺序执行并分别保留结果。未提交的表单在当前浏览器保存，已确认的队列保存在服务端，刷新页面或重启服务不会重复提交训练。

> The quickstart below is for the computation engine, not the dual-mode web workbench. Historical migration notes are in the [architecture documentation](docs/architecture/agent-migration.md); distributed training setup is in [Python workers](trainning/worker/README.md).


---

## Table of Contents

1. [Quick Start: 5-Minute Setup](#quick-start-5-minute-setup)
2. [Agent Workflow Skill: THETA Workflow](#agent-workflow-skill-theta-workflow)
3. [Data Format Requirements](#data-format-requirements)
4. [Configuration System: From Hardware to Experiments](#configuration-system-from-hardware-to-experiments)
5. [Running Modes: Beginner vs Expert](#running-modes-beginner-vs-expert)
6. [Output Map: Where Are the Results?](#output-map-where-are-the-results)
7. [Scientific Evaluation Standards](#scientific-evaluation-standards)
8. [Supported Models](#supported-models)
9. [Training Parameters Reference](#training-parameters-reference)
10. [FAQ](#faq)

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

###  Automatic Model Download

THETA supports **automatic detection and download** of missing model weight files during training.

**Automatic Download** (Recommended): Simply run the training command, and the system will automatically detect and download missing models:

```bash
# CTM/BERTopic training - auto-download SBERT
bash scripts/train_baseline.sh ctm --dataset your_dataset --num_topics 20

# THETA training - auto-download Qwen
bash scripts/train_theta.sh --dataset your_dataset --model_size 0.6B
```

> **Note**: The first run may take a few minutes to download models. If automatic download fails due to network issues, please manually download from the links above.

---

## Agent Workflow Skill: THETA Workflow

This repository includes a portable agent workflow skill at [`skills/theta-workflow/`](skills/theta-workflow/). A standalone public copy is published at [CodeSoul-co/theta-skill](https://github.com/CodeSoul-co/theta-skill) for agents that want to install the workflow without cloning the full THETA project. The standalone repository also links back to [CodeSoul-co/THETA](https://github.com/CodeSoul-co/THETA), so both publication surfaces point to each other. Use it when you want an agent to guide the full THETA process: repository check, data confirmation, environment preflight, embedding selection, model recommendation, command review, training confirmation, result analysis, tuning, and reporting. The skill is plain Markdown plus a read-only Python helper, so it does not require Codex-specific APIs. It also includes separate Chinese and English user-question and confirmation flows.

Generic usage:

1. Point your agent at [`skills/theta-workflow/SKILL.md`](skills/theta-workflow/SKILL.md), or install/copy the whole `skills/theta-workflow/` folder into your agent's skill directory.
2. Ask the agent to use the THETA Workflow skill or instructions:

```text
Use the THETA Workflow skill in skills/theta-workflow to help me run THETA topic modeling on a policy-text dataset.
```

If your agent runtime supports named skill invocation, this can also be called as `$theta-workflow`.

To install from the standalone repository instead of this THETA checkout:

```bash
git clone https://github.com/CodeSoul-co/theta-skill.git
cp -R theta-skill/theta-workflow "<AGENT_SKILLS_DIR>/theta-workflow"
```

Generic folder-based install:

```bash
mkdir -p "<AGENT_SKILLS_DIR>"
cp -R skills/theta-workflow "<AGENT_SKILLS_DIR>/theta-workflow"
```

Optional Codex-compatible local install:

```bash
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
cp -R skills/theta-workflow "${CODEX_HOME:-$HOME/.codex}/skills/theta-workflow"
```

Restart or reload your agent runtime after installation if it caches available skills.

The standalone repository includes sync scripts for maintainers. Use `scripts/sync_from_theta.sh` there to pull updates from this THETA copy, or `scripts/sync_to_theta.sh` to push standalone edits back into `skills/theta-workflow/`.

When the skill is active, it first checks whether the current workspace is a usable THETA repository. If no repository is found, the first confirmed step is cloning `https://github.com/CodeSoul-co/THETA.git`; it will not configure the environment, inspect data, or generate training commands before the repository exists.

Run the read-only preflight helper before changing local files or launching training:

```bash
python skills/theta-workflow/scripts/inspect_theta_env.py \
  --dataset path/to/data.csv \
  --text-column text \
  --mode zero_shot
```

The skill enforces explicit confirmation before writing files, installing dependencies, editing `.env`, calling cloud embedding APIs, downloading models, running training, overwriting results, deleting files, or changing git state. Cloud embedding is only allowed for `zero_shot`; supervised or unsupervised finetune-capable workflows must use local models.

---

## Data Format Requirements

THETA uses a **strict column naming convention** for data files.

### Column Naming Convention

| Purpose | Column Name | Required For | Format |
|---------|-------------|--------------|--------|
| **Text** | `text` | All models | String |
| **Timestamp** | `timestamp` | DTM | `2026`, `2026-10-17`, or `2026-10-17 14:30:00` |
| **Covariates** | `cov_*` | STM | Prefix with `cov_` (e.g., `cov_province`) |
| **Label** | `label` | Supervised | String or integer |

### Model-Specific Requirements

| Model | Required Columns |
|-------|------------------|
| **DTM** | `text`, `timestamp` |
| **STM** | `text`, `cov_*` |

>  **DTM Note**: Only supports **year-level** granularity. Dates like `2026-10-17` are converted to `2026`.
> 
>  **STM Note**: All covariate columns **must** use the `cov_` prefix.

 **See [example/DATA_FORMAT_TEMPLATE.md](example/DATA_FORMAT_TEMPLATE.md) for CSV templates.**

---

## Configuration System: From Hardware to Experiments

THETA uses a **layered configuration** architecture for flexible control from hardware paths to experiment parameters.

### Core Configuration File `.env` (Hardware Physical Paths)

Create a `.env` file (refer to `.env.example`) with **required** settings:

```bash
# Default for zero_shot: cloud embedding service
EMBEDDING_PROVIDER=cloud
EMBEDDING_CLOUD_PROVIDER=openai
OPENAI_API_KEY=your-openai-api-key
# Optional: EMBEDDING_MODEL=text-embedding-3-small

# Required for supervised/unsupervised modes because fine-tuning needs local weights
# EMBEDDING_PROVIDER=local
# QWEN_MODEL_0_6B=./models/qwen3_embedding_0.6B

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

#### Common Parameters

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

#### Model-Specific Additional Parameters

**THETA**

Additional parameters beyond common defaults:

| Parameter      | Type  | Default     | Range                               | Description             |
| -------------- | ----- | ----------- | ----------------------------------- | ----------------------- |
| `--model_size` | str   | `0.6B`      | `0.6B` / `4B` / `8B`                | Qwen model size         |
| `--embedding-provider` | str | `cloud` for `zero_shot` | `cloud` / `local` / provider preset | Embedding provider; supervised/unsupervised require local Qwen |
| `--embedding-cloud-provider` | str | `openai` | `openai` / `dashscope` / `siliconflow` / `zhipu` / `volcengine` / `openai_compatible` | Cloud embedding preset |
| `--mode`       | str   | `zero_shot` | `zero_shot` / `supervised` / `unsupervised` | Embedding mode          |
| `--kl_start`   | float | 0.0         | 0–1                                 | KL annealing start weight |
| `--kl_end`     | float | 1.0         | 0–1                                 | KL annealing end weight |
| `--kl_warmup`  | int   | 50          | 0–epochs                            | KL warmup epochs        |
| `--language`   | str   | `zh`        | `en` / `zh`                         | Visualization language  |

**LDA**

Additional parameters beyond common defaults:

| Parameter    | Type  | Default  | Range  | Description                    |
| ------------ | ----- | -------- | ------ | ------------------------------ |
| `--max_iter` | int   | 100      | 10–500 | Maximum EM iterations          |
| `--alpha`    | float | 1/K (auto) | >0     | Document-topic Dirichlet prior |

**HDP**

Additional parameters beyond common defaults:

| Parameter      | Type  | Default | Range  | Description                            |
| -------------- | ----- | ------- | ------ | -------------------------------------- |
| `--max_topics` | int   | 150     | 50–300 | Upper bound on number of topics (replaces `--num_topics`) |
| `--alpha`      | float | 1.0     | >0     | Document-level concentration parameter |

**STM**

Additional parameters beyond common defaults:

| Parameter    | Type | Default | Range  | Description           |
| ------------ | ---- | ------- | ------ | --------------------- |
| `--max_iter` | int  | 100     | 10–500 | Maximum EM iterations |

**BTM**

Additional parameters beyond common defaults:

| Parameter  | Type  | Default | Range  | Description                            |
| ---------- | ----- | ------- | ------ | -------------------------------------- |
| `--n_iter` | int   | 100     | 10–500 | Gibbs sampling iterations (replaces `--epochs`) |
| `--alpha`  | float | 1.0     | >0     | Topic distribution Dirichlet prior     |
| `--beta`   | float | 0.01    | >0     | Word distribution Dirichlet prior      |

**ETM**

Additional parameters beyond common defaults:

| Parameter         | Type | Default | Range   | Description                    |
| ----------------- | ---- | ------- | ------- | ------------------------------ |
| `--embedding_dim` | int  | 300     | 50–1024 | Word embedding dimension (Word2Vec) |

**CTM**

Additional parameters beyond common defaults:

| Parameter          | Type | Default    | Range                   | Description                                         |
| ------------------ | ---- | ---------- | ----------------------- | --------------------------------------------------- |
| `--inference_type` | str  | `zeroshot` | `zeroshot` / `combined` | Inference mode: SBERT only or SBERT + BOW           |
| `--hidden_dim`     | int  | 100        | 32–1024                 | Overrides common default (512 → 100)                |

**DTM**

Additional parameters beyond common defaults:

| Parameter         | Type | Default | Range   | Description          |
| ----------------- | ---- | ------- | ------- | -------------------- |
| `--embedding_dim` | int  | 300     | 50–1024 | Word embedding dimension |

> **Note**: DTM does not use `--num_layers`, `--dropout`, or `--patience`.  
> **Data requirement**: DTM requires a `timestamp` column. Run `python prepare_data.py --dataset your_data --model dtm` before training.

**NVDM / GSM / ProdLDA**

No additional parameters — all settings covered by common defaults.  
> **Note**: `--hidden_dim` defaults to 256 for these models.

**BERTopic**

Additional parameters beyond common defaults:

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
  journal={TBD},
  year={2026},
  doi={TBD}
}
```

---

## Contact

For questions, please contact:
- duanzhenke@code-soul.com

- Agent through CLI Application is coming soon.

---

## License

Apache-2.0

[Publication figures and high-resolution export](docs/publication-visualization.md)

[Bilingual figure atlas examples](docs/examples/figure-atlas/README.md)

## Agent, CLI and local workbench

The open-source edition now includes the conversational Agent, free statistical analysis, a CLI, and the existing web workbench with direct access (no account or simulated user). See the [startup guide](docs/opensource-agent.md), [Agent documentation](agent/README.md), and [bundled Data Viz skill](agent/docs/data-viz-skill.md).
