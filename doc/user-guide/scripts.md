# Shell Scripts Reference

All scripts are **non-interactive** (pure command-line parameters), suitable for DLC/batch environments. No stdin input required.

## Script Overview

| Script | Description |
|--------|-------------|
| `01_setup.sh` | Install dependencies and download data from HuggingFace |
| `02_clean_data.sh` | Clean raw text data (tokenization, stopword removal, lemmatization) |
| `02_generate_embeddings.sh` | Generate Qwen embeddings (sub-script of 03, for failure recovery) |
| `03_prepare_data.sh` | One-stop data preparation: BOW + embeddings for all 12 models |
| `04_train_theta.sh` | Train THETA model (train + evaluate + visualize) |
| `05_train_baseline.sh` | Train 11 baseline models for comparison with THETA |
| `06_visualize.sh` | Generate visualizations for trained models |
| `07_evaluate.sh` | Standalone evaluation with 7 unified metrics |
| `08_compare_models.sh` | Cross-model metric comparison table |
| `09_download_from_hf.sh` | Download pre-trained data from HuggingFace |
| `10_quick_start_english.sh` | Quick start for English datasets |
| `11_quick_start_chinese.sh` | Quick start for Chinese datasets |
| `12_train_multi_gpu.sh` | Multi-GPU training with DistributedDataParallel |
| `13_test_agent.sh` | Test LLM Agent connection and functionality |
| `14_start_agent_api.sh` | Start the Agent API server (FastAPI) |

---

## A) Data Cleaning — `02_clean_data.sh`

Row-by-row text cleaning with user-specified column selection. Two modes:
- **CSV mode**: User specifies `--text_column` (cleaned) and `--label_columns` (preserved as-is)
- **Directory mode**: Convert docx/txt files into a single cleaned CSV

**Supported languages**: `english`, `chinese`, `german`, `spanish`

```bash
# 1. Preview columns (recommended first step for CSV)
bash scripts/02_clean_data.sh \
    --input data/FCPB/complaints_text_only.csv --preview

# 2. Clean text column only
bash scripts/02_clean_data.sh \
    --input data/FCPB/complaints_text_only.csv \
    --language english \
    --text_column 'Consumer complaint narrative'

# 3. Clean text + keep label column
bash scripts/02_clean_data.sh \
    --input data/hatespeech/hatespeech_text_only.csv \
    --language english \
    --text_column cleaned_content --label_columns Label

# 4. Keep ALL columns, only clean the text column
bash scripts/02_clean_data.sh \
    --input raw.csv --language english \
    --text_column text --keep_all

# 5. Directory mode (docx/txt → CSV)
bash scripts/02_clean_data.sh \
    --input data/edu_data/ --language chinese
```

| Parameter | Required | Description | Default |
|-----------|----------|-------------|---------|
| `--input` | ✓ | Input CSV file or directory (docx/txt) | - |
| `--language` | ✓ (not for preview) | Data language: english, chinese, german, spanish | - |
| `--text_column` | ✓ (CSV mode) | Name of the text column to clean | - |
| `--label_columns` | | Comma-separated label/metadata columns to keep as-is | - |
| `--keep_all` | | Keep ALL original columns (only text column is cleaned) | false |
| `--preview` | | Show CSV columns and sample rows, then exit | false |
| `--output` | | Output CSV path | auto-generated |
| `--min_words` | | Min words per document after cleaning | 3 |

**Output**: `data/{dataset}/{dataset}_cleaned.csv`

---

## B) Data Preparation — `03_prepare_data.sh`

One-stop data preparation for all 12 models. Generates BOW matrix and model-specific embeddings.

**Data requirements by model**:

| Model | Type | Data Needed |
|-------|------|-------------|
| lda, hdp, btm | Traditional | BOW only |
| stm | Traditional | BOW + covariates (document metadata) |
| nvdm, gsm, prodlda | Neural | BOW only |
| etm | Neural | BOW + Word2Vec |
| ctm | Neural | BOW + SBERT |
| dtm | Neural | BOW + SBERT + time slices |
| bertopic | Neural | SBERT + raw text |
| theta | THETA | BOW + Qwen embeddings |

> **Note**: Models 1-7 (BOW-only) share the same data experiment. Prepare once, train all.

```bash
# ---- Baseline models ----

# BOW-only models (lda, hdp, btm, nvdm, gsm, prodlda share this)
bash scripts/03_prepare_data.sh \
    --dataset edu_data --model lda --vocab_size 3500 --language chinese

# CTM (BOW + SBERT embeddings)
bash scripts/03_prepare_data.sh \
    --dataset edu_data --model ctm --vocab_size 3500 --language chinese

# ETM (BOW + Word2Vec embeddings)
bash scripts/03_prepare_data.sh \
    --dataset edu_data --model etm --vocab_size 3500 --language chinese

# DTM (BOW + SBERT + time slices, requires time column)
bash scripts/03_prepare_data.sh \
    --dataset edu_data --model dtm --vocab_size 3500 --language chinese --time_column year

# BERTopic (SBERT + raw text)
bash scripts/03_prepare_data.sh \
    --dataset edu_data --model bertopic --vocab_size 3500 --language chinese

# ---- THETA model ----

# Zero-shot (fastest, no training needed)
bash scripts/03_prepare_data.sh \
    --dataset edu_data --model theta --model_size 0.6B --mode zero_shot \
    --vocab_size 3500 --language chinese

# Unsupervised (LoRA fine-tuned Qwen embeddings)
bash scripts/03_prepare_data.sh \
    --dataset edu_data --model theta --model_size 0.6B --mode unsupervised \
    --vocab_size 3500 --language chinese

# Supervised (requires label column)
bash scripts/03_prepare_data.sh \
    --dataset edu_data --model theta --model_size 0.6B --mode supervised \
    --vocab_size 3500 --language chinese

# ---- Advanced options ----

# BOW only (skip embedding generation)
bash scripts/03_prepare_data.sh --dataset mydata --model theta --bow-only --vocab_size 5000

# Check if data files already exist
bash scripts/03_prepare_data.sh --dataset mydata --model theta --check-only

# Custom vocabulary size and max sequence length
bash scripts/03_prepare_data.sh --dataset mydata \
    --model theta --model_size 0.6B --mode zero_shot \
    --vocab_size 10000 --batch_size 64 --gpu 0
```

| Parameter | Required | Description | Default |
|-----------|----------|-------------|---------|
| `--dataset` | ✓ | Dataset name | - |
| `--model` | ✓ | Target model: lda, hdp, stm (requires covariates), btm, nvdm, gsm, prodlda, ctm, etm, dtm, bertopic, theta | - |
| `--model_size` | | Qwen model size (theta only): 0.6B, 4B, 8B | 0.6B |
| `--mode` | | Embedding mode (theta only): zero_shot, unsupervised, supervised | zero_shot |
| `--vocab_size` | | Vocabulary size | 5000 |
| `--batch_size` | | Embedding generation batch size | 32 |
| `--gpu` | | GPU device ID | 0 |
| `--language` | | Data language: english, chinese (controls tokenization) | english |
| `--bow-only` | | Only generate BOW, skip embeddings | false |
| `--check-only` | | Only check if files exist | false |
| `--time_column` | | Time column name (DTM only) | year |
| `--label_column` | | Label column (theta supervised only) | - |
| `--emb_epochs` | | Embedding fine-tuning epochs (theta only) | 10 |
| `--emb_batch_size` | | Embedding fine-tuning batch size (theta only) | 8 |
| `--exp_name` | | Experiment name tag | auto-generated |

**Embedding recovery** — If embedding generation fails (e.g., OOM), re-run only the embedding step:

```bash
bash scripts/02_generate_embeddings.sh \
    --dataset edu_data --mode zero_shot --model_size 0.6B \
    --batch_size 4 --exp_dir result/0.6B/edu_data/data/exp_xxx
```

---

## C) THETA Model Training — `04_train_theta.sh`

Train THETA model with integrated training + evaluation + visualization.

```bash
# ---- Basic usage ----

# Zero-shot mode (simplest command)
bash scripts/04_train_theta.sh \
    --dataset edu_data --model_size 0.6B --mode zero_shot --num_topics 20

# Unsupervised mode
bash scripts/04_train_theta.sh \
    --dataset edu_data --model_size 0.6B --mode unsupervised --num_topics 20

# Supervised mode (requires label column)
bash scripts/04_train_theta.sh \
    --dataset hatespeech --model_size 0.6B --mode supervised --num_topics 20

# Larger model for better quality
bash scripts/04_train_theta.sh \
    --dataset hatespeech --model_size 4B --mode zero_shot --num_topics 20

# ---- Full parameters ----

bash scripts/04_train_theta.sh \
    --dataset edu_data --model_size 0.6B --mode zero_shot \
    --num_topics 20 --epochs 100 --batch_size 64 \
    --hidden_dim 512 --learning_rate 0.002 \
    --kl_start 0.0 --kl_end 1.0 --kl_warmup 50 \
    --patience 10 --gpu 0 --language zh

# Custom KL annealing
bash scripts/04_train_theta.sh \
    --dataset hatespeech --model_size 0.6B --mode zero_shot \
    --num_topics 20 --epochs 200 \
    --kl_start 0.1 --kl_end 0.8 --kl_warmup 40

# ---- Specify data experiment ----

bash scripts/04_train_theta.sh \
    --dataset edu_data --model_size 0.6B --mode zero_shot \
    --data_exp exp_20260208_151906_vocab3500_theta_0.6B_zero_shot \
    --num_topics 20 --epochs 50 --language zh

# ---- Skip options ----

# Skip visualization (train + evaluate only, faster)
bash scripts/04_train_theta.sh \
    --dataset edu_data --model_size 0.6B --mode zero_shot \
    --num_topics 20 --skip-viz

# Skip training (evaluate + visualize existing model)
bash scripts/04_train_theta.sh \
    --dataset edu_data --model_size 0.6B --mode zero_shot \
    --skip-train --language zh
```

| Parameter | Required | Description | Default |
|-----------|----------|-------------|---------|
| `--dataset` | ✓ | Dataset name | - |
| `--model_size` | | Qwen model size: 0.6B, 4B, 8B | 0.6B |
| `--mode` | | Embedding mode: zero_shot, unsupervised, supervised | zero_shot |
| `--num_topics` | | Number of topics K | 20 |
| `--epochs` | | Training epochs | 100 |
| `--batch_size` | | Training batch size | 64 |
| `--hidden_dim` | | Encoder hidden dimension | 512 |
| `--learning_rate` | | Learning rate | 0.002 |
| `--kl_start` | | KL annealing start weight | 0.0 |
| `--kl_end` | | KL annealing end weight | 1.0 |
| `--kl_warmup` | | KL warmup epochs | 50 |
| `--patience` | | Early stopping patience | 10 |
| `--gpu` | | GPU device ID | 0 |
| `--language` | | Visualization language: en, zh | en |
| `--skip-train` | | Skip training, only evaluate | false |
| `--skip-viz` | | Skip visualization | false |
| `--data_exp` | | Data experiment ID | auto latest |
| `--exp_name` | | Experiment name tag | auto-generated |

---

## D) Baseline Model Training — `05_train_baseline.sh`

Train 11 baseline topic models for comparison with THETA.

### Supported Models

| Model | Type | Description | Model-Specific Parameters |
|-------|------|-------------|---------------------------|
| **lda** | Traditional | Latent Dirichlet Allocation | `--max_iter` |
| **hdp** | Traditional | Hierarchical Dirichlet Process (auto topic count) | `--max_topics`, `--alpha` |
| **stm** | Traditional | Structural Topic Model (**requires covariates**) | `--max_iter` |
| **btm** | Traditional | Biterm Topic Model (best for short texts) | `--n_iter`, `--alpha`, `--beta` |
| **nvdm** | Neural | Neural Variational Document Model | `--epochs`, `--dropout` |
| **gsm** | Neural | Gaussian Softmax Model | `--epochs`, `--dropout` |
| **prodlda** | Neural | Product of Experts LDA | `--epochs`, `--dropout` |
| **ctm** | Neural | Contextualized Topic Model (requires SBERT) | `--epochs`, `--inference_type` |
| **etm** | Neural | Embedded Topic Model (requires Word2Vec) | `--epochs` |
| **dtm** | Neural | Dynamic Topic Model (requires timestamps) | `--epochs` |
| **bertopic** | Neural | BERT-based Topic Model (auto topic count) | - |

### Complete Per-Model Examples

```bash
# ============================================================
# 1. LDA — Latent Dirichlet Allocation
#    Type: Traditional | Data: BOW only
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
# 2. HDP — Hierarchical Dirichlet Process
#    Note: Auto-determines topic count, --num_topics is IGNORED
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
# 3. STM — Structural Topic Model
#    REQUIRES covariates — auto-skipped if dataset has no metadata
# ============================================================
#
# To use STM:
#   1. Ensure your cleaned CSV has metadata columns
#   2. Register covariates in ETM/config.py → DATASET_CONFIGS
#   3. Prepare data (same as other BOW models)
#   4. Train STM

bash scripts/05_train_baseline.sh \
    --dataset my_dataset_with_covariates --models stm --num_topics 20

bash scripts/05_train_baseline.sh \
    --dataset my_dataset_with_covariates --models stm \
    --num_topics 20 --max_iter 200 \
    --gpu 0 --language en --with-viz \
    --data_exp exp_20260208_153424_vocab3500_lda \
    --exp_name stm_full

# ============================================================
# 4. BTM — Biterm Topic Model
#    Best suited for short texts (tweets, comments)
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
# 5. NVDM / 6. GSM / 7. ProdLDA — BOW-only neural models
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

# (Replace nvdm with gsm or prodlda for those models)

# ============================================================
# 8. CTM — Contextualized Topic Model
#    Requires SBERT data_exp (prepared with --model ctm)
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

# Combined inference (uses both BOW and SBERT)
bash scripts/05_train_baseline.sh \
    --dataset edu_data --models ctm \
    --num_topics 20 --epochs 100 --inference_type combined \
    --gpu 0 --language zh --with-viz

# ============================================================
# 9. ETM — Embedded Topic Model (BOW + Word2Vec)
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
# 10. DTM — Dynamic Topic Model (requires timestamps)
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
# 11. BERTopic — Auto-determines topic count
# ============================================================

bash scripts/05_train_baseline.sh \
    --dataset edu_data --models bertopic

bash scripts/05_train_baseline.sh \
    --dataset edu_data --models bertopic \
    --gpu 0 --language zh --with-viz \
    --data_exp exp_20260208_154645_vocab3500_ctm \
    --exp_name bertopic_full

# ============================================================
# Batch training (multiple models at once)
# ============================================================

# Train all BOW-only models (share the same data_exp)
bash scripts/05_train_baseline.sh \
    --dataset edu_data \
    --models lda,hdp,btm,nvdm,gsm,prodlda \
    --num_topics 20 --epochs 100 \
    --data_exp exp_20260208_153424_vocab3500_lda

# Train CTM + BERTopic (share SBERT data_exp)
bash scripts/05_train_baseline.sh \
    --dataset edu_data --models ctm,bertopic \
    --num_topics 20 --epochs 100 \
    --data_exp exp_20260208_154645_vocab3500_ctm

# Skip training, only evaluate and visualize existing model
bash scripts/05_train_baseline.sh \
    --dataset edu_data --models lda --num_topics 20 --skip-train

# Enable visualization (disabled by default)
bash scripts/05_train_baseline.sh \
    --dataset edu_data --models lda --num_topics 20 \
    --with-viz --language zh
```

> **Important notes**:
> - BTM uses Gibbs sampling and is very slow on long documents (samples max 50 words/doc). Best for short texts.
> - HDP and BERTopic auto-determine topic count; `--num_topics` is ignored for these models.
> - STM requires document-level covariates. If your dataset has no `covariate_columns` in `DATASET_CONFIGS`, STM will be automatically skipped.
> - DTM requires a data experiment containing `time_slices.json` (prepared with `--model dtm`).
> - CTM and BERTopic require a data experiment containing SBERT embeddings.

### Parameter Reference

**Common parameters**:

| Parameter | Required | Description | Default |
|-----------|----------|-------------|---------|
| `--dataset` | ✓ | Dataset name | - |
| `--models` | ✓ | Model list (comma-separated) | - |
| `--num_topics` | | Number of topics (ignored for hdp/bertopic) | 20 |
| `--vocab_size` | | Vocabulary size | 5000 |
| `--epochs` | | Training epochs (neural models) | 100 |
| `--batch_size` | | Batch size | 64 |
| `--hidden_dim` | | Hidden layer dimension | 512 |
| `--learning_rate` | | Learning rate | 0.002 |
| `--gpu` | | GPU device ID | 0 |
| `--language` | | Visualization language: en, zh | en |
| `--skip-train` | | Skip training | false |
| `--with-viz` | | Enable visualization | false |
| `--data_exp` | | Data experiment ID | auto latest |
| `--exp_name` | | Experiment name tag | auto-generated |

**Model-specific parameters**:

| Parameter | Applicable Models | Description | Default |
|-----------|-------------------|-------------|---------|
| `--max_iter` | lda, stm | Max iterations (EM algorithm) | 100 |
| `--max_topics` | hdp | Max topic count | 150 |
| `--n_iter` | btm | Gibbs sampling iterations | 100 |
| `--alpha` | hdp, btm | Alpha prior | 1.0 |
| `--beta` | btm | Beta prior | 0.01 |
| `--inference_type` | ctm | Inference type: zeroshot, combined | zeroshot |
| `--dropout` | Neural models (nvdm, gsm, prodlda, ctm, etm, dtm) | Dropout rate | 0.2 |

---

## E) Visualization — `06_visualize.sh`

Generate visualizations for trained models without re-training.

```bash
# THETA model visualization
bash scripts/06_visualize.sh \
    --dataset edu_data --model_size 0.6B --mode zero_shot --language zh

# English charts + high DPI (for papers)
bash scripts/06_visualize.sh \
    --dataset edu_data --model_size 0.6B --mode zero_shot --language en --dpi 600

# Baseline model visualization
bash scripts/06_visualize.sh \
    --baseline --dataset edu_data --model lda --num_topics 20 --language zh

# HDP (auto topic count, use actual K from training)
bash scripts/06_visualize.sh \
    --baseline --dataset edu_data --model hdp --num_topics 150 --language zh

# DTM (includes topic evolution charts)
bash scripts/06_visualize.sh \
    --baseline --dataset edu_data --model dtm --num_topics 20 --language zh

# Specify a model experiment explicitly
bash scripts/06_visualize.sh \
    --baseline --dataset edu_data --model ctm --model_exp exp_20260208_xxx --language zh
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--dataset` | Dataset name (required) | — |
| `--baseline` | Baseline model mode | false |
| `--model` | Baseline model name | — |
| `--model_exp` | Model experiment ID (auto-selects latest if not specified) | auto latest |
| `--model_size` | THETA model size | 0.6B |
| `--mode` | THETA mode | zero_shot |
| `--language` | Visualization language: en, zh | en |
| `--dpi` | Image DPI | 300 |

**Generated charts** (20+ types):

| Chart | Description | Filename |
|-------|-------------|----------|
| Topic Table | Top words per topic | topic_table.png |
| Topic Network | Inter-topic similarity network | topic_network.png |
| Document Clusters | UMAP document distribution | doc_topic_umap.png |
| Cluster Heatmap | Topic-document heatmap | cluster_heatmap.png |
| Topic Proportion | Document proportion per topic | topic_proportion.png |
| Training Loss | Loss curve | training_loss.png |
| Evaluation Metrics | 7-metric radar chart | metrics.png |
| Topic Coherence | Per-topic NPMI | topic_coherence.png |
| Topic Exclusivity | Per-topic exclusivity | topic_exclusivity.png |
| Word Clouds | All topic word clouds | topic_wordclouds.png |
| Topic Similarity | Inter-topic cosine similarity | topic_similarity.png |
| pyLDAvis | Interactive topic explorer | pyldavis_interactive.html |
| Per-topic Words | Per-topic word weights | topics/topic_N/word_importance.png |

---

## F) Evaluation — `07_evaluate.sh`

Standalone evaluation with 7 unified metrics.

```bash
# Evaluate baseline models
bash scripts/07_evaluate.sh --dataset edu_data --model lda --num_topics 20
bash scripts/07_evaluate.sh --dataset edu_data --model hdp --num_topics 150
bash scripts/07_evaluate.sh --dataset edu_data --model ctm --num_topics 20
bash scripts/07_evaluate.sh --dataset edu_data --model etm --num_topics 20
bash scripts/07_evaluate.sh --dataset edu_data --model dtm --num_topics 20
bash scripts/07_evaluate.sh --dataset edu_data --model bertopic --num_topics 20

# With custom vocab size
bash scripts/07_evaluate.sh --dataset edu_data --model lda --num_topics 20 --vocab_size 3500

# Evaluate THETA models
bash scripts/07_evaluate.sh --dataset edu_data --model theta --model_size 0.6B --mode zero_shot
bash scripts/07_evaluate.sh --dataset edu_data --model theta --model_size 0.6B --mode unsupervised
bash scripts/07_evaluate.sh --dataset edu_data --model theta --model_size 4B --mode supervised
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--dataset` | Dataset name (required) | — |
| `--model` | Model name (required): lda, hdp, stm, btm, nvdm, gsm, prodlda, ctm, etm, dtm, bertopic, theta | — |
| `--num_topics` | Number of topics | 20 |
| `--vocab_size` | Vocabulary size | 5000 |
| `--model_size` | THETA model size: 0.6B, 4B, 8B | 0.6B |
| `--mode` | THETA mode: zero_shot, unsupervised, supervised | zero_shot |

**Evaluation Metrics (7 metrics)**:

| Metric | Full Name | Direction | Description |
|--------|-----------|-----------|-------------|
| **TD** | Topic Diversity | ↑ Higher is better | Proportion of unique words across topics |
| **iRBO** | Inverse Rank-Biased Overlap | ↑ Higher is better | Rank-based topic diversity |
| **NPMI** | Normalized PMI | ↑ Higher is better | Normalized pointwise mutual information coherence |
| **C_V** | C_V Coherence | ↑ Higher is better | Sliding-window based coherence |
| **UMass** | UMass Coherence | → Closer to 0 is better | Document co-occurrence based coherence |
| **Exclusivity** | Topic Exclusivity | ↑ Higher is better | How exclusive words are to their topics |
| **PPL** | Perplexity | ↓ Lower is better | Model fit (lower = better generalization) |

---

## G) Model Comparison — `08_compare_models.sh`

Cross-model metric comparison table.

```bash
# Compare all baseline models
bash scripts/08_compare_models.sh \
    --dataset edu_data \
    --models lda,hdp,btm,nvdm,gsm,prodlda,ctm,etm,dtm,bertopic \
    --num_topics 20

# Compare traditional models only
bash scripts/08_compare_models.sh \
    --dataset edu_data --models lda,hdp,btm --num_topics 20

# Compare neural models only
bash scripts/08_compare_models.sh \
    --dataset edu_data --models nvdm,gsm,prodlda,ctm,etm,dtm --num_topics 20

# Export to CSV
bash scripts/08_compare_models.sh \
    --dataset edu_data --models lda,hdp,nvdm,gsm,prodlda,ctm,etm,dtm \
    --num_topics 20 --output comparison.csv
```

**Example output**:
```
================================================================================
Model Comparison: edu_data (K=20)
================================================================================

Model              TD     iRBO     NPMI      C_V    UMass  Exclusivity        PPL
--------------------------------------------------------------------------------
lda            0.8500   0.7200   0.0512   0.4231  -2.1234       0.6543     123.45
prodlda        0.9200   0.8100   0.0634   0.4567  -1.8765       0.7234      98.76
ctm            0.8800   0.7800   0.0589   0.4412  -1.9876       0.6987     105.32
--------------------------------------------------------------------------------

Best Models:
  - Best TD (Topic Diversity): prodlda (0.9200)
  - Best NPMI (Coherence):     prodlda (0.0634)
  - Best PPL (Perplexity):     prodlda (98.76)
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--dataset` | Dataset name (required) | — |
| `--models` | Comma-separated model list (required) | — |
| `--num_topics` | Number of topics | 20 |
| `--output` | Output CSV file path | terminal only |

---

## H) Multi-GPU Training — `12_train_multi_gpu.sh`

THETA supports multi-GPU training using PyTorch DistributedDataParallel (DDP).

```bash
# Train with 2 GPUs
bash scripts/12_train_multi_gpu.sh --dataset hatespeech --num_gpus 2 --num_topics 20

# Full parameters
bash scripts/12_train_multi_gpu.sh --dataset hatespeech \
    --num_gpus 4 --model_size 0.6B --mode zero_shot \
    --num_topics 25 --epochs 150 --batch_size 64 \
    --hidden_dim 768 --learning_rate 0.001

# Custom master port (for multiple concurrent jobs)
bash scripts/12_train_multi_gpu.sh --dataset socialTwitter \
    --num_gpus 2 --master_port 29501

# Or use torchrun directly
torchrun --nproc_per_node=2 --master_port=29500 \
    ETM/main.py train \
    --dataset hatespeech --mode zero_shot --num_topics 20 --epochs 100
```

---

## I) Agent API — `14_start_agent_api.sh`

Start the AI Agent API server for interactive analysis.

```bash
# Start the Agent API (default port 8000)
bash scripts/14_start_agent_api.sh --port 8000

# Test agent connection
bash scripts/13_test_agent.sh
```

API endpoints: `POST /chat`, `POST /api/chat/v2`, `POST /api/interpret/metrics`, `POST /api/interpret/topics`, `POST /api/vision/analyze`. See `agent/docs/API_REFERENCE.md` for full details.

---

## J) Batch Processing Examples

```bash
# Train THETA across multiple datasets
for dataset in hatespeech mental_health socialTwitter; do
    bash scripts/04_train_theta.sh --dataset $dataset \
        --model_size 0.6B --mode zero_shot --num_topics 20
done

# Compare different topic counts
for k in 10 15 20 25 30; do
    bash scripts/04_train_theta.sh --dataset hatespeech \
        --model_size 0.6B --mode zero_shot --num_topics $k
done

# Generate visualizations for all trained baseline models
for model in lda etm ctm prodlda; do
    bash scripts/06_visualize.sh --baseline --dataset hatespeech \
        --model $model --num_topics 20 --language en
done
```

---

## K) End-to-End Example: edu_data

Complete workflow using `edu_data` (823 Chinese education policy documents).

```bash
# 1. Setup
bash scripts/01_setup.sh

# 2. Data cleaning
bash scripts/02_clean_data.sh --input ./data/edu_data/ --language chinese

# 3. Data preparation — baselines
bash scripts/03_prepare_data.sh \
    --dataset edu_data --model lda --vocab_size 3500 --language chinese
bash scripts/03_prepare_data.sh \
    --dataset edu_data --model ctm --vocab_size 3500 --language chinese
bash scripts/03_prepare_data.sh \
    --dataset edu_data --model dtm --vocab_size 3500 --language chinese --time_column year

# 4. Data preparation — THETA
bash scripts/03_prepare_data.sh \
    --dataset edu_data --model theta --model_size 0.6B --mode zero_shot \
    --vocab_size 3500 --language chinese

# 5. Train baselines
bash scripts/05_train_baseline.sh \
    --dataset edu_data --models lda,hdp,btm,nvdm,gsm,prodlda \
    --num_topics 20 --epochs 100

# 6. Train THETA
bash scripts/04_train_theta.sh \
    --dataset edu_data --model_size 0.6B --mode zero_shot \
    --num_topics 20 --epochs 100 --language zh

# 7. Compare models
bash scripts/08_compare_models.sh \
    --dataset edu_data \
    --models lda,hdp,btm,nvdm,gsm,prodlda,ctm,etm \
    --num_topics 20
```
