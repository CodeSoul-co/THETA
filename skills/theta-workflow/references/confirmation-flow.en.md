# THETA Skill Workflow: English Confirmation Version

## Step 0: Decide Whether to Use the THETA Skill

Use this skill when the user mentions:

```text
THETA
topic modeling
topic model
LDA / STM / DTM / CTM / BERTopic
policy text topic analysis
social media text topic analysis
topic evolution
automatic modeling
text clustering
topic discovery
```

This is only a routing decision. It does not require user confirmation.

## Step 1: Check Whether the Repository Exists

When the THETA Skill is active, the first step is to check whether the current workspace or user-provided location already contains a usable THETA repository.

Read-only checks do not require confirmation:

- Whether the THETA repository exists.
- Whether `README.md` exists.
- Whether `scripts/` exists.
- Whether `src/models/run_pipeline.py` exists.
- Whether `.env.example` exists.

If the repository does not exist, the first action is to clone the repository. Do not configure the environment, inspect data, or generate training commands first.

Clone command:

```bash
git clone https://github.com/CodeSoul-co/THETA.git
```

This creates a new local folder, so confirmation is required:

```text
I did not find a local THETA repository.

I am preparing to perform the following action:

Action:
Clone the THETA repository locally.

Command:
git clone https://github.com/CodeSoul-co/THETA.git

Impact:
This will create a new THETA folder. It will not modify existing files.

Please confirm whether to continue.
```

Only after the repository exists should the agent continue to environment checks, `.env` configuration, dataset checks, and modeling workflow.

## Step 2: Understand the User's Task

Ask only the necessary questions first:

1. Where is the dataset file?
2. What is the text column called? For example: `text`, `content`, `body`, `abstract`.
3. Is the data Chinese, English, or mixed?
4. Is there a time column? For example: `year`, `date`, `timestamp`.
5. Are there covariates? For example: `province`, `policy_type`, `source`, `department`.
6. What is the goal: quick smoke test, best topic quality, best interpretability, topic evolution, or paper-grade model comparison?

This stage is only for understanding the task. No extra confirmation is needed.

Example:

```text
I need to confirm a few details first:

1. What is the dataset path?
2. What is the text column name?
3. Is the dataset Chinese, English, or mixed?
4. Does it include a time column?
5. Does it include covariates such as province, organization, category, or source?
6. Do you want a quick run, or a paper-grade model comparison?
```

## Step 3: Re-check Repository Structure

Before continuing, re-check the repository structure. Read-only checks do not require confirmation:

- Whether the THETA repository exists.
- Whether `README.md` exists.
- Whether `scripts/` exists.
- Whether `src/models/run_pipeline.py` exists.
- Whether `.env.example` exists.

If the repository is still missing, return to Step 1 and ask for confirmation before cloning:

```bash
git clone https://github.com/CodeSoul-co/THETA.git
```

Confirmation wording:

```text
I did not find a local THETA repository.

I am preparing to perform the following action:

Action:
Clone the THETA repository locally.

Command:
git clone https://github.com/CodeSoul-co/THETA.git

Impact:
This will create a new THETA folder. It will not modify existing files.

Please confirm whether to continue.
```

## Step 4: Check the Environment

Read-only environment checks do not require confirmation:

- Current Python version.
- Whether conda exists.
- Whether a `theta` environment exists.
- Whether requirements files exist.
- Whether `.env` exists.
- Whether model path variables are configured.

Creating environments or installing dependencies requires confirmation:

```bash
conda create -n theta python=3.10 -y
bash scripts/env_setup.sh
pip install -r requirements.txt
```

Confirmation wording:

```text
THETA needs a Python 3.10 environment.

I am preparing to perform the following action:

Action:
Create a conda environment and install THETA dependencies.

Command:
conda create -n theta python=3.10 -y
conda activate theta
bash scripts/env_setup.sh

Impact:
This will create a new conda environment named theta and install dependencies.

Please confirm whether to continue.
```

## Step 5: Check `.env` Configuration

Reading `.env.example` and identifying required keys does not require confirmation.

Creating or modifying `.env` requires confirmation:

```bash
cp .env.example .env
```

or writing:

```text
EMBEDDING_PROVIDER=cloud
OPENAI_API_KEY=...
QWEN_MODEL_0_6B=...
```

Never display, print, or save the user's API key in chat.

Confirmation wording:

```text
THETA needs a `.env` configuration file.

I am preparing to perform the following action:

Action:
Create `.env` from `.env.example`.

Command:
cp .env.example .env

Impact:
This will create a new `.env` file. If `.env` already exists, I will not overwrite it without warning you first.

Note:
If cloud embedding is used, you should place the API key in `.env` yourself. I will not display or store your secret in chat.

Please confirm whether to continue.
```

## Step 6: Choose Embedding Mode

### 5.1 Cloud Embedding

Cloud embedding is easier for normal zero-shot use, but it may incur API cost.

```text
EMBEDDING_PROVIDER=cloud
EMBEDDING_CLOUD_PROVIDER=openai
```

Because this may call a paid API, confirmation is required:

```text
You selected cloud embedding.

This may call an external API and may incur cost depending on your account.

I am preparing to use this configuration:

EMBEDDING_PROVIDER=cloud
EMBEDDING_CLOUD_PROVIDER=openai

Please confirm whether to continue.
```

### 5.2 Local Embedding

Local embedding is better for research workflows, but it requires local model files.

```text
QWEN_MODEL_0_6B=./models/qwen3_embedding_0.6B
SBERT_MODEL_PATH=./models/sbert
```

Checking paths does not require confirmation. Downloading models requires confirmation:

```text
The local model path does not exist.

I am preparing to download or configure local embedding models.

Action:
Download or configure Qwen / SBERT model paths.

Impact:
This may use significant disk space and take time to download.

Please confirm whether to continue.
```

### 5.3 Larger Local Models

4B / 8B embedding models need more GPU memory and disk space, so confirmation is required:

```text
You selected a larger local embedding model.

This may require more GPU memory and disk space.

I recommend first using the 0.6B model to validate the workflow, then deciding whether to switch to 4B or 8B.

Do you still want to continue with the larger model?
```

## Step 7: Inspect the Dataset

Read-only dataset inspection does not require confirmation:

- Row count.
- Column names.
- Whether the text column exists.
- Empty text count.
- Duplicate text count.
- Average text length.
- Whether a time column exists.
- Whether covariate columns exist.

Example:

```text
I inspected the dataset:

Total rows: 8420
Detected text column: content
Detected time column: year
No severe missing text issue found
Duplicate texts: 132
Average text length: 486 characters

This dataset is suitable for policy text topic evolution analysis.
```

## Step 8: Convert Data Format

THETA usually expects the text column to be `text`, the time column to be `timestamp`, and covariates to use the `cov_` prefix.

If the original columns are:

```text
content
year
province
policy_type
```

suggest converting them to:

```text
text
timestamp
cov_province
cov_policy_type
```

Any file write requires confirmation. Do not modify original data by default; create a derived file such as `policy_data.theta_ready.csv`.

Confirmation wording:

```text
Your dataset columns do not fully match THETA's expected format.

I recommend creating a converted copy:

Original file:
policy_data.csv

New file:
policy_data.theta_ready.csv

Conversion rules:
content -> text
year -> timestamp
province -> cov_province
policy_type -> cov_policy_type

The original file will not be modified.

Please confirm whether to create the converted file.
```

## Step 9: Recommend Models

Model recommendations do not require confirmation:

- Unknown topic count: HDP / BERTopic.
- Short text: BTM / BERTopic.
- Time column: DTM + THETA.
- Covariates: STM + THETA.
- High-quality semantic topics: THETA.
- Paper-grade comparison: LDA + CTM + BERTopic + ProdLDA + THETA.

Example:

```text
Based on your dataset, I recommend:

Main model:
THETA

Baseline models:
LDA, CTM, BERTopic

If topic evolution is important, add DTM.

Reason:
Your data is policy text and includes a year column. THETA is suitable as a high-quality semantic topic model, while DTM is useful for showing topic changes over time.
```

## Step 10: Recommend Parameters

Parameter suggestions do not require confirmation:

```text
I recommend the first run with:

num_topics = 20
vocab_size = 5000
epochs = 100
batch_size = 64
learning_rate = 0.002

If topics overlap heavily, reduce K to 15.
If topics are too broad, increase K to 30.
```

Scale guidance:

- Small dataset: K = 8 / 12 / 15.
- Medium dataset: K = 15 / 20 / 30.
- Large dataset: K = 30 / 50 / 80.

## Step 11: Generate Run Commands

Generating commands for review does not require confirmation. Executing them does.

Commands must be adjusted to the current repository CLI:

```bash
python src/models/run_pipeline.py \
  --dataset policy_data \
  --models theta,lda,ctm,bertopic \
  --num_topics 20 \
  --language en \
  --epochs 100 \
  --vocab_size 5000
```

## Step 12: Execute Model Training

Training may use GPU, take time, incur API cost, write result files, or run multiple models. Confirmation is required.

Confirmation wording:

```text
I am preparing to start model training.

Action:
Run THETA, LDA, CTM, and BERTopic.

Command:
python src/models/run_pipeline.py \
  --dataset policy_data \
  --models theta,lda,ctm,bertopic \
  --num_topics 20 \
  --language en \
  --epochs 100 \
  --vocab_size 5000

Output directory:
result/policy_data/

Resource impact:
This may use local compute resources. If cloud embedding is enabled, it may incur API cost.

Will this overwrite existing results:
No, by default it should create a new experiment directory.

Please confirm whether to execute.
```

Only proceed after explicit confirmation.

## Step 13: Read Results

Reading existing results does not require confirmation:

- `metrics.json`
- `topic_words.json`
- `topic_table.csv`
- `topic_network.png`
- `topic_similarity.png`
- `training_history.json`

Explain:

- Topic quality.
- Topic coherence.
- Topic diversity.
- Exclusivity.
- Topic overlap.
- Whether K should change.
- Whether the model should change.

Example:

```text
I read the results from this run.

Overall:
THETA has good topic diversity, but Topic 7 and Topic 12 overlap noticeably.
NPMI is moderate, so semantic coherence can still improve.
Exclusivity is low, which means some topic boundaries are not clear enough.

Suggested next run:
Reduce K from 20 to 15 and add a custom stopword list.
```

## Step 14: Tuning Suggestions

Suggestions do not require confirmation:

- Topics too fragmented: reduce `num_topics`.
- Topics too broad: increase `num_topics`.
- Repeated topic words: improve stopwords, reduce K, increase exclusivity.
- Low coherence: adjust `vocab_size`, improve preprocessing, try THETA/CTM.
- Unstable training: reduce `learning_rate`, increase patience.

## Step 15: Run Tuning Experiments

Suggesting experiments does not require confirmation. Running another experiment requires confirmation.

```text
I recommend two next experiments:

Experiment A:
K = 15

Experiment B:
K = 30

Current experiment:
K = 20

Goal:
Compare coherence, diversity, and exclusivity across topic counts.

This will start 2 additional model runs.

Please confirm whether to continue.
```

## Step 16: Generate Final Report

Summarizing in chat does not require confirmation.

Writing a report file requires confirmation, for example:

- `report.md`
- `experiment_summary.docx`
- `topic_modeling_result.tex`

Confirmation wording:

```text
I can organize the results into a report file:

Output file:
theta_experiment_report.md

Contents:
Model configuration
Dataset description
Evaluation metrics
Topic interpretation
Tuning suggestions
Paper-ready wording

Please confirm whether to generate this file.
```

## Step 17: Clean or Delete Files

Any deletion requires confirmation:

- Delete caches.
- Delete old results.
- Delete model files.
- Delete intermediate data.
- Reset workspace.

Confirmation wording:

```text
I found old experiment result directories:

result/policy_data/exp_001
result/policy_data/exp_002

If I delete them, they cannot be recovered from the current directory.

Please confirm whether to delete them.
```

## Confirmation Table

| Step | Operation | Confirmation Required |
| --- | --- | --- |
| Decide whether to use skill | Route THETA topic-modeling task | No |
| Check whether repository exists | Read directory structure | No |
| Clone repository when missing | `git clone` | Yes |
| Understand task | Ask about data, language, and goal | No |
| Inspect repository | Read directory structure | No |
| Clone repository | `git clone` | Yes |
| Inspect environment | Check Python / conda / `.env` | No |
| Create environment | `conda create` | Yes |
| Install dependencies | `pip install` / `env_setup.sh` | Yes |
| Read `.env.example` | Inspect config template | No |
| Modify `.env` | Write config | Yes |
| Check API key presence | Check variable presence only | No |
| Use cloud API | May incur cost | Yes |
| Inspect data | Read columns, rows, missing values | No |
| Convert data | Create CSV / JSONL | Yes |
| Recommend models | LDA / DTM / STM / THETA | No |
| Recommend parameters | K, epochs, batch size | No |
| Generate command | Show command to user | No |
| Execute model | Run training code | Yes |
| Read results | Read metrics / topics | No |
| Tuning advice | Suggest next changes | No |
| Execute tuning | Train again | Yes |
| Summarize in chat | Reply with summary | No |
| Write report file | Generate md / docx / tex | Yes |
| Delete results | Clean files or directories | Yes |
