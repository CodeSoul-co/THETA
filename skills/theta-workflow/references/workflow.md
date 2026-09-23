# THETA Workflow

This reference describes the end-to-end process for using THETA safely. Match the user's language, but keep commands and file names exact. For ready-to-use Chinese and English question templates, read `user-questions.md`.

## 1. Repository Gate

When this skill is active, first determine whether the current workspace or a known local path is a usable THETA repository. This read-only gate comes before environment setup, `.env` creation, dataset conversion, model recommendation, or training commands.

Check for:

- `README.md`
- `.env.example`
- `scripts/`
- `src/models/run_pipeline.py`
- `src/models/prepare_data.py`
- `src/models/main.py`

If these files exist, continue with task intake and preflight.

If the repository is missing, cloning is the first required action and it requires explicit confirmation:

```bash
git clone https://github.com/CodeSoul-co/THETA.git
```

Do not create environments, install dependencies, create `.env`, download models, inspect datasets, or generate run commands before the repository exists.

## 2. Task Intake

Ask for the minimum information needed to choose a path:

- Dataset path.
- Text column.
- Language: Chinese, English, or mixed.
- Time column, if topic evolution is needed.
- Covariates, especially columns that should become `cov_*`.
- Goal: quick run, best topic quality, interpretability, topic evolution, or paper-grade comparisons.
- Resource limits: local CPU/GPU, time budget, cloud API availability, and whether text may leave the machine.

Do not ask for API keys in chat. Ask the user to place secrets in `.env` themselves.

Use Chinese questions for Chinese users and English questions for English users. If the user mixes languages, prefer the language of the latest user message.

## 3. Repository and Environment Preflight

Read-only checks can run without confirmation:

- Git status and branch.
- Required files: `README.md`, `.env.example`, `scripts/`, `src/models/run_pipeline.py`, `src/models/prepare_data.py`, `src/models/main.py`.
- Python version and conda availability.
- `.env` existence and which keys are present, without printing values.
- Current THETA scripts and command help.
- Dataset schema and basic text quality.

Use:

```bash
python skills/theta-workflow/scripts/inspect_theta_env.py --dataset <path> --text-column <column> --mode <mode>
```

## 4. Mode and Embedding Selection

Choose the modeling mode before selecting embedding behavior:

- `zero_shot`: cloud embedding may be used after explicit confirmation.
- `supervised`: must use local model paths because finetuning-capable workflows require local embeddings.
- `unsupervised`: must use local model paths because finetuning-capable workflows require local embeddings.

If the user wants cloud embedding in a non-`zero_shot` mode, explain the constraint and propose local Qwen/SBERT configuration.

## 5. Data Preparation

Inspect first. If conversion is required, propose a new derived file rather than modifying original data.

Common mapping:

- Source text column -> `text`.
- Source time column -> `timestamp`.
- Covariates -> `cov_<name>`.

Writing converted data requires confirmation. Include source path, output path, mapping, and overwrite status.

## 6. Model Recommendation

Recommend based on the user's goal and data:

- Unknown topic count: HDP or BERTopic for exploration.
- Short text: BERTopic or BTM when available.
- Time column and evolution goal: DTM plus THETA.
- Covariates and interpretability goal: STM plus THETA.
- Semantic topic quality: THETA as the main model.
- Paper comparison: LDA, CTM, BERTopic, ProdLDA, and THETA.

Start with a small experiment when cost or runtime is unclear.

## 7. Command Review and Execution

Generating commands does not need confirmation. Executing commands usually does.

Before running any command that writes files, calls APIs, trains models, installs packages, switches branches, or deletes data, show:

- Operation.
- Command.
- Affected files/directories.
- Resource cost.
- Whether existing files will be overwritten.

Proceed only after explicit confirmation.

## 8. Results and Iteration

Read existing outputs without confirmation. Explain:

- Topic coherence.
- Topic diversity.
- Exclusivity.
- Topic overlap.
- Whether topic count is too high or too low.
- Whether preprocessing, stopwords, vocabulary size, or model family should change.

Confirm before running any second-round experiments or writing report files.
