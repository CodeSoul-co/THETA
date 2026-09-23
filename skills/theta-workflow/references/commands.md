# THETA Commands

Use current repository entrypoints. Do not invent old numbered scripts unless they exist in the checkout.

## Current Script Entrypoints

Read-only listing can be done with:

```bash
find scripts src/models -maxdepth 2 -type f | sort
```

Known current scripts:

- `scripts/env_setup.sh` - install/setup environment dependencies. Requires confirmation.
- `scripts/quick_start.sh` - beginner workflow for a dataset. Requires confirmation before execution.
- `scripts/clean_data.sh` - clean or prepare data. Requires confirmation because it can write files.
- `scripts/train_theta.sh` - train THETA. Requires confirmation.
- `scripts/train_baseline.sh` - train baselines. Requires confirmation.
- `scripts/visualize.sh` - generate visualizations. Requires confirmation when it writes outputs.
- `scripts/sweep_topics.sh` - run topic-count sweeps. Requires confirmation because it can be expensive and write many results.
- `scripts/scrape.sh` - scrape data. Requires confirmation because it can access network and write data.

Python entrypoints:

- `src/models/run_pipeline.py` - expert end-to-end pipeline.
- `src/models/prepare_data.py` - data preparation and embedding workflow.
- `src/models/main.py` - lower-level training CLI.
- `src/models/visualization/run_visualization.py` - visualization generation.

## Safe Help Checks

Help commands are read-only and can run without confirmation:

```bash
python src/models/run_pipeline.py --help
python src/models/prepare_data.py --help
python src/models/main.py --help
```

## Beginner Workflow

Generate this command for review when the user wants a quick run:

```bash
bash scripts/quick_start.sh <dataset_name> --language chinese
```

Use `--language english` for English datasets if supported by the script help.

Do not execute it until the user confirms, because it may prepare data, train models, and write outputs.

## Expert Pipeline Pattern

Use `run_pipeline.py --help` before finalizing exact flags in a fresh checkout. A typical reviewed command should include:

- Dataset identifier.
- Model list.
- Number of topics.
- Language.
- Epochs or runtime constraints.
- Embedding provider/mode where supported.
- Output directory or experiment name if supported.

Executing the pipeline requires confirmation.

## Embedding Commands

For `zero_shot`, cloud embedding may be used only after confirmation:

```text
EMBEDDING_PROVIDER=cloud
EMBEDDING_CLOUD_PROVIDER=openai
```

For `supervised`, `unsupervised`, or finetune-capable workflows, use local model paths:

```text
EMBEDDING_PROVIDER=local
QWEN_MODEL_0_6B=<local path>
SBERT_MODEL_PATH=<local path>
```

Creating or editing `.env` requires confirmation. Never print API key values.

## Confirmation Before Execution

Before executing a command, include:

```text
我准备执行以下操作：

操作：
<what will run>

命令：
<command>

影响的文件或目录：
<paths>

可能的资源消耗：
<compute / GPU / storage / API cost>

是否会覆盖已有文件：
<yes / no / unknown, explain>

请确认是否继续。
```
