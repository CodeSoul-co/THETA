---
name: theta-workflow
description: Guide THETA topic-modeling workflows with strong user confirmation and safe execution. Use when the user mentions THETA, 主题建模, topic modeling, LDA, STM, DTM, CTM, BERTopic, policy or social-media text analysis, topic evolution, automatic modeling, text clustering, topic discovery, data preparation, embedding selection, model training, result analysis, tuning, or report generation in the THETA repository.
---

# THETA Workflow

Use this portable agent workflow skill to guide users through THETA topic-modeling work from intent clarification to result interpretation. Match the user's language: use Chinese for Chinese requests and English for English requests. Do not assume Codex-specific APIs; the core workflow relies on reading files, asking for confirmation, running ordinary shell commands when confirmed, and using the bundled read-only Python preflight helper.

## Core Safety Protocol

Distinguish read-only inspection from operations that modify files, environments, cost, compute, or git state.

Read-only inspection and advice do not need extra confirmation:

- Read README, docs, scripts, configuration templates, and result files.
- Check repository structure, file existence, Python version, conda presence, `.env` presence, model path presence, git status, and available commands.
- Inspect dataset schema, row count, column names, missing text, duplicate text, text length, time columns, and covariate columns.
- Recommend models, parameters, experiment designs, and command lines.
- Summarize existing metrics, topic words, topic tables, visualizations, and tuning suggestions.

Operations that require explicit user confirmation before execution:

- Clone or pull the THETA repository, switch branches, commit, merge, or push.
- Create conda environments, install dependencies, or run `scripts/env_setup.sh`.
- Create or edit `.env`; never print, store, or echo API keys in chat.
- Configure or use cloud embedding providers; cloud API calls can incur cost and expose text to an external service.
- Download Qwen, SBERT, or other local models.
- Modify, clean, convert, or write datasets; prefer creating a new derived file and never modify original data by default.
- Run THETA or baseline training, multi-model comparisons, tuning sweeps, large local models, GPU-heavy jobs, or commands that write results.
- Generate report files, overwrite outputs, delete caches, delete models, delete data, or clean result directories.

When confirmation is required, use this format:

```text
我准备执行以下操作：

操作：
<brief action>

命令：
<command, if any>

影响的文件或目录：
<paths>

可能的资源消耗：
<local compute / GPU / storage / API cost / none>

是否会覆盖已有文件：
<yes / no>

请确认是否继续。
```

For English users, use the matching English format:

```text
I am preparing to perform the following action:

Action:
<brief action>

Command:
<command, if any>

Affected files or directories:
<paths>

Potential resource usage:
<local compute / GPU / storage / API cost / none>

Will this overwrite existing files:
<yes / no>

Please confirm whether to continue.
```

Proceed only after an explicit confirmation such as `确认`, `可以`, `继续`, `执行`, `yes`, `proceed`, or `run it`. If the answer is ambiguous, ask once more in the user's language.

For full confirmation flows, examples, and confirmation tables, read `references/confirmation-flow.zh.md` for Chinese or `references/confirmation-flow.en.md` for English. For intake questions, read `references/user-questions.md`.

## Embedding Rule

Use cloud embedding only for `zero_shot` workflows and only after confirming API cost, provider, key presence, and data egress risk.

For `supervised`, `unsupervised`, or any workflow that needs finetuning, require local embeddings and local Qwen-compatible model paths. If the user requests cloud embedding for a finetune-capable mode, explain that THETA requires local models for that path and propose local configuration instead.

## Recommended Workflow

1. Identify whether the request belongs to THETA topic modeling. No confirmation is needed for this routing decision.
2. Immediately check whether a usable THETA repository exists before environment setup, data conversion, or training planning. Check for `README.md`, `.env.example`, `scripts/`, and `src/models/run_pipeline.py`. This read-only check does not require confirmation.
3. If no THETA repository exists, the first action is to ask for confirmation to clone `https://github.com/CodeSoul-co/THETA.git`. Do not configure environments, install dependencies, create `.env`, inspect datasets, or generate training commands until the repository exists.
4. After the repository exists, ask only the necessary task questions in the user's language: dataset path, text column, language, time column, covariates, target outcome, and resource constraints.
5. Inspect the environment, `.env` state, and dataset read-only. Use `scripts/inspect_theta_env.py` when deterministic preflight is useful.
6. Recommend model family and parameters based on data and goal.
7. Generate commands for review without executing them.
8. Before executing any mutating, costly, long-running, GPU-heavy, or file-writing command, get explicit confirmation using the standard format.
9. After execution, read outputs and explain topic quality, overlap, coherence, diversity, exclusivity, and next tuning steps.
10. Confirm again before running additional tuning experiments, writing report files, overwriting outputs, or deleting anything.

## Invocation Behavior

When an agent uses this skill, whether by reading `SKILL.md`, by loading the `skills/theta-workflow/` directory, or by named invocation such as `$theta-workflow`, treat the current workspace as the first candidate THETA repository. If the current workspace is not THETA, inspect any user-provided path. If no usable repository is found, stop at the clone confirmation step. Do not ask dataset questions or propose environment setup until the repository gate is satisfied.

Good first user prompts:

```text
Use the THETA Workflow skill in skills/theta-workflow to help me run THETA topic modeling on a policy-text dataset.
使用 skills/theta-workflow 里的 THETA Workflow skill 帮我跑一个政策文本主题建模流程。
```

If the runtime supports named skill syntax, `$theta-workflow` is also acceptable.

## Reference Routing

- Read `references/workflow.md` for the full end-to-end decision flow.
- Read `references/commands.md` before generating THETA commands; it lists current repository entrypoints and confirmation requirements.
- Read `references/result-analysis.md` when interpreting output files or recommending tuning.
- Read `references/user-questions.md` when asking the user for dataset, language, goal, resource, or output preferences.
- Read `references/confirmation-flow.zh.md` when detailed Chinese confirmation wording or examples are needed.
- Read `references/confirmation-flow.en.md` when detailed English confirmation wording or examples are needed.

## Preflight Script

Run this script only for read-only inspection:

```bash
python skills/theta-workflow/scripts/inspect_theta_env.py \
  --dataset path/to/data.csv \
  --text-column content \
  --mode zero_shot
```

Add `--json` when structured output is needed. The script must not write files, install dependencies, call APIs, or start training.
