# THETA Agent

**English** | [中文](README_zh.md)

Discuss research goals, import data, choose models, confirm training, and interpret results in natural language. The CLI and Web workbench share the same local Agent core. Normal use does not require an account, MySQL, Redis, or Go services.

## Install and launch

Install Node.js 22.13+, pnpm, and a Python compute environment. See [engine installation](../doc/getting-started/installation.md) for topic-model dependencies and the [statistics guide](docs/free-analysis.en.md) for free analysis.

```bash
cd agent
pnpm install --frozen-lockfile
pnpm build
# First-time setup only; preserve existing private configuration.
cp .env.example .env.local
pnpm start
```

Enter your model endpoint, model name, and API key in private `agent/.env.local`. Supported providers include DeepSeek, OpenAI, MiniMax, GLM, and custom Chat Completions-compatible services. Process environment takes precedence over Agent configuration, then repository configuration. `THETA_ENV_FILE` selects one explicit file. Do not commit keys.

You can also run `./theta` from the repository root. On Windows, run `node agent/cli/bin/theta.mjs`. For the Web interface, use `./theta-web start` and follow the [Web guide](../frontend/README.md). The [desktop app](../docs/desktop.md) bundles the runtime for users who do not want to set up a development environment.

## Conversations and data

Describe your research question or drop a file path into the terminal. The Agent discusses the unit of analysis and evidence needed, then selects methods for your goal. Text analysis can combine descriptive statistics with topic modeling or use either separately.

| Command | Purpose |
| --- | --- |
| `/model`, `/models` | Select a conversation model |
| `/attach /path/to/data.csv` | Import data; optionally add instructions after the path |
| `/new`, `/sessions`, `/resume` | Create, list, and resume sessions |
| `/context`, `/contexts` | Inspect and reuse research context |
| `/mode free` | Switch to free statistical analysis |
| `/reports`, `/open <number>` | Find and open reports |
| `/knowledge` | Browse the local knowledge catalog |
| `/trace` | Inspect the latest execution trace |
| `/deny`, `/help`, `/exit` | Reject an action, show help, or exit |

Supported inputs include CSV, TSV, TXT, Markdown, JSON, JSONL, NDJSON, Excel, Parquet, PDF, and DOCX. Local imports are limited to 200 MiB; training normalization supports up to one million rows. Original files are retained. Training uses verified managed copies with selected text, time, label, and covariate columns.

Data understanding may send limited excerpts to your configured conversation model. Common email, phone, and credential patterns are masked, but this is not complete anonymization. Prepare sensitive data before importing it.

## Compute, confirmation, and results

Training, cloud embeddings, result preparation, and deeper interpretation each require confirmation. Cards identify data, models, parameters, runtime limits, and external request scope. Revised plans require new confirmation. Status queries do not submit duplicate jobs.

Local compute defaults to CPU. A source environment with configured CUDA can explicitly select a GPU. `THETA_PYTHON` selects Python; otherwise the Agent prefers `agent/.venv`, then `python3`. Run `./theta doctor --json` to inspect readiness.

THETA zero-shot supports local or cloud embeddings. Cloud requests are bound to the confirmed endpoint, text scope, and budget. Fine-tuning, CTM, and BERTopic require compatible local models. Missing model weights are not downloaded automatically. Plot language controls chart presentation without translating source text.

Training uses independent background processes. Exiting the CLI or stopping generation does not cancel training; cancellation is a separate action. Reuse the same `THETA_AGENT_HOME` to restore sessions, data, and jobs. The default directory is `.theta_agent/`.

Result preparation uses native plotting code and delivers available matrices, tables, figures, and HTML reports. Missing time series or training history is reported rather than fabricated. Deeper interpretation saves a separate report and distinguishes observations from inferences. Original training results are retained.

## Extensions and references

- [Free analysis](docs/free-analysis.en.md): statistics, econometrics, forecasting, survival, and optimization.
- [Visualization skill](docs/data-viz-skill.en.md): 22 editable Python chart templates.
- [Local embeddings](docs/local-embedding.en.md).
- [Training progress](docs/training-progress.en.md), [confirmation cards](docs/web-confirmation-cards.en.md).
- [HTTP API](docs/THETA_AGENT_API.md).

External agents can inspect tools with `theta tools list` and call `theta tools call <name> --session <id>`. The host displays confirmation cards and approves or rejects actions. Approval endpoints must not be exposed as tools the language model can call directly.
