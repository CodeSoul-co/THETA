# THETA Web workbench

**English** | [中文](README_zh.md)

The `frontend/` directory contains the Web interface. This repository maintains the local workbench; it does not use the former hosted website or a remote fallback backend. Conversation mode shares the CLI Agent core. Conversation and manual modes use the same result view and consultation sidebar.

## Local setup

Prepare Node.js 22.13+, Agent dependencies, private `agent/.env.local`, and the Python compute environment using the [Agent guide](../agent/README.md). Install the frontend dependencies once:

```bash
npm --prefix frontend ci
```

From the repository root on macOS or Linux:

```bash
./theta-web start
./theta-web status
./theta-web restart
./theta-web stop
./theta-web check
```

`start` builds, checks, and launches the frontend and both APIs in the background. `status` shows processes, health, logs, and storage. `restart` rebuilds before restarting. `stop` stops Web services without deleting data or cancelling submitted training. `check` validates the build and worker contract without starting training.

Open `http://127.0.0.1:4320/workbench?mode=conversation`. Switch between conversation and manual modes at the top. Closing the terminal does not stop services; run `start` again after reboot. There is no automatic system-start service.

| Service | Address |
| --- | --- |
| Frontend | `http://127.0.0.1:4320` |
| Conversation API | `http://127.0.0.1:4318` |
| Manual API | `http://127.0.0.1:4321` |

Port conflicts are reported without terminating unrelated processes. Logs are stored under `.local/workbench/`. Preserve `.theta_agent/` and `.local/manual-workbench/`, including their databases, uploads, jobs, and reports. Restarting does not recreate projects or resubmit completed training.

Keep using the same browser origin: `localhost` and `127.0.0.1` do not share browser drafts. Finish active uploads or generated responses before restarting. Submitted background training runs independently.

For individual service development, use `npm --prefix agent run web:api`, `npm --prefix agent run web:manual`, and `npm --prefix frontend run dev`. Do not run them on ports already used by the launcher. Configure frontend `.env.local` from `.env.example` only when needed; preserve existing private settings.

The source launcher uses local development mode. Desktop apps use a production entry protected by an application token. Remote Host/Origin values are rejected. `THETA_LOCAL_AUTH_ENABLED=false` disables local proxy access. Model settings come from private `agent/.env.local`; restart APIs after editing it.

## Analysis and text processing

Manual uploads must return a valid file identifier before analysis configuration becomes available. Missing model files or dependencies are reported before training.

Plot language changes chart labels, axes, and legends; it neither translates source text nor selects a tokenizer. The engine handles mixed scripts and built-in stopwords. Chinese uses jieba; Japanese tokenization is basic and is not a specialized morphological analyzer.

The text-processing settings accept a UTF-8 TXT stopword file, one word per line, with `#` comments and limits of 512 KB or 20,000 words. A custom list replaces built-in stopwords for that analysis. Restore defaults to resume automatic selection. Downloads export the actual built-in resources under `src/models/resources/stopwords/`. Confirmed tasks retain their own stopword configuration.

## Workbench modes

- **Conversation:** research is organized into project conversations. The Agent prepares plans, checks progress, reads results, and performs confirmed interpretation.
- **Manual:** project cards organize data, parameters, jobs, and results without creating training conversations.
- **Consultation sidebar:** answers questions about the project or selected CSV/JSON chart evidence. It does not create, modify, or cancel training. Consultation history is scoped to the project or task.

Training, result preparation, and deeper interpretation have separate confirmation cards. Stopping generation interrupts the current model request; cancelling training requires a separate action. See [confirmation cards](../agent/docs/web-confirmation-cards.en.md).
