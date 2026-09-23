# Ways to use THETA

**English** | [中文](https://github.com/CodeSoul-co/THETA/blob/main/doc/getting-started/interfaces.zh.md)

Choose the desktop app for installation without a development environment, the Web workbench for a local browser, or the CLI Agent for terminal conversations. All three use local project storage and the same analysis engine.

## Desktop app

Download the installer from [GitHub Releases](https://github.com/CodeSoul-co/THETA/releases). Use the DMG for Apple Silicon Macs and the EXE for Windows x64. On macOS, drag THETA into Applications; on Windows, run the installer.

Python and CPU compute dependencies are included. You do not need to install Python, Node.js, or Conda. Model weights and user credentials are not bundled. This is an unsigned preview; your operating system may show a security prompt.

Open Settings inside THETA to enter your conversation model endpoint, model name, and API key. Local embeddings accept a compatible model directory, with Qwen3-Embedding-0.6B as the default suggestion. Cloud embeddings accept your endpoint and key, with GLM embedding-3 as the preset. THETA zero-shot supports cloud embeddings; fine-tuning, CTM, and BERTopic require compatible local weights.

Projects and results remain on your computer. Application data is stored under `~/Library/Application Support/THETA` on macOS and `%APPDATA%/THETA` on Windows. Installing an update preserves this data.

## Web workbench

For source use, install Node.js 22.13+, pnpm, and the [Python compute dependencies](installation.md). Create private `agent/.env.local` from the example only on first setup, then configure your model endpoint and key. Preserve existing configuration.

From the repository root:

```sh
pnpm --dir agent install --frozen-lockfile
npm --prefix frontend ci
./theta-web start
```

Open `http://127.0.0.1:4320/workbench`. Choose conversation or manual mode at the top. Use `./theta-web status` to check services and `./theta-web stop` to stop them.

The launcher starts local services and preserves projects in `.theta_agent/` and `.local/manual-workbench/`. Keep these directories when updating source code.

## CLI Agent

After source setup, run `./theta` from the repository root. On Windows, use `node agent/cli/bin/theta.mjs`. Describe your question and attach a data file. Review model settings and confirm the proposed computation before execution.

Use `/help` for commands, `/sessions` to find conversations, and `/resume` to continue one. See the [CLI guide](https://github.com/CodeSoul-co/THETA/blob/main/agent/README.md) for configuration and command details.

## Workflow skill

Compatible agents can read `skills/theta-workflow/SKILL.md` to guide data preparation, model selection, execution confirmation, and result interpretation. See the [workflow instructions](https://github.com/CodeSoul-co/THETA/tree/main/skills/theta-workflow).
