# THETA 0.3.3 · Desktop Preview

**English** | [中文](https://github.com/CodeSoul-co/THETA/blob/main/docs/releases/desktop-v0.3.3.zh.md)

THETA provides a local **Web workbench, CLI Agent, and desktop app**, with an optional workflow Skill for compatible agents.

## Download and install

| Platform | Download | Installation |
| --- | --- | --- |
| macOS Apple Silicon | [DMG](https://github.com/CodeSoul-co/THETA/releases/download/desktop-v0.3.3/THETA-0.3.3-mac-arm64.dmg) | Open the DMG and drag THETA into Applications |
| Windows x64 | [EXE](https://github.com/CodeSoul-co/THETA/releases/download/desktop-v0.3.3/THETA-0.3.3-win-x64.exe) | Run the installer |

The macOS ZIP is an alternative app archive; use the DMG for a standard installation. [SHA256SUMS.txt](https://github.com/CodeSoul-co/THETA/releases/download/desktop-v0.3.3/SHA256SUMS.txt) contains checksums for each installer. Source archives do not include the desktop runtime.

Python 3.12 and CPU compute dependencies are bundled. You do not need a separate Python, Node.js, or Conda installation. The app is named THETA and uses the ragdoll-cat logo.

The home page links to the restored [documentation site](https://codesoul-co.github.io/THETA/). The team page shows the current member list.

Missing API configuration and embedding weights now include setup guidance. Data above 20 MB and compute-intensive models show a prominent time notice before training.

Click **Done** to save changes in Settings. Saved API keys are displayed as asterisks.

Chart citation badges now follow the assistant’s current attachments. Sending clears the submitted text immediately while preserving any new draft typed during the response.

TXT, Markdown, PDF, and Word inputs use text previews without column selection. If the data directory is not writable, another location can be selected. Windows upgrades remove old application files and UI caches while preserving projects, results, models, and settings.

## Models and data

Open Settings to enter your conversation-model endpoint, model name, and API key. Local embeddings use a compatible model directory; the default suggestion is Qwen3-Embedding-0.6B. Cloud embeddings accept a user-configured API, with GLM embedding-3 as the preset.

Installers contain no user keys, private data, or model weights. Traditional models such as LDA do not require neural weights. Download local embedding models separately. Cloud embeddings currently support THETA zero-shot and require confirmation of the text scope and request budget.

Projects and results are stored locally. Updating the application preserves application data. Desktop and source installations have independent data directories.

## Other entry points

- **Web:** after source setup, run `./theta-web start` and visit `http://127.0.0.1:4320/workbench`.
- **CLI Agent:** after Agent setup, run `./theta`.
- **Skill:** use `skills/theta-workflow/SKILL.md` with a compatible agent.

See the [README](https://github.com/CodeSoul-co/THETA/blob/main/README.md) and [desktop guide](https://github.com/CodeSoul-co/THETA/blob/main/docs/desktop.md).

## Preview requirements

This release is unsigned and not notarized. The operating system may show a security warning. Supported installers are macOS arm64 and Windows x64; native Intel Mac and Windows ARM installers are not provided. Updates are installed manually.
