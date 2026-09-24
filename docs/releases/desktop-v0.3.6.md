# THETA 0.3.6 · Desktop Preview

**English** | [中文](https://github.com/CodeSoul-co/THETA/blob/main/docs/releases/desktop-v0.3.6.zh.md)

THETA provides a local **Web workbench, CLI Agent, and desktop app**, with an optional workflow Skill for compatible agents.

## Download and install

| Platform | Download | Installation |
| --- | --- | --- |
| macOS Apple Silicon | [DMG](https://github.com/CodeSoul-co/THETA/releases/download/desktop-v0.3.6/THETA-0.3.6-mac-arm64.dmg) | Open the DMG and drag THETA into Applications |
| Windows x64 | [EXE](https://github.com/CodeSoul-co/THETA/releases/download/desktop-v0.3.6/THETA-0.3.6-win-x64.exe) | Run the installer |

The macOS ZIP is an alternative app archive; use the DMG for a standard installation. [SHA256SUMS.txt](https://github.com/CodeSoul-co/THETA/releases/download/desktop-v0.3.6/SHA256SUMS.txt) contains checksums for each installer. Source archives do not include the desktop runtime.

Python 3.12 and compute dependencies are bundled. You do not need a separate Python, Node.js, or Conda installation. The app is named THETA and uses the ragdoll-cat logo.

## What changed

- Drag-and-drop uploads and in-project dataset replacement, with original data and historical results retained.
- Folder uploads combine supported documents into a traceable text collection. Word paragraphs and tables, PDF pages and long text blocks are segmented by content.
- Fixed a shared macOS/Windows proxy limit that could truncate uploads larger than 10 MiB. The complete upload path now supports 200 MiB and checks transfer completeness.
- Clearer file-format, permissions and training errors, with retry and replacement controls. Small corpora no longer lose every word to the default document-frequency threshold.
- New Windows installations prefer an installation-local `THETA-data` folder, with writable-user-directory fallback. Existing data locations remain unchanged; upgrades and uninstall preserve the dedicated data folder.
- Fixed abrupt Windows exits when cleaning caches or upload temporary files under Unicode data paths. Service shutdown also waits for database handles to be released.

Python and compute dependencies remain bundled. Windows GPU acceleration and automatic CPU fallback are retained. Quit THETA before upgrading. Updates are installed manually.

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
