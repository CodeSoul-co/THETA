# THETA 0.3.5 · Desktop Preview

**English** | [中文](https://github.com/CodeSoul-co/THETA/blob/main/docs/releases/desktop-v0.3.5.zh.md)

THETA provides a local **Web workbench, CLI Agent, and desktop app**, with an optional workflow Skill for compatible agents.

## Download and install

| Platform | Download | Installation |
| --- | --- | --- |
| macOS Apple Silicon | [DMG](https://github.com/CodeSoul-co/THETA/releases/download/desktop-v0.3.5/THETA-0.3.5-mac-arm64.dmg) | Open the DMG and drag THETA into Applications |
| Windows x64 | [EXE](https://github.com/CodeSoul-co/THETA/releases/download/desktop-v0.3.5/THETA-0.3.5-win-x64.exe) | Run the installer |

The macOS ZIP is an alternative app archive; use the DMG for a standard installation. [SHA256SUMS.txt](https://github.com/CodeSoul-co/THETA/releases/download/desktop-v0.3.5/SHA256SUMS.txt) contains checksums for each installer. Source archives do not include the desktop runtime.

Python 3.12 and compute dependencies are bundled. You do not need a separate Python, Node.js, or Conda installation. The app is named THETA and uses the ragdoll-cat logo.

## What changed

Windows now automatically uses a compatible NVIDIA CUDA GPU for supported local models and embeddings. The installer includes CUDA 12.8-enabled PyTorch and still runs on CPU-only computers. Python and CUDA Toolkit do not need to be installed separately; GPU acceleration requires a compatible NVIDIA driver. AMD and Intel graphics use CPU in this version. macOS continues to use CPU.

Before training, THETA tests GPU allocation and kernel execution. If unavailable, it selects CPU. If an automatic GPU run fails with a CUDA error, including insufficient GPU memory, the analysis restarts once on CPU with a clean workspace. The failed log is retained, and the original timeout, cancellation and cloud request budget remain in effect. Data errors and timeouts are not automatically retried.

Execution logs show the selected device and warn when CPU fallback restarts the analysis. LDA, BTM, HDP and STM continue to use CPU. An explicitly selected CPU or CUDA device is respected.

The Windows installer is larger because it includes GPU libraries. Quit THETA before upgrading: replace the app in Applications on macOS, or run the new EXE on Windows. Projects and settings are retained. Updates are installed manually.

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
