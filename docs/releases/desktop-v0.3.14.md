# THETA 0.3.14

**English** | [中文](https://github.com/CodeSoul-co/THETA/blob/main/docs/releases/desktop-v0.3.14.zh.md)

| Platform | Installer |
| --- | --- |
| macOS Apple Silicon | [DMG](https://github.com/CodeSoul-co/THETA/releases/download/desktop-v0.3.14/THETA-0.3.14-mac-arm64.dmg) |
| Windows x64 | [EXE](https://github.com/CodeSoul-co/THETA/releases/download/desktop-v0.3.14/THETA-0.3.14-win-x64.exe) |

Launch no longer requests a Keychain password. API keys use local user-owned encryption; users upgrading from legacy Keychain storage re-enter their API keys once. Windows runs with ordinary user privileges and the installer does not offer privilege elevation. Operating-system warnings for unsigned applications remain subject to platform policy; this preview has no official signing or notarization.

Training submission, progress, cancellation and results now use one worker HTTP API. The desktop worker service runs separately from the UI and Agent service. Web and CLI use the same contract. An optional HTTPS worker endpoint can replace local transport; remote deployments must provide worker-accessible data and artifact storage.

The conversation sidebar includes **Skill Manager**. Import folders, ZIP, tar.gz or SKILL.md, install from a GitHub repository or skill folder, enable or disable skills, export packages and remove imported skills. Bundled Data Viz plotting templates appear in the same catalog. Agent-requested downloads use the same persistent store and become visible here. Importing a skill does not execute its scripts or install dependencies.

Large-corpus evaluation counts only selected topic words in bounded batches, and perplexity avoids full document-by-vocabulary probability copies. Evaluation progress identifies the current metric instead of remaining at the last training iteration. Exact observed corpus counts are retained.

Python and compute dependencies are bundled. Personal API keys and model weights are excluded. Existing projects are retained when upgrading.

[Help documentation](https://codesoul-co.github.io/THETA/) · [Report a problem](mailto:duanzhenke@code-soul.com)
