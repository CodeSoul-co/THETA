# THETA 0.3.13

**English** | [中文](https://github.com/CodeSoul-co/THETA/blob/main/docs/releases/desktop-v0.3.13.zh.md)

## Downloads

| Platform | Installer |
| --- | --- |
| macOS Apple Silicon | [DMG](https://github.com/CodeSoul-co/THETA/releases/download/desktop-v0.3.13/THETA-0.3.13-mac-arm64.dmg) |
| Windows x64 | [EXE](https://github.com/CodeSoul-co/THETA/releases/download/desktop-v0.3.13/THETA-0.3.13-win-x64.exe) |

## Independent model training

When a batch contains multiple models or topic counts, configuration and startup errors are recorded separately for each item. Valid models continue training, and successful results remain accessible alongside the failed model's reason. Training progress finishes when all items have settled. The same behavior applies to manual and Agent workflows.

## Local and cloud embeddings

THETA zero-shot, CTM and BERTopic share local or cloud embedding selection. Training configuration now respects the saved embedding preference. Cloud embedding no longer requires irrelevant local weights; local encoding failures are reported instead of silently continuing without embeddings. Sending text to a cloud service still requires task confirmation. THETA fine-tuning uses local weights.

## Folder datasets and file management

Folder selection recursively includes supported files and preserves relative source names. Uploaded files can be selected for combined analysis or removed individually. Table collections support text-column selection; document collections include all extracted records. Removing a source invalidates dependent collections while preserving existing training results. File-size and count limits are replaced by warnings, and streamed uploads avoid buffering an entire file in server memory.

Python and compute dependencies are bundled. Model weights and personal API keys are not included. Existing projects and settings are retained when upgrading. The release contains a Mac update ZIP, a Windows installer and update manifests for in-app updates. This remains a preview without official signing or notarization.

[Help documentation](https://codesoul-co.github.io/THETA/) · [Report a problem](mailto:duanzhenke@code-soul.com)
