# THETA 0.3.7

**English** | [中文](https://github.com/CodeSoul-co/THETA/blob/main/docs/releases/desktop-v0.3.7.zh.md)

## Downloads

| Platform | Installer |
| --- | --- |
| macOS Apple Silicon | [DMG](https://github.com/CodeSoul-co/THETA/releases/download/desktop-v0.3.7/THETA-0.3.7-mac-arm64.dmg) |
| Windows x64 | [EXE](https://github.com/CodeSoul-co/THETA/releases/download/desktop-v0.3.7/THETA-0.3.7-win-x64.exe) |

Python and compute dependencies are bundled. Model weights are downloaded separately; enter your own API keys in Settings. Quit THETA before installing over the previous version. Existing projects and settings are retained.

## Fixes

- Deleting a manual project releases its name and keeps removed projects out of the list after refresh. Success is shown only after the server confirms deletion. Existing data and results remain on disk.
- File selection and successful uploads show clear feedback in both workbench modes.
- Completed manual results can be copied into a conversation when Windows uses a different filename sort order. File integrity checks remain enforced.
- Windows training uses extended-length paths when exporting model files and reading results, including directories with non-ASCII names.
- Model selection can be cleared completely. Continuing with no model selected shows a prompt without starting a task; selecting another model restores its parameter panel.

Web and desktop use the same updated workbench and local services. Source users should update the repository and restart the Web workbench. Desktop users need this installer to receive the changes.

The installers are unsigned preview builds. The macOS ZIP is an alternative app archive. [SHA-256 checksums](https://github.com/CodeSoul-co/THETA/releases/download/desktop-v0.3.7/SHA256SUMS.txt) are provided for all installers.

See the [help documentation](https://codesoul-co.github.io/THETA/). Report problems to [duanzhenke@code-soul.com](mailto:duanzhenke@code-soul.com).
