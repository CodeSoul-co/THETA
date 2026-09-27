# THETA 0.3.9

**English** | [中文](https://github.com/CodeSoul-co/THETA/blob/main/docs/releases/desktop-v0.3.9.zh.md)

## Downloads

| Platform | Installer |
| --- | --- |
| macOS Apple Silicon | [DMG](https://github.com/CodeSoul-co/THETA/releases/download/desktop-v0.3.9/THETA-0.3.9-mac-arm64.dmg) |
| Windows x64 | [EXE](https://github.com/CodeSoul-co/THETA/releases/download/desktop-v0.3.9/THETA-0.3.9-win-x64.exe) |

## Dataset splits

- Enable custom splitting in the data selection card: adjustable training / validation / test ratios, random or sequential allocation, or three separate uploads with their own formats and column selections.
- With custom splitting disabled, use a 70% / 30% training / validation split and test on all data. The result page explicitly identifies this overlapping test.
- Fit vocabulary and models on training records. Review full-data, training, validation and test results separately, including metrics, average topic weights and downloadable document results.
- Shared Web and desktop behavior, with reproducible ratio splitting also available in CLI Agent training plans.

Python and compute dependencies are bundled; model weights are optional downloads and API keys are supplied by each user. In-app update checks are available from version 0.3.8. Projects and settings are preserved. The Mac preview remains unsigned and opens a verified DMG for replacement; Windows supports confirmed restart-to-install.

[Help documentation](https://codesoul-co.github.io/THETA/) · [Report a problem](mailto:duanzhenke@code-soul.com)
