# THETA 0.3.11

**English** | [中文](https://github.com/CodeSoul-co/THETA/blob/main/docs/releases/desktop-v0.3.11.zh.md)

## Downloads

| Platform | Installer |
| --- | --- |
| macOS Apple Silicon | [DMG](https://github.com/CodeSoul-co/THETA/releases/download/desktop-v0.3.11/THETA-0.3.11-mac-arm64.dmg) |
| Windows x64 | [EXE](https://github.com/CodeSoul-co/THETA/releases/download/desktop-v0.3.11/THETA-0.3.11-win-x64.exe) |

## On-demand academic reports

Generate a complete analysis report explicitly from a completed result. The report Agent reads verified statistics, redacted excerpts, model parameters, metrics and plotting data. An optional research question guides the analysis. Data description, data analysis, task analysis, model description and conclusions appear in that order.

A dedicated academic writing prompt asks for coherent paragraphs and sustained reasoning, targeting roughly 5,000–8,000 Chinese characters when the evidence supports it. The report is written in Chinese. Necessary limitations are explained concretely without repetitive disclaimers. Markdown and searchable Chinese PDF downloads are available separately. Generation continues in the background while navigating; a failed PDF export can be retried without repeating model calls. Nothing is generated until requested. The configured model API receives the evidence and may charge for inference.

## Optional topic-count experiments

Topic-count exploration is off by default. Enable it in analysis configuration and enter counts such as 5, 10, 15 and 20. Each selected model runs once per count, sequentially, with separate job identities and results labelled by requested K. Up to 12 counts and 48 model/count combinations are supported. Runs share the data-split settings. HDP uses a truncation ceiling; BERTopic uses a merge target, so the observed topic count can differ. More experiments require more time and may increase cloud embedding costs.

These features are shared by the Web workbench and desktop app.

Python and compute dependencies are bundled. Model weights and API keys are not included. Windows supports confirmed restart-to-install. The unsigned Mac preview downloads and verifies its DMG, then opens it for replacement in Applications. Existing projects and settings are retained. If an older API-only updater reports HTTP 403, install this version over the existing app once.

[Help documentation](https://codesoul-co.github.io/THETA/) · [Report a problem](mailto:duanzhenke@code-soul.com)
