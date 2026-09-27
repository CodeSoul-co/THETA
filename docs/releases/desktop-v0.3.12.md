# THETA 0.3.12

**English** | [中文](https://github.com/CodeSoul-co/THETA/blob/main/docs/releases/desktop-v0.3.12.zh.md)

## Downloads

| Platform | Installer |
| --- | --- |
| macOS Apple Silicon | [DMG](https://github.com/CodeSoul-co/THETA/releases/download/desktop-v0.3.12/THETA-0.3.12-mac-arm64.dmg) |
| Windows x64 | [EXE](https://github.com/CodeSoul-co/THETA/releases/download/desktop-v0.3.12/THETA-0.3.12-win-x64.exe) |

## Reports with traceable figures and tables

Academic reports now follow a narrative sequence: develop a research claim, show one relevant training figure or table, then interpret its numerical evidence and implications before continuing the argument. Adjacent exhibits and figure lists without substantive interpretation are rejected. Each exhibit has a fixed number, caption, source and data scope. The report Agent uses those identifiers to explain numerical evidence and support its conclusions, distinguishing the full corpus, training, validation and test data. Table excerpts state their limits; the original CSVs remain available in the download bundle. Missing figures are not invented or regenerated automatically.

Repeated chapter titles and inconsistent heading levels are corrected. PDF export supports Markdown tables, lists and local images, keeps figures with captions, and repeats table headers when splitting long tables. Downloads include a standalone PDF, a Markdown file, and a ZIP containing Markdown, images and source tables. Keep the assets folder beside the Markdown file when editing it.

Existing reports can be explicitly regenerated using the configured model API. Generation remains asynchronous and starts only on click. If writing fails, the previous report stays available; a PDF-only retry does not repeat model calls. Web and desktop use the same implementation.

Python and compute dependencies are bundled. Model weights and personal API keys are not included. Existing projects and settings are retained when upgrading.

[Help documentation](https://codesoul-co.github.io/THETA/) · [Report a problem](mailto:duanzhenke@code-soul.com)
