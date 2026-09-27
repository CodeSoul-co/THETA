# THETA 0.3.10

**English** | [中文](https://github.com/CodeSoul-co/THETA/blob/main/docs/releases/desktop-v0.3.10.zh.md)

## Downloads

| Platform | Installer |
| --- | --- |
| macOS Apple Silicon | [DMG](https://github.com/CodeSoul-co/THETA/releases/download/desktop-v0.3.10/THETA-0.3.10-mac-arm64.dmg) |
| Windows x64 | [EXE](https://github.com/CodeSoul-co/THETA/releases/download/desktop-v0.3.10/THETA-0.3.10-win-x64.exe) |

## Model selection

- New analyses select LDA by default and display a recommendation badge. Saved model choices remain unchanged.
- Traditional models appear first. THETA moves to the end of the model choices and is explicitly marked Beta.
- Conversation recommendations prefer LDA as an interpretable baseline without embedding downloads, while respecting explicit model requests and research needs.
- Web and desktop share these changes.

## Dataset results and training fixes

- One dataset selector now controls full-data, training, validation and test metrics, figures, plotting data and chart references. Existing split-aware jobs can regenerate charts without retraining.
- Mean weights retain more precision and show differences from the full-data mean. NVDM values are correctly displayed as signed latent coordinates, not shares.
- Fix THETA frozen-embedding mode returning a constant zero loss and skipping topic-model updates. Small datasets and tail batches are retained. Embeddings stay frozen while the topic model learns from reconstruction loss and KL regularization. Earlier zero-loss THETA jobs require retraining; regenerating figures cannot repair an untrained model.
- When the GitHub API is unavailable or rate-limited, update checks use the same repository's public release feed and checksums. Installer size and SHA-256 verification remain mandatory.

Installed versions from 0.3.8 can check for this release in Settings → App updates. Automatic checks show a download notification for a newer compatible version. Windows supports confirmed restart-to-install. The unsigned Mac preview downloads and verifies the DMG, then opens it for replacement in Applications; it does not silently replace the app. Projects and settings are preserved.

Python and compute dependencies are bundled. Model weights and API keys are not included.

[Help documentation](https://codesoul-co.github.io/THETA/) · [Report a problem](mailto:duanzhenke@code-soul.com)
