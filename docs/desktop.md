# THETA desktop app

**English** | [中文](desktop.zh.md)

The desktop app combines the CLI Agent, Web workbench, and Python compute engine. It offers conversation and manual modes. Python 3.12 and CPU compute dependencies are bundled; Python, Node.js, and Conda do not need to be installed separately. Model weights are not included.

## Download and install

Download from [GitHub Releases](https://github.com/CodeSoul-co/THETA/releases/tag/desktop-v0.3.2):

| Platform | Installer | Installation |
| --- | --- | --- |
| macOS Apple Silicon | `THETA-0.3.2-mac-arm64.dmg` | Open the DMG and drag THETA into Applications |
| Windows x64 | `THETA-0.3.2-win-x64.exe` | Run the installer |

This is an unsigned preview without macOS notarization. Your operating system may show a security warning. Native Intel Mac and Windows ARM installers are not provided.

Compare your downloaded file with `SHA256SUMS.txt` on the release page. On macOS, run `shasum -a 256 filename.dmg`; in Windows PowerShell, run `Get-FileHash filename.exe -Algorithm SHA256`.

## Configure models

The app opens directly into the workbench. Open Settings:

- **Model API:** enter the conversation model provider, Base URL, model name, and your API key.
- **Local embeddings:** select a complete compatible model directory containing configuration, tokenizer, and weights. The default suggestion is `Qwen/Qwen3-Embedding-0.6B`. The model name is a label; the selected directory supplies the files. CTM and BERTopic can use a separate Sentence Transformers model.
- **Cloud embeddings:** the preset is GLM / Zhipu at `https://open.bigmodel.cn/api/paas/v4`, model `embedding-3`. Set the endpoint, model, dimensions, and a separate embedding API key. OpenAI-compatible embedding services are also supported.

You can configure models later. Traditional algorithms such as LDA do not need neural model weights. Download local embedding weights yourself using the links in Settings. Cloud embeddings are supported for THETA zero-shot; each task requires confirmation of the text scope and request budget. THETA fine-tuning, CTM, and BERTopic require compatible local models.

Saved settings apply to new tasks immediately. Running tasks retain their starting configuration. Keys are encrypted through Electron safeStorage using operating-system facilities, stored in the application data directory, and never displayed again in the form.

Installers contain no API keys or personal configuration. Settings retained after reinstallation come from your application data directory.

## Data and logs

Use the application menu to open data and logs or restart local services.

- macOS: `~/Library/Application Support/THETA`
- Windows: `%APPDATA%/THETA`

Settings, projects, uploads, results, and caches are separate from the application installation. Updating or uninstalling does not actively delete these directories. Desktop storage is independent of the source installation's `.theta_agent` and `.local/manual-workbench` directories.

Services listen only on loopback and use a per-launch access token. The preferred workbench port is 14320; another available port is selected if needed. Quitting closes the app's services. Submitted training follows the existing worker lifecycle and is not resubmitted automatically.

## Build from source

The build machine needs Node.js 22.13+, pnpm, and uv. Build Python dependencies on the target operating system and architecture; a macOS runtime cannot be reused for Windows.

```sh
pnpm --dir agent install --frozen-lockfile
npm --prefix frontend ci
npm --prefix desktop ci
npm --prefix desktop run prepare:python
npm --prefix desktop run prepare:runtime
npm --prefix desktop test
npm --prefix desktop run smoke
npm --prefix desktop run start
npm --prefix desktop run dist
```

`prepare:python` installs standalone Python and CPU dependencies from platform locks. On macOS it repairs native-library paths. It does not download model weights. Allow several GB for the environment and additional space for packaging.

`prepare:runtime` compiles the Agent and Next.js standalone app, copying only required engine files, templates, and skills. Private configuration, user data, training results, and model directories are excluded.

Output is under `desktop/release`: macOS app, DMG, and ZIP; Windows NSIS EXE. Build artifacts are not committed to Git. The desktop workflow supports manual builds and `desktop-v*` release tags. Tagged builds publish a preview after both platforms pass validation. Updates are installed manually from Releases.
