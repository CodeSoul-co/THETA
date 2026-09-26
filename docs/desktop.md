# THETA desktop app

**English** | [中文](desktop.zh.md)

The desktop app combines the CLI Agent, Web workbench, and Python compute engine. It offers conversation and manual modes. Python 3.12 and compute dependencies are bundled; Python, Node.js, and Conda do not need to be installed separately. Model weights are not included.

## Windows GPU acceleration

Windows defaults to automatic device selection for local training and embeddings. The installer includes Python and a CPU runtime. On a compatible NVIDIA system, the first automatic GPU task downloads a publisher-verified CUDA 12.8 component from PyTorch (about 2.6 GiB; allow about 9 GiB free during setup). The component is cached, supports resumed downloads, and requires a compatible NVIDIA driver. A separate Python or CUDA Toolkit installation is not required. AMD/Intel GPUs and incompatible NVIDIA devices use CPU in this version. macOS continues to use CPU.

THETA tests a real GPU allocation and operation before training. If the probe fails, it uses CPU immediately. If a CUDA error occurs during an automatic GPU run (including insufficient GPU memory), THETA preserves the failed log and restarts the analysis once on CPU in a clean workspace. The original time limit, cancellation and cloud request budget remain in force. Data errors and timeouts are not retried. Execution logs show the selected device and any fallback; a restarted analysis can take longer.

LDA, BTM, HDP and STM use their existing CPU implementations. CLI Agent plans accept `device: "auto"`, `"cpu"`, or `"cuda:0"`; an explicit selection is respected. If the component cannot be downloaded, verified, or loaded, THETA uses the bundled CPU runtime. Download and setup progress appear in execution logs. Component requests contain no research text. GPU components are stored in the application data directory, outside the installed program; updating the app preserves this cache. Neural model weights still need to be provided separately.

## Download and install

Download from [GitHub Releases](https://github.com/CodeSoul-co/THETA/releases/tag/desktop-v0.3.7):

| Platform | Installer | Installation |
| --- | --- | --- |
| macOS Apple Silicon | `THETA-0.3.7-mac-arm64.dmg` | Open the DMG and drag THETA into Applications |
| Windows x64 | `THETA-0.3.7-win-x64.exe` | Run the installer |

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
- Windows: new installations prefer `THETA-data` inside the installation directory, falling back to `%APPDATA%/THETA` when that directory is not writable. Upgrades keep the existing data location. Settings shows the active path.

Settings, projects, uploads, results, and model weights live in a dedicated data directory and are retained during upgrades. Installation-local `THETA-data` is also retained on uninstall. The Windows installer replaces registered older versions and removes their program files. The first launch of a new version clears disposable interface caches. Desktop storage is independent of the source installation's `.theta_agent` and `.local/manual-workbench` directories.

Windows installs for the current user by default. Data is not written to the drive root; the dedicated `THETA-data` folder is managed separately from program files. If the data directory is not writable, the app asks you to choose a dedicated folder, such as `D:\THETA-data`, and remembers it. You can also start `THETA.exe --data-dir="D:\THETA-data"`. Choosing a new folder does not migrate existing data; select an existing THETA data directory to reopen its projects.

TXT, Markdown, PDF, and Word uploads open a text preview directly, without column selection. Structured files retain text, time, and label column selection. Scanned PDFs require text recognition first.

Services listen only on loopback and use a per-launch access token. The preferred workbench port is 14320; another available port is selected if needed. Quitting closes the app's services. Submitted training follows the existing worker lifecycle and is not resubmitted automatically.

## Upload and replace data

Drag files or folders into an upload area, or use the file picker. “Replace dataset” keeps the current project and switches only after a successful upload. Original data and historical results are retained. Conversation mode starts a fresh conversation within the same project when data changes, so previous task settings and approvals are not reused.

Text, Markdown, PDF and Word documents are split by actual paragraphs. Long paragraphs are split at sentence boundaries, and Word table rows are included. Document folders recursively combine TXT, Markdown, PDF and DOCX files while retaining source filenames, pages and paragraph positions. A collection supports up to 500 files and 200 MiB in total. Upload tables separately to choose their text columns; scanned PDFs require text recognition first.

Individual files support up to 200 MiB. Files larger than 10 MiB uploaded with older versions may have been truncated; upload the originals again. Invalid formats, encryption, incomplete transfers and permission errors receive actionable messages. Training logs show specific failure details. Corpora with fewer than 10 records use small-sample word-frequency defaults; use these results only to verify the workflow and add independent records for substantive analysis.

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

`prepare:python` installs standalone Python and platform compute dependencies from platform locks. On macOS it repairs native-library paths. It does not download model weights. Allow several GB for the environment and additional space for packaging.

`prepare:runtime` compiles the Agent and Next.js standalone app, copying only required engine files, templates, and skills. Private configuration, user data, training results, and model directories are excluded.

Output is under `desktop/release`: macOS app, DMG, and ZIP; Windows NSIS EXE. Build artifacts are not committed to Git. The desktop workflow supports manual builds and `desktop-v*` release tags. Tagged builds publish a preview after both platforms pass validation. Updates are installed manually from Releases.
