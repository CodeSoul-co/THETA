# Local setup guide

**English** | [中文](opensource-agent.zh.md)

THETA provides a Web workbench, CLI Agent, and desktop apps. Local use does not require an account, Go services, MySQL, or Redis. Projects and results stay on your computer. You can configure your own cloud conversation and embedding APIs.

## Desktop apps

Download the Mac DMG or Windows EXE from [GitHub Releases](https://github.com/CodeSoul-co/THETA/releases/tag/desktop-v0.3.0). Python and CPU compute dependencies are bundled. Download model weights as needed and enter your API keys in Settings. See the [desktop guide](desktop.md).

## Source installation

Prepare Node.js 22.13+, pnpm, and a Python compute environment. Follow the [Agent guide](../agent/README.md) and [engine installation](../doc/getting-started/installation.md). From the repository root:

```sh
pnpm --dir agent install --frozen-lockfile
npm --prefix agent run build
npm --prefix frontend ci
# First-time setup only; preserve existing configuration.
cp agent/.env.example agent/.env.local
```

Enter your conversation-model settings in private `agent/.env.local`, then choose an entry point:

```sh
./theta
./theta-web start
```

The Web address is `http://127.0.0.1:4320/workbench?mode=conversation`. Switch between conversation and manual modes at the top. Use `./theta-web status` to inspect services, `restart` to rebuild and restart, and `stop` to shut down Web services. On Windows, launch the CLI with `node agent/cli/bin/theta.mjs`.

## Storage and runtime

Services listen only on the local computer. Source installations use `.theta_agent/` and `.local/manual-workbench/`; desktop apps use the operating system's THETA application-data directory. Preserve these directories to retain projects, uploads, and results.

Refreshing the page does not cancel accepted compute jobs. After an interruption, inspect saved job records before resubmitting. Training, cloud embeddings, and result interpretation require explicit confirmation. Installers contain no model weights, personal data, or API keys.

Details: [CLI Agent](../agent/README.md), [Web workbench](../frontend/README.md), [desktop apps](desktop.md).
