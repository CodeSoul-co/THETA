import { mkdirSync } from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

// One Node utility process owns both APIs and the compiled Next.js server.
// No system Node, npm, shell, or Python installation is used at application startup.
const root = process.env.THETA_PROJECT_ROOT;
const home = process.env.THETA_DESKTOP_HOME;
if (!root || !home || !process.env.THETA_DESKTOP_TOKEN) throw new Error('Desktop runtime configuration is missing');
const load = relative => import(pathToFileURL(path.join(root, relative)).href);
const { createAgentServer } = await load('agent/dist/web/server.js');
const { createManualServer } = await load('agent/dist/web/manual-server.js');
const manualHome = path.join(home, 'manual');
const agentHome = path.join(home, 'agent');
mkdirSync(manualHome, { recursive: true }); mkdirSync(agentHome, { recursive: true });
const localToken = process.env.THETA_DESKTOP_TOKEN;
const servers = [createAgentServer(agentHome, undefined, { mode: 'local', manualHome, localToken }), createManualServer(manualHome, undefined, { localToken })];
async function listen(server) {
  await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
  return `http://127.0.0.1:${server.address().port}`;
}
process.env.THETA_AGENT_API_URL = await listen(servers[0]);
process.env.THETA_MANUAL_LOCAL_API_URL = await listen(servers[1]);
process.env.HOSTNAME = '127.0.0.1';
process.env.THETA_LOCAL_AUTH_ENABLED = 'true';
process.env.NODE_ENV = 'production';
await load('frontend/server.js');
const base = `http://127.0.0.1:${process.env.PORT}`;
let healthy = false;
for (let attempt = 0; attempt < 180; attempt++) {
  try {
    const response = await fetch(base + '/api/v3/health', { headers: { 'x-theta-desktop-token': localToken }, signal: AbortSignal.timeout(2000) });
    if (response.ok) { healthy = true; break; }
  } catch {}
  await new Promise(resolve => setTimeout(resolve, 250));
}
if (!healthy) throw new Error('Desktop frontend did not become healthy');
process.parentPort?.postMessage({ type: 'ready', url: base, agentUrl: process.env.THETA_AGENT_API_URL, manualUrl: process.env.THETA_MANUAL_LOCAL_API_URL });
let closing = false;
process.parentPort?.on('message', ({ data }) => {
  if (data?.type === 'configure' && !closing) {
    const configurable = /^(THETA_INFERENCE_|OPENAI_|DEEPSEEK_|MINIMAX_|GLM_|EMBEDDING_|QWEN_|SBERT_)/;
    for (const key of Object.keys(process.env)) if (configurable.test(key)) delete process.env[key];
    for (const [key, value] of Object.entries(data.env)) if (configurable.test(key)) process.env[key] = value;
    process.parentPort.postMessage({ type: 'configured', requestId: data.requestId });
    return;
  }
  if (data !== 'shutdown' || closing) return;
  closing = true;
  for (const server of servers) { server.close(); server.closeAllConnections(); }
  // Give API shutdown hooks time to flush their SQLite state. Detached training
  // workers retain their own lifecycle and are never silently resubmitted.
  setTimeout(() => process.exit(0), 500);
});
