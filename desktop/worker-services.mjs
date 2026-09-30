import { pathToFileURL } from 'node:url';
import path from 'node:path';
const { createWorkerServer } = await import(pathToFileURL(path.join(process.env.THETA_PROJECT_ROOT, 'agent/dist/web/worker-server.js')).href);
const server = createWorkerServer(process.env.THETA_WORKER_API_TOKEN);
await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
process.parentPort.postMessage({ type: 'ready', url: `http://127.0.0.1:${server.address().port}` });
process.parentPort.on('message', ({ data }) => {
  if (data?.type === 'configure') {
    const configurable = /^(THETA_INFERENCE_|OPENAI_|DEEPSEEK_|MINIMAX_|GLM_|EMBEDDING_|QWEN_|SBERT_)/;
    for (const key of Object.keys(process.env)) if (configurable.test(key)) delete process.env[key];
    for (const [key, value] of Object.entries(data.env)) if (configurable.test(key)) process.env[key] = value;
    process.parentPort.postMessage({ type: 'configured', requestId: data.requestId });
  }
  if (data === 'shutdown') { server.close(); server.closeAllConnections(); setTimeout(() => process.exit(0), 500); }
});
