import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { z } from 'zod';
import { createConfiguredProvider } from '../../src/providers/configured-provider.js';
import { LocalProductTools } from '../../src/tools/local-tools.js';
import { observationText } from '../../src/conversation/conversation-agent.js';
import { productTools, toolDescriptor } from '../../src/tools/tool-catalog.js';
import { ProductSessionStore } from '../../src/memory/session-store.js';
import { ExecutionTrace, type ExecutionEvent } from '../../src/conversation/execution-events.js';

const requestSchema = z.object({ input: z.record(z.unknown()).default({}), userMessage: z.string().max(32000).default('') }).strict();
const hostTools = [
  { name: 'session_create', description: 'Create a persistent conversation session.', inputSchema: { type: 'object', properties: {}, additionalProperties: false } },
  { name: 'dataset_attach', description: 'Host imports an explicitly user-selected local file; returns an opaque managed reference.', inputSchema: { type: 'object', properties: { filePath: { type: 'string' } }, required: ['filePath'], additionalProperties: false } },
  { name: 'checkpoint_approve', description: 'Host only: approve an already presented checkpoint with the actual human approval and exact expectedContentHash. Never expose this tool to the research model.', inputSchema: { type: 'object', properties: { expectedContentHash: { type: 'string' } }, required: ['expectedContentHash'], additionalProperties: false } },
  { name: 'checkpoint_reject', description: 'Host only: reject the displayed action without executing it. Keep this outside the language model tool set.', inputSchema: { type: 'object', properties: { expectedContentHash: { type: 'string' } }, required: ['expectedContentHash'], additionalProperties: false } },
];

export const runToolsCli = async (args: string[], output: { write(text: string): void; writeError(text: string): void }): Promise<number> => {
  const requestId = randomUUID();
  const executionEvents: ExecutionEvent[] = [];
  let endCall: ReturnType<ExecutionTrace['start']> | undefined;
  const reply = (value: unknown): void => output.write(JSON.stringify({ protocol: 'theta.tools.v1', requestId, ...value as object, ...(executionEvents.length ? { executionEvents } : {}) }));
  if (args[0] === 'list') {
    reply({ ok: true, tools: [
      ...productTools.map((tool) => ({ ...toolDescriptor(tool), worker: tool.worker, effect: tool.effect, version: '1.0.0', hostOnly: false })),
      ...hostTools.map((tool) => ({ ...tool, worker: 'host', effect: 'write', version: '1.0.0', hostOnly: true })),
    ], requestSchema: { input: 'tool-specific JSON object', userMessage: 'verbatim user message when answering, revising or approving' } });
    return 0;
  }
  let store: ProductSessionStore | undefined;
  let lease: string | undefined;
  let sessionId: string | undefined;
  let timer: ReturnType<typeof setInterval> | undefined;
  try {
    if (args[0] !== 'call' || !args[1]) throw new Error('Usage: theta tools list | theta tools call <name> --session <id>; JSON request on stdin.');
    if (args.slice(2).some((arg, index, rest) => arg !== '--session' && rest[index - 1] !== '--session')) throw new Error('Only --session <id> is accepted. Tool input is JSON on stdin.');
    const index = args.indexOf('--session');
    sessionId = index >= 0 ? args[index + 1] : undefined;
    const name = args[1];
    if (process.stdin.isTTY) throw new Error('Provide a JSON request on stdin.');
    let raw = '';
    for await (const chunk of process.stdin) {
      raw += String(chunk);
      if (Buffer.byteLength(raw) > 1024 * 1024) throw new Error('Request exceeds 1 MiB.');
    }
    const request = requestSchema.parse(raw.trim() ? JSON.parse(raw) : {});
    const directory = path.resolve(process.env.THETA_AGENT_HOME ?? path.join(process.cwd(), '.theta_agent'));
    process.env.THETA_MEMORY_BACKEND ??= 'sqlite';
    store = new ProductSessionStore(directory);
    if (name === 'session_create') {
      z.object({}).strict().parse(request.input);
      const session = store.create(); reply({ ok: true, sessionId: session.id }); return 0;
    }
    if (!sessionId) throw new Error('--session is required. Call session_create first.');
    lease = store.acquire(sessionId);
    timer = setInterval(() => store!.renew(sessionId!, lease!), 30000);
    const session = store.get(sessionId);
    const tools = new LocalProductTools({ runtimeDb: path.join(directory, 'research.sqlite'), uploadDir: path.join(directory, 'uploads'), inferenceFactory: createConfiguredProvider });
    const save = (): void => store!.save(session, lease);
    const trace = new ExecutionTrace(event => { executionEvents.push(event); session.executionEvents = [...(session.executionEvents ?? []), event].slice(-1000); save(); });
    endCall = trace.start('tool', name);
    if (['research_answer', 'checkpoint_revise', 'checkpoint_approve', 'run_create', 'training_cancel'].includes(name) && !request.userMessage.trim()) throw new Error('This tool requires the actual userMessage.');
    let data: unknown;
    if (name === 'dataset_attach') {
      const input = z.object({ filePath: z.string().min(1) }).strict().parse(request.input);
      data = await tools.attach(input.filePath, session);
    } else if (name === 'checkpoint_approve' || name === 'checkpoint_reject') {
      const input = z.object({ expectedContentHash: z.string().min(1) }).strict().parse(request.input);
      if (input.expectedContentHash !== session.pendingConfirmation?.contentHash) throw new Error('Stale or missing checkpoint confirmation. Read checkpoint_review first.');
      data = name === 'checkpoint_reject' ? tools.deny(session) : await tools.approve(session, request.userMessage);
    } else data = await tools.execute(name, request.input, { session, userMessage: request.userMessage, save });
    endCall('completed');
    session.messages.push({ role: 'user', content: `外部工具调用 ${name}。用户指令：${request.userMessage}` }, { role: 'assistant', content: `工具执行证据：${observationText(data)}` });
    save();
    reply({ ok: true, sessionId, data: JSON.parse(observationText(data)), ...(session.pendingConfirmation ? { pendingConfirmation: session.pendingConfirmation } : {}) });
    return 0;
  } catch (error) { endCall?.('failed'); reply({ ok: false, sessionId, error: { code: 'TOOL_CALL_FAILED', message: error instanceof Error ? error.message : String(error), retryable: false } }); return 1; }
  finally { if (timer) clearInterval(timer); if (lease && sessionId) store?.release(sessionId, lease); store?.close(); }
};
