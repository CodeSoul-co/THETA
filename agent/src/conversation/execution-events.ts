import { randomUUID } from 'node:crypto';

/** Transport-neutral events: CLI today, SSE/WebSocket consumers later. No secrets/arguments. */
export interface ExecutionEvent {
  version: 1; eventId: string; turnId: string; callId: string;
  kind: 'inference' | 'tool' | 'provider'; name: string;
  status: 'started' | 'completed' | 'failed' | 'cancelled' | 'cached';
  startedAt: string; timestamp: string; durationMs: number;
  attempt?: number;
  detail?: string;
}
export class ExecutionTrace {
  readonly turnId = randomUUID();
  constructor(private readonly emit: (event: ExecutionEvent) => void) {}
  start(kind: ExecutionEvent['kind'], name: string, attempt?: number, detail?: string): (status: ExecutionEvent['status']) => void {
    const callId = randomUUID(); const started = Date.now(); const monotonic = performance.now();
    const send = (status: ExecutionEvent['status']): void => this.emit({ version: 1, eventId: randomUUID(), turnId: this.turnId, callId,
      kind, name, status, startedAt: new Date(started).toISOString(), timestamp: new Date().toISOString(),
      durationMs: Math.max(0, performance.now() - monotonic), ...(attempt ? { attempt } : {}), ...(detail ? { detail: detail.slice(0, 160) } : {}) });
    send('started'); let ended = false;
    return (status) => { if (!ended) { ended = true; send(status); } };
  }
}
