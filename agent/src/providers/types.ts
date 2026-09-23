/** Structural inference port, compatible with the former Hypha provider adapter.
 * No framework, terminal or compute runtime dependency is needed to start a chat. */
export interface PromptMessage {
  role: 'system' | 'developer' | 'user' | 'assistant' | 'tool' | 'context' | 'memory';
  content: string;
  name?: string;
  metadata?: Record<string, unknown>;
}
export interface InferenceRequest {
  runId: string; stepId: string; agentId?: string; modelAlias: string;
  input: unknown;
  tools?: Array<{ id: string; name: string; description?: string; inputSchema: Record<string, unknown> }>;
  options?: { temperature?: number; maxTokens?: number; extra?: Record<string, unknown> };
}
export interface InferenceResponse {
  id: string; output: unknown;
  usage?: { inputTokens?: number; outputTokens?: number; totalTokens?: number };
  metadata?: Record<string, unknown>;
}
export interface InferenceProvider {
  id: string;
  infer(request: InferenceRequest): Promise<InferenceResponse>;
}
