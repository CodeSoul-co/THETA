import type {
  InferenceProvider,
  InferenceRequest,
  InferenceResponse,
  PromptMessage,
} from './types.js';
import { openAiToolFunctionName } from './openai-tool-name.js';

const DEFAULT_BASE_URL = 'https://api.minimax.io/v1';
const DEFAULT_MODEL = 'MiniMax-M2.7';
const DEFAULT_TIMEOUT_MS = 60_000;
const MAX_TIMEOUT_MS = 180_000;
const DEFAULT_MAX_ATTEMPTS = 3;
const RETRY_BASE_DELAY_MS = 800;

export interface MiniMaxProviderConfig {
  apiKey: string;
  providerId?: string;
  providerName?: string;
  baseUrl?: string;
  model?: string;
  reasoningMode?: 'auto' | 'chat' | 'reasoning';
  reasoningEffort?: 'low' | 'medium' | 'high' | 'xhigh';
  maxTokens?: number;
  temperature?: number;
  timeoutMs?: number;
  maxAttempts?: number;
  fetchImpl?: typeof fetch;
}

interface MiniMaxProviderInput {
  messages: PromptMessage[];
  instructions?: string;
}

export class MiniMaxInferenceProvider implements InferenceProvider {
  readonly id: string;
  readonly model: string;
  private readonly providerName: string;
  private readonly apiKey: string;
  private readonly baseUrl: string;
  private readonly reasoningMode: 'auto' | 'chat' | 'reasoning';
  private readonly reasoningEffort: 'low' | 'medium' | 'high' | 'xhigh';
  private readonly defaultMaxTokens: number;
  private readonly defaultTemperature: number;
  private readonly timeoutMs: number;
  private readonly maxAttempts: number;
  private readonly fetchImpl: typeof fetch;

  constructor(config: MiniMaxProviderConfig) {
    this.id = config.providerId ?? 'minimax-openai-compatible';
    this.providerName = config.providerName?.trim() || this.id;
    this.apiKey = required(config.apiKey, `${this.providerName} API key`);
    this.baseUrl = normalizeBaseUrl(config.baseUrl ?? DEFAULT_BASE_URL);
    this.model = required(config.model ?? DEFAULT_MODEL, `${this.providerName} model`);
    this.reasoningMode = config.reasoningMode ?? 'auto';
    this.reasoningEffort = config.reasoningEffort ?? 'high';
    this.defaultMaxTokens = config.maxTokens ?? 4096;
    this.defaultTemperature = config.temperature ?? 0.2;
    this.timeoutMs = positiveInteger(
      config.timeoutMs ?? DEFAULT_TIMEOUT_MS,
      `${this.providerName} timeout`,
    );
    this.maxAttempts = boundedAttempts(
      config.maxAttempts ?? DEFAULT_MAX_ATTEMPTS,
    );
    this.fetchImpl = config.fetchImpl ?? fetch;
  }

  async infer(request: InferenceRequest): Promise<InferenceResponse> {
    const input = providerInput(request.input);
    try {
      const baseMessages = [
        ...(input.instructions ? [{ role: 'system', content: input.instructions }] : []),
        ...input.messages.map(apiMessage),
      ];
      const first = await this.request(request, baseMessages);
      let payload = await responseJson(first);
      let toolCalls: ReturnType<typeof responseToolCalls> = [];
      let protocolError: unknown;
      try { toolCalls = responseToolCalls(payload); } catch (error) { protocolError = error; }
      const choice = miniMaxToolChoice(request.options?.extra?.toolChoice);
      const allowed = new Set(request.tools?.map(tool => tool.name));
      const toolMarkup = () => /<[^>]*(?:DSML|tool_calls?|function_call)[^>]*>/iu.test(optionalResponseContent(payload) ?? '');
      const validCalls = () => choice !== 'none' && toolCalls.length > 0 && toolCalls.every(call => allowed.has(call.name)) &&
        (typeof choice !== 'object' || toolCalls.every(call => call.name === choice.function.name));
      if (toolMarkup() || protocolError || toolCalls.length > 0 && !validCalls() || request.tools?.length && (choice === 'required' || typeof choice === 'object') && !validCalls()) {
        const invalidContent = optionalResponseContent(payload);
        const repaired = await this.request(request, [
          ...baseMessages,
          ...(invalidContent ? [{ role: 'assistant', content: invalidContent }] : []),
          {
            role: 'system',
            content: choice === 'none' || !allowed.size
              ? 'Protocol error: tools are unavailable. Answer the user in plain language using the evidence already collected. State any remaining uncertainty. Do not emit tool calls, XML, DSML, or pretend to execute anything.'
              : `Protocol error: choose exactly one of the currently provided functions: ${[...allowed].join(', ')}. Historical functions that are absent from this list are unavailable. Return a native function call with a complete JSON object, never XML or DSML markup. Keep respond concise (at most 1200 characters). Do not continue an unfinished argument string.`,
          },
        ]);
        payload = await responseJson(repaired);
        toolCalls = responseToolCalls(payload);
        protocolError = undefined;
        const validText = toolCalls.length === 0 && (choice === 'auto' || choice === 'none') && request.options?.extra?.allowTextResponse === true && !!optionalResponseContent(payload)?.trim();
        if (toolMarkup() || !validCalls() && !validText) {
          throw new MiniMaxProviderError('non_json_response', `${this.providerName} did not produce a required native function call after one bounded protocol repair.`);
        }
      }
      if (protocolError && toolCalls.length === 0) throw protocolError;
      const output = toolCalls.length
        ? { kind: 'tool_calls', toolCalls }
        : request.options?.extra?.allowTextResponse === true
          ? { kind: 'tool_calls', toolCalls: [{ id: `response-${Date.now()}`, name: 'respond', arguments: { message: responseContent(payload).replace(/<think>[\s\S]*?<\/think>/giu, '').trim() } }] }
          : parseJsonObject(responseContent(payload));
      const usage = record(payload.usage);
      return {
        id:
          typeof payload.id === 'string'
            ? payload.id
            : `minimax-${Date.now()}`,
        output,
        usage: {
          inputTokens: optionalNumber(usage.prompt_tokens),
          outputTokens: optionalNumber(usage.completion_tokens),
          totalTokens: optionalNumber(usage.total_tokens),
        },
        metadata: {
          providerId: this.id,
          model: this.model,
          // Retain the model's public tool-call narrative, not hidden reasoning.
          ...(toolCalls.length ? {assistantText: (optionalResponseContent(payload) ?? '').replace(/<think>[\s\S]*?<\/think>/giu, '').trim()} : {}),
        },
      };
    } catch (error) {
      if (error instanceof MiniMaxProviderError) throw error;
      if (isAbortError(error)) {
        throw new MiniMaxProviderError(
          'timeout',
          `${this.providerName} request exceeded ${this.timeoutMs} ms.`,
        );
      }
      throw new MiniMaxProviderError(
        'network_failure',
        error instanceof Error ? error.message : String(error),
      );
    }
  }

  private async request(
    request: InferenceRequest,
    messages: Array<Record<string, unknown>>,
  ): Promise<Response> {
    let lastError: unknown;
    const signal = request.options?.extra?.signal instanceof AbortSignal ? request.options.extra.signal : undefined;
    const onAttempt = request.options?.extra?.onProviderAttempt as ((status: 'started' | 'completed' | 'failed' | 'cancelled', attempt: number) => void) | undefined;
    for (let attempt = 1; attempt <= this.maxAttempts; attempt += 1) {
      if (signal?.aborted) throw new MiniMaxProviderError('cancelled', 'Language model request cancelled.');
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), this.timeoutMs);
      let completed = false;
      onAttempt?.('started', attempt);
      try {
        const response = await this.fetchImpl(
        `${this.baseUrl}/chat/completions`,
        {
          method: 'POST',
          headers: {
            Authorization: `Bearer ${this.apiKey}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            model: this.model,
            messages,
            ...(request.tools?.length
              ? {
                  tools: request.tools.map((tool) => ({
                    type: 'function',
                    function: {
                      name: tool.name,
                      description: tool.description,
                      parameters:
                        this.id === 'deepseek'
                          ? deepSeekToolInputSchema(tool.inputSchema)
                          : tool.inputSchema,
                    },
                  })),
                  tool_choice: miniMaxToolChoice(
                    request.options?.extra?.toolChoice,
                  ),
                }
              : {}),
            ...this.completionOptions(request),
          }),
          signal: signal ? AbortSignal.any([controller.signal, signal]) : controller.signal,
        },
      );
        if (response.ok) {
          // Include body consumption in the attempt timeout and duration.
          const body = await response.text();
          completed = true;
          return new Response(body, { status: response.status, headers: response.headers });
        }
        if (!isRetryableHttpStatus(response.status) || attempt === this.maxAttempts) {
          const details = await providerErrorDetails(response);
          throw new MiniMaxProviderError(
            'provider_error',
            `${this.providerName} request failed with HTTP ${response.status}${details ? `: ${details}` : '.'}`,
          );
        }
        lastError = new Error(`HTTP ${response.status}`);
      } catch (error) {
        if (signal?.aborted) throw new MiniMaxProviderError('cancelled', 'Language model request cancelled.');
        if (error instanceof MiniMaxProviderError) throw error;
        lastError = error;
        if (attempt === this.maxAttempts) {
          if (isAbortError(error)) {
            throw new MiniMaxProviderError('timeout', `${this.providerName} request exceeded ${this.timeoutMs} ms after ${this.maxAttempts} attempts.`);
          }
          throw new MiniMaxProviderError(
            'network_failure',
            `${this.providerName} network request failed after ${this.maxAttempts} attempts: ${error instanceof Error ? error.message : String(error)}`,
          );
        }
      } finally {
        clearTimeout(timer);
        onAttempt?.(completed ? 'completed' : signal?.aborted ? 'cancelled' : 'failed', attempt);
      }
      await delay(RETRY_BASE_DELAY_MS * 2 ** (attempt - 1));
    }
    if (lastError instanceof MiniMaxProviderError) throw lastError;
    throw new MiniMaxProviderError(
      'network_failure',
      `${this.providerName} network request failed: ${lastError instanceof Error ? lastError.message : String(lastError)}`,
    );
  }

  private completionOptions(request: InferenceRequest): Record<string, unknown> {
    const maxTokens = Math.min(
      request.options?.maxTokens ?? this.defaultMaxTokens,
      8192,
    );
    const temperature =
      request.options?.temperature ?? this.defaultTemperature;

    if (this.id !== 'deepseek') {
      return {
        temperature,
        ...(['gpt', 'openai'].includes(this.id) ? { max_completion_tokens: maxTokens } : { max_tokens: maxTokens }),
      };
    }

    const supportsThinking =
      this.reasoningMode === 'reasoning' && !request.tools?.length && request.options?.extra?.allowTextResponse !== true;
    return {
      max_tokens: maxTokens,
      ...(supportsThinking
        ? {
            thinking: { type: 'enabled' },
            reasoning_effort: deepSeekReasoningEffort(this.reasoningEffort),
          }
        : this.reasoningMode === 'chat' || request.tools?.length || request.options?.extra?.allowTextResponse === true
          ? { thinking: { type: 'disabled' }, temperature }
          : { temperature }),
    };
  }
}

const deepSeekReasoningEffort = (
  effort: 'low' | 'medium' | 'high' | 'xhigh',
): 'low' | 'high' | 'max' => {
  if (effort === 'low') return 'low';
  if (effort === 'xhigh') return 'max';
  return 'high';
};

const deepSeekToolInputSchema = (
  schema: unknown,
): Record<string, unknown> => {
  if (!schema || typeof schema !== 'object' || Array.isArray(schema)) {
    return {
      type: 'object',
      properties: {},
      additionalProperties: false,
    };
  }
  return {
    ...(schema as Record<string, unknown>),
    type: 'object',
  };
};

const providerErrorDetails = async (
  response: Response,
): Promise<string | undefined> => {
  try {
    const raw = (await response.text()).replace(/\s+/g, ' ').trim();
    if (!raw) return undefined;
    const payload = JSON.parse(raw) as {
      error?: { message?: unknown; code?: unknown };
      message?: unknown;
    };
    const message =
      typeof payload.error?.message === 'string'
        ? payload.error.message
        : typeof payload.message === 'string'
          ? payload.message
          : raw;
    const code =
      typeof payload.error?.code === 'string' ||
      typeof payload.error?.code === 'number'
        ? ` (${payload.error.code})`
        : '';
    return `${message}${code}`.slice(0, 300);
  } catch {
    return undefined;
  }
};

export class MiniMaxProviderError extends Error {
  constructor(
    readonly code:
      | 'network_failure'
      | 'timeout'
      | 'provider_error'
      | 'cancelled'
      | 'non_json_response',
    message: string,
  ) {
    super(message);
  }
}

export const isMiniMaxConfigured = (): boolean =>
  Boolean(process.env.MINIMAX_API_KEY?.trim());

export const createMiniMaxProviderFromEnv = (
  overrides: { timeoutMs?: number } = {},
):
  | MiniMaxInferenceProvider
  | undefined => {
  const apiKey = process.env.MINIMAX_API_KEY?.trim();
  if (!apiKey) return undefined;
  return new MiniMaxInferenceProvider({
    apiKey,
    baseUrl: process.env.MINIMAX_API_BASE,
    model: process.env.MINIMAX_MODEL,
    timeoutMs:
      overrides.timeoutMs ??
      environmentInteger(process.env.MINIMAX_TIMEOUT_MS, DEFAULT_TIMEOUT_MS),
  });
};

const providerInput = (value: unknown): MiniMaxProviderInput => {
  if (!value || typeof value !== 'object' || !('messages' in value)) {
    throw new MiniMaxProviderError(
      'provider_error',
      'MiniMax inference input must contain messages.',
    );
  }
  const messages = (value as { messages?: unknown }).messages;
  if (
    !Array.isArray(messages) ||
    messages.length === 0 ||
    !messages.every(
      (message) =>
        Boolean(message) &&
        typeof message === 'object' &&
        typeof (message as PromptMessage).role === 'string' &&
        typeof (message as PromptMessage).content === 'string',
    )
  ) {
    throw new MiniMaxProviderError(
      'provider_error',
      'MiniMax inference messages are invalid.',
    );
  }
  const candidate = value as unknown as { instructions?: unknown };
  const instructions = typeof candidate.instructions === 'string'
    ? candidate.instructions.trim()
    : '';
  return { messages: messages as PromptMessage[], ...(instructions ? { instructions } : {}) };
};

const apiRole = (
  role: PromptMessage['role'],
): 'system' | 'user' | 'assistant' | 'tool' =>
  role === 'system' || role === 'assistant' || role === 'tool' ? role : 'user';

const apiMessage = (message: PromptMessage): Record<string, unknown> => {
  const metadata = record(message.metadata);
  const toolCalls = Array.isArray(metadata.toolCalls) ? metadata.toolCalls : [];
  const embeddedCall = message.role === 'assistant' ? embeddedToolCall(message.content) : undefined;
  const directToolCallId = (message as PromptMessage & { toolCallId?: unknown }).toolCallId;
  return {
    role: apiRole(message.role),
    content: embeddedCall ? null : message.content,
    ...(message.name ? { name: openAiToolFunctionName(message.name) } : {}),
    ...(message.role === 'tool' && (typeof directToolCallId === 'string' || typeof metadata.toolCallId === 'string')
      ? { tool_call_id: typeof directToolCallId === 'string' ? directToolCallId : metadata.toolCallId }
      : {}),
    ...(embeddedCall
      ? { tool_calls: [embeddedCall] }
      : message.role === 'assistant' && toolCalls.length > 0
      ? {
          tool_calls: toolCalls.map((rawCall) => {
            const call = record(rawCall);
            return {
              id: String(call.id ?? ''),
              type: 'function',
              function: {
                name: openAiToolFunctionName(String(call.name ?? '')),
                arguments: JSON.stringify(record(call.arguments)),
              },
            };
          }),
        }
      : {}),
  };
};

const embeddedToolCall = (content: string): Record<string, unknown> | undefined => {
  try {
    const value = JSON.parse(content) as Record<string, unknown>;
    if (value.type !== 'tool_call' || typeof value.tool !== 'string') return undefined;
    return {
      id: typeof value.id === 'string' && value.id ? value.id : `tool-call-${Date.now()}`,
      type: 'function',
      function: {
        name: openAiToolFunctionName(value.tool),
        arguments: JSON.stringify(record(value.input)),
      },
    };
  } catch {
    return undefined;
  }
};

const miniMaxToolChoice = (value: unknown): 'auto' | 'none' | 'required' | { type: 'function'; function: { name: string } } => {
  const candidate = record(value);
  const name = record(candidate.function).name;
  if (candidate.type === 'function' && typeof name === 'string') return { type: 'function', function: { name } };
  return value === 'none' ? 'none' : value === 'required' ? 'required' : 'auto';
};

const responseJson = async (
  response: Response,
): Promise<Record<string, unknown>> => {
  try {
    const value = await response.json();
    if (!value || typeof value !== 'object' || Array.isArray(value)) {
      throw new Error('Response is not an object.');
    }
    return value as Record<string, unknown>;
  } catch (error) {
    throw new MiniMaxProviderError(
      'non_json_response',
      error instanceof Error ? error.message : String(error),
    );
  }
};

const responseContent = (payload: Record<string, unknown>): string => {
  const choices = Array.isArray(payload.choices) ? payload.choices : [];
  const first = record(choices[0]);
  const message = record(first.message);
  if (typeof message.content !== 'string' || !message.content.trim()) {
    throw new MiniMaxProviderError(
      'non_json_response',
      'MiniMax response did not contain message content.',
    );
  }
  return message.content;
};

const responseToolCalls = (
  payload: Record<string, unknown>,
): Array<{ id: string; name: string; arguments: Record<string, unknown> }> => {
  const choices = Array.isArray(payload.choices) ? payload.choices : [];
  const message = record(record(choices[0]).message);
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  return calls.map((rawCall, index) => {
    const call = record(rawCall);
    const fn = record(call.function);
    const name = typeof fn.name === 'string' ? fn.name.trim() : '';
    if (!name) {
      throw new MiniMaxProviderError(
        'non_json_response',
        'MiniMax tool call did not contain a function name.',
      );
    }
    let args: unknown = fn.arguments;
    if (typeof args === 'string') {
      try {
        args = args.trim() ? parseJsonWithConservativeRepair(args) : {};
      } catch (error) {
        throw new MiniMaxProviderError(
          'non_json_response',
          `MiniMax tool arguments were not valid JSON: ${error instanceof Error ? error.message : String(error)}`,
        );
      }
    }
    if (!args || typeof args !== 'object' || Array.isArray(args)) {
      throw new MiniMaxProviderError(
        'non_json_response',
        'MiniMax tool arguments must be a JSON object.',
      );
    }
    return {
      id: typeof call.id === 'string' ? call.id : `tool-call-${index + 1}`,
      name,
      arguments: args as Record<string, unknown>,
    };
  });
};

const parseJsonObject = (content: string): Record<string, unknown> => {
  const withoutThinking = content
    .replace(/<think>[\s\S]*?<\/think>/giu, '')
    .replace(/^```(?:json)?\s*/iu, '')
    .replace(/\s*```$/u, '')
    .trim();
  const start = withoutThinking.indexOf('{');
  const end = withoutThinking.lastIndexOf('}');
  if (start === -1 || end <= start) {
    throw new MiniMaxProviderError(
      'non_json_response',
      'MiniMax response did not contain a JSON object.',
    );
  }
  const candidate = withoutThinking.slice(start, end + 1);
  try {
    const value = parseJsonWithConservativeRepair(candidate);
    if (!value || typeof value !== 'object' || Array.isArray(value)) {
      throw new Error('Parsed response is not an object.');
    }
    return value as Record<string, unknown>;
  } catch (error) {
    throw new MiniMaxProviderError(
      'non_json_response',
      error instanceof Error ? error.message : String(error),
    );
  }
};

const optionalResponseContent = (payload: Record<string, unknown>): string | undefined => {
  const choices = Array.isArray(payload.choices) ? payload.choices : [];
  const content = record(record(choices[0]).message).content;
  return typeof content === 'string' && content.trim() ? content.trim() : undefined;
};

/**
 * MiniMax occasionally returns otherwise valid JSON with a trailing comma or
 * full-width structural punctuation.  Repair only punctuation outside quoted
 * strings; never attempt to invent a missing field or value.
 */
const parseJsonWithConservativeRepair = (value: string): unknown => {
  try {
    return JSON.parse(value);
  } catch {
    const normalized = normalizeJsonPunctuation(value)
      .replace(/,\s*([}\]])/gu, '$1');
    try {
      return JSON.parse(normalized);
    } catch {
      const firstObject = firstCompleteJsonObject(normalized);
      if (firstObject === undefined) throw new Error('No complete JSON object was found.');
      return JSON.parse(firstObject);
    }
  }
};

const firstCompleteJsonObject = (value: string): string | undefined => {
  const start = value.indexOf('{');
  if (start < 0) return undefined;
  let depth = 0;
  let quoted = false;
  let escaped = false;
  for (let index = start; index < value.length; index += 1) {
    const character = value[index];
    if (quoted) {
      if (escaped) escaped = false;
      else if (character === '\\') escaped = true;
      else if (character === '"') quoted = false;
      continue;
    }
    if (character === '"') quoted = true;
    else if (character === '{') depth += 1;
    else if (character === '}') {
      depth -= 1;
      if (depth === 0) return value.slice(start, index + 1);
      if (depth < 0) return undefined;
    }
  }
  return undefined;
};

const normalizeJsonPunctuation = (value: string): string => {
  let result = '';
  let quoted = false;
  let escaped = false;
  for (const character of value) {
    if (quoted) {
      result += character;
      if (escaped) escaped = false;
      else if (character === '\\') escaped = true;
      else if (character === '"') quoted = false;
      continue;
    }
    if (character === '"') {
      quoted = true;
      result += character;
      continue;
    }
    result += character === '，' ? ',' : character === '：' ? ':' : character;
  }
  return result;
};

const normalizeBaseUrl = (value: string): string => {
  const url = new URL(required(value, 'MiniMax API base URL'));
  if (url.protocol !== 'https:' && !(url.protocol === 'http:' && ['localhost', '127.0.0.1', '[::1]'].includes(url.hostname))) {
    throw new Error('API base URL must use HTTPS, or HTTP on a loopback address for a local model.');
  }
  return url.toString().replace(/\/+$/u, '');
};

const required = (value: string, name: string): string => {
  const normalized = value.trim();
  if (!normalized) throw new Error(`${name} is required.`);
  return normalized;
};

const positiveInteger = (value: number, name: string): number => {
  if (!Number.isInteger(value) || value <= 0 || value > MAX_TIMEOUT_MS) {
    throw new Error(`${name} must be an integer between 1 and ${MAX_TIMEOUT_MS}.`);
  }
  return value;
};

const environmentInteger = (
  value: string | undefined,
  fallback: number,
): number => {
  if (!value?.trim()) return fallback;
  return positiveInteger(Number(value), 'MINIMAX_TIMEOUT_MS');
};

const boundedAttempts = (value: number): number => {
  if (!Number.isInteger(value) || value < 1 || value > 5) {
    throw new Error('MiniMax max attempts must be an integer between 1 and 5.');
  }
  return value;
};

const isRetryableHttpStatus = (status: number): boolean =>
  status === 408 || status === 425 || status === 429 || status >= 500;

const delay = async (milliseconds: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, milliseconds));

const record = (value: unknown): Record<string, unknown> =>
  value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};

const optionalNumber = (value: unknown): number | undefined =>
  typeof value === 'number' && Number.isFinite(value) ? value : undefined;

const isAbortError = (error: unknown): boolean =>
  error instanceof Error && error.name === 'AbortError';
