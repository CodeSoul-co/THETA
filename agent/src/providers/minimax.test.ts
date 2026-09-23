import assert from 'node:assert/strict';
import test from 'node:test';
import type { InferenceRequest } from './types.js';
import { MiniMaxInferenceProvider } from './minimax.js';

test('public tool-call narrative is retained without exposing hidden reasoning',async()=>{
  const provider=new MiniMaxInferenceProvider({apiKey:'fixture',fetchImpl:async()=>Response.json({choices:[{message:{content:'<think>private</think>Check actual evidence.',reasoning_content:'hidden',tool_calls:[{id:'c',type:'function',function:{name:'inspect',arguments:'{}'}}]}}]})});
  const result=await provider.infer({runId:'r',stepId:'s',modelAlias:'m',input:{messages:[{role:'user',content:'inspect'}]},tools:[{id:'inspect',name:'inspect',inputSchema:{type:'object'}}]});
  assert.equal(result.metadata?.assistantText,'Check actual evidence.');assert.doesNotMatch(JSON.stringify(result),/private|hidden/);
});

test('HTTP attempts and retries each emit their own completed or failed event', async () => {
  const events: string[] = []; let calls = 0;
  const provider = new MiniMaxInferenceProvider({ apiKey: 'fixture', maxAttempts: 2, fetchImpl: async () => {
    if (++calls === 1) return new Response('', { status: 503 });
    return Response.json({ choices: [{ message: { content: '{"message":"ready"}' } }] });
  } });
  await provider.infer({ runId: 'fixture', stepId: 'fixture', modelAlias: 'fixture', input: { messages: [{ role: 'user', content: 'fixture' }] },
    options: { extra: { onProviderAttempt: (status: string, attempt: number) => events.push(`${attempt}:${status}`) } } });
  assert.deepEqual(events, ['1:started', '1:failed', '2:started', '2:completed']);
});

test('local endpoints are supported and cancelled inference never retries or performs a request', async () => {
  let calls = 0;
  const provider = new MiniMaxInferenceProvider({ apiKey: 'fixture', model: 'fixture', baseUrl: 'http://127.0.0.1:11434/v1', fetchImpl: async () => { calls++; throw new Error('must not fetch'); } });
  const controller = new AbortController(); controller.abort();
  await assert.rejects(provider.infer({ runId: 'test', stepId: 'step', modelAlias: 'test', input: { messages: [{ role: 'user', content: 'test' }] }, options: { extra: { signal: controller.signal } } }), /cancelled/);
  assert.equal(calls, 0);
  assert.throws(() => new MiniMaxInferenceProvider({ apiKey: 'fixture', baseUrl: 'http://external.example/v1' }), /HTTPS/);
});

test('DeepSeek requests always send object-rooted function schemas', async () => {
  let requestBody: Record<string, unknown> | undefined;
  const provider = new MiniMaxInferenceProvider({
    apiKey: 'test-key',
    providerId: 'deepseek',
    model: 'deepseek-v4-flash',
    baseUrl: 'https://api.deepseek.com',
    reasoningMode: 'reasoning',
    reasoningEffort: 'high',
    maxAttempts: 1,
    fetchImpl: async (_input, init) => {
      requestBody = JSON.parse(String(init?.body)) as Record<string, unknown>;
      return new Response(JSON.stringify({
        id: 'test-response',
        choices: [{
          message: {
            tool_calls: [{
              id: 'call-1',
              type: 'function',
              function: { name: 'theta_finish_phase', arguments: '{}' },
            }],
          },
        }],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    },
  });

  await provider.infer({
    runId: 'run-test',
    stepId: 'step-test',
    modelAlias: 'deepseek-v4-flash',
    input: { messages: [{ role: 'user', content: 'Finish the phase.' }] },
    tools: [{
      id: 'theta_finish_phase',
      name: 'theta_finish_phase',
      description: 'Finish the current phase.',
      inputSchema: {
        oneOf: [
          {
            type: 'object',
            required: ['kind'],
            properties: { kind: { const: 'phase_completion_proposed' } },
          },
        ],
      },
    }],
    options: { extra: { toolChoice: 'required' } },
  } as unknown as InferenceRequest);

  assert.ok(requestBody);
  const tools = requestBody.tools as Array<{
    function: { parameters: Record<string, unknown> };
  }>;
  assert.equal(tools[0]?.function.parameters.type, 'object');
  assert.ok(Array.isArray(tools[0]?.function.parameters.oneOf));
  assert.equal(requestBody.max_tokens, 4096);
  assert.equal('max_completion_tokens' in requestBody, false);
  assert.deepEqual(requestBody.thinking, { type: 'disabled' });
  assert.equal('reasoning_effort' in requestBody, false);
  assert.equal(requestBody.tool_choice, 'required');
});

test('ordinary conversational text ends a turn without forcing a tool or protocol repair', async () => {
  let requests = 0;
  const provider = new MiniMaxInferenceProvider({ apiKey: 'fixture', fetchImpl: async (_url, init) => {
    requests++; const body = JSON.parse(String(init?.body)); assert.equal(body.tool_choice, 'auto');
    return Response.json({ choices: [{ message: { content: '可以先讨论你的业务目标。' } }] });
  } });
  const result = await provider.infer({ runId: 'r', stepId: 's', modelAlias: 'm', input: { messages: [{ role: 'user', content: '你好' }] },
    tools: [{ id: 'respond', name: 'respond', description: '', inputSchema: { type: 'object' } }], options: { extra: { toolChoice: 'auto', allowTextResponse: true } } });
  assert.equal(requests, 1);
  assert.deepEqual((result.output as { toolCalls: Array<{ name: string; arguments: unknown }> }).toolCalls[0].arguments, { message: '可以先讨论你的业务目标。' });
});

test('malformed or obsolete function calls are repaired once and never silently executed', async () => {
  for (const first of [{ name: 'respond', arguments: '{"message":"unfinished' }, { name: 'obsolete_tool', arguments: '{}' }]) {
    let requests = 0;
    const provider = new MiniMaxInferenceProvider({ apiKey: 'fixture', fetchImpl: async () => {
      const fn = ++requests === 1 ? first : { name: 'respond', arguments: '{"message":"已暂停。"}' };
      return Response.json({ choices: [{ message: { tool_calls: [{ id: 'fixture', function: fn }] } }] });
    } });
    const result = await provider.infer({ runId: 'r', stepId: 's', modelAlias: 'm', input: { messages: [{ role: 'user', content: '继续' }] },
      tools: [{ id: 'respond', name: 'respond', description: '', inputSchema: { type: 'object' } }], options: { extra: { toolChoice: 'auto', allowTextResponse: true } } });
    assert.equal(requests, 2); assert.equal((result.output as { toolCalls: Array<{ name: string }> }).toolCalls[0].name, 'respond');
  }
});


test('protocol repair accepts a plain-language configuration blocker', async () => {
  let requests = 0;
  const provider = new MiniMaxInferenceProvider({ apiKey: 'fixture', fetchImpl: async () => {
    return Response.json({ choices: [{ message: ++requests === 1
      ? { tool_calls: [{ id: 'fixture', function: { name: 'obsolete_tool', arguments: '{}' } }] }
      : { content: '缺少配置，无法生成确认卡。' } }] });
  } });
  const result = await provider.infer({ runId: 'r', stepId: 's', modelAlias: 'm', input: { messages: [{ role: 'user', content: '继续' }] },
    tools: [{ id: 'respond', name: 'respond', description: '', inputSchema: { type: 'object' } }], options: { extra: { toolChoice: 'auto', allowTextResponse: true } } });
  assert.equal(requests, 2);
  assert.deepEqual((result.output as { toolCalls: Array<{ arguments: unknown }> }).toolCalls[0].arguments, { message: '缺少配置，无法生成确认卡。' });
});

test('final summaries repair leaked tool markup once, never execute it or display repeated markup', async () => {
  for (const repaired of [true, false]) {
    let requests = 0;
    const markup = '<｜｜DSML｜｜tool_calls><｜｜DSML｜｜invoke name="models_inspect">';
    const provider = new MiniMaxInferenceProvider({ apiKey: 'fixture', fetchImpl: async (_url, init) => {
      const body = JSON.parse(String(init?.body)); assert.equal(body.tools, undefined);
      requests++;
      if (requests === 2) assert.match(body.messages.at(-1).content, /tools are unavailable/);
      return Response.json({ choices: [{ message: { content: repaired && requests === 2 ? '已核实四个模型均已注册，实际训练仍需检查依赖。' : markup } }] });
    } });
    const request: InferenceRequest = { runId: 'r', stepId: 's', modelAlias: 'm', input: { messages: [{ role: 'user', content: '核实支持范围' }] }, tools: [], options: { extra: { toolChoice: 'none', allowTextResponse: true } } };
    if (repaired) {
      const result = await provider.infer(request);
      assert.deepEqual((result.output as { toolCalls: Array<{ name: string; arguments: unknown }> }).toolCalls, [{ id: (result.output as any).toolCalls[0].id, name: 'respond', arguments: { message: '已核实四个模型均已注册，实际训练仍需检查依赖。' } }]);
    } else await assert.rejects(provider.infer(request), /bounded protocol repair/);
    assert.equal(requests, 2);
  }
});
