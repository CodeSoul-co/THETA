import assert from 'node:assert/strict';
import test from 'node:test';
import type { InferenceProvider, InferenceRequest } from '../providers/types.js';
import { ConversationAgent, boundedHistory, researchHistory } from './conversation-agent.js';
import type { ProductSession } from '../memory/session-store.js';

const session = (): ProductSession => ({ id: 'chat-test', title: '新的研究对话', messages: [], datasetRefs: [], updatedAt: '' });
const call = (name: string, args: Record<string, unknown> = {}) => ({ kind: 'tool_calls', toolCalls: [{ id: `call-${name}`, name, arguments: args }] });
const provider = (outputs: unknown[], inspect?: (request: InferenceRequest) => void): InferenceProvider => ({ id: 'fixture', infer: async (request) => { inspect?.(request); assert.ok(outputs.length, 'Unexpected model call'); return { id: 'fixture', output: outputs.shift() }; } });

test('host unbounded mode removes cumulative limits while preserving external cancellation',async()=>{
  const current=session();let executed=0;
  const outputs=[...Array.from({length:15},(_,i)=>call('calculate',{step:i})),call('respond',{message:'Finished'})];
  const agent=new ConversationAgent({unboundedResearch:true,maxRounds:1,maxToolCalls:1,timeoutMs:1,
    inference:provider(outputs),tools:{execute:async()=>{executed++;await new Promise(resolve=>setTimeout(resolve,2));return {}; }},save(){}});
  assert.equal(await agent.turn(current,'Analyze'),'Finished');assert.equal(executed,15);
  const controller=new AbortController();controller.abort();
  await agent.turn(session(),'Analyze',controller.signal);
  assert.equal(executed,15);
});

test('research checkpoint context retains the current request and latest pair without replaying old loops',()=>{
  const messages:ProductSession['messages']=[{role:'user',content:'Authoritative current request'},
    {role:'assistant',content:'obsolete repeated exploration'},
    {role:'assistant',content:'',metadata:{toolCalls:[{id:'note',name:'analysis_checkpoint',arguments:{}}]}},
    {role:'tool',content:'{"ok":true,"data":{"saved":true}}',metadata:{toolCallId:'note',analysisScope:'study'}},
    {role:'assistant',content:'',metadata:{toolCalls:[{id:'new',name:'compute',arguments:{}}]}},
    {role:'tool',content:'actual evidence',metadata:{toolCallId:'new'}}];
  const view=researchHistory(messages,'study');
  assert.equal(view[0].content,'Authoritative current request');assert.equal(view.at(-1)?.content,'actual evidence');
  assert.doesNotMatch(JSON.stringify(view),/obsolete repeated exploration/);
  assert.match(JSON.stringify(researchHistory(messages,'other')),/obsolete repeated exploration/);
  messages[3].content='{"ok":false,"data":{"saved":false}}';
  assert.match(JSON.stringify(researchHistory(messages,'study')),/obsolete repeated exploration/);
});

test('research mode checkpoints long work without granting execution or losing the next action', async()=>{
  let n=0;const current=session();const outputs=[...Array.from({length:8},(_,i)=>call('calculate',{step:i})),call('analysis_checkpoint',{}),call('calculate',{step:9}),call('respond',{message:'Delivered'})];
  const agent=new ConversationAgent({mode:'research',maxRounds:12,maxToolCalls:16,inference:provider(outputs,request=>{
    const input=request.input as {instructions:string};assert.match(input.instructions,/autonomous research mode/);
    assert.match(input.instructions,/never approve it yourself/);
    if (++n===9) assert.deepEqual(request.tools?.map(t=>t.name),['analysis_checkpoint','respond']);
    if(n===10) assert.ok(request.tools!.length>2);
  }),tools:{execute:async()=>({completed:true})},save(){}});
  assert.equal(await agent.turn(current,'Complete the research'),'Delivered');
});

test('research delivery checks actual receipts rather than accepting a promise',async()=>{
  const current=session();let delivered=false;
  const agent=new ConversationAgent({mode:'research',maxRounds:4,maxToolCalls:4,inference:provider([
    call('respond',{message:'I will save it next'}),call('analysis_deliver'),call('respond',{message:'Files archived'})
  ]),tools:{analysisProgress:()=>({plan:{requiredFiles:['report']},workspace:{delivered}}),execute:async()=>{delivered=true;return {delivered:true};}},save(){}});
  assert.equal(await agent.turn(current,'Complete the research'),'Files archived');
  assert.ok(current.messages.some(m=>m.content.includes('no verified archived deliverable')));
});

test('delivery recovery is bounded and genuine blockers can still be reported',async()=>{
  const current=session();const agent=new ConversationAgent({mode:'research',maxRounds:4,maxToolCalls:4,inference:provider([
    call('respond',{message:'Missing capability'}),call('respond',{message:'Still missing'}),call('respond',{message:'Actual blocker remains'})
  ]),tools:{analysisProgress:()=>({plan:{},workspace:{delivered:false}}),execute:async()=>null},save(){}});
  assert.equal(await agent.turn(current,'Research'),'Actual blocker remains');
});

test('disabled host tools are not advertised and report storage excludes decoration',async()=>{
  const current=session();let saved='';
  const agent=new ConversationAgent({inference:provider([call('respond',{message:'Original interpretation'})],request=>{
    assert.ok(!request.tools?.some(tool=>tool.name.startsWith('analysis_')));
  }),tools:{toolAvailable:name=>!name.startsWith('analysis_'),execute:async()=>null,recordUnderstanding:(_s,answer)=>{saved=answer;},decorateAnswer:(_s,answer)=>answer+' many links'},save(){}});
  assert.equal(await agent.turn(current,'Explain'),'Original interpretation many links');
  assert.equal(saved,'Original interpretation');
});

test('interleaved identical computations are detected without unlimited cycling', async()=>{
  const current=session();let a=0;const agent=new ConversationAgent({maxRounds:8,maxToolCalls:8,inference:provider([
    call('calculate',{operation:'a'}),call('calculate',{operation:'b'}),call('calculate',{operation:'a'}),call('calculate',{operation:'b'}),call('calculate',{operation:'a'}),call('respond',{message:'Blocked loop'})
  ]),tools:{execute:async(_name,args)=>{if((args as {operation:string}).operation==='a')a++;return {value:1};}},save(){}});
  await agent.turn(current,'Research');assert.equal(a,2);
});

test('generic repeated computation requires a notebook checkpoint then a different action', async () => {
  const current = session(); const executed: string[] = []; let inferenceCalls = 0;
  const agent = new ConversationAgent({maxRounds:8, maxToolCalls:10, inference:provider([
    call('external_compute',{code:'inspect'}), call('external_compute',{code:'inspect'}), call('external_compute',{code:'inspect'}),
    call('analysis_checkpoint',{}), call('external_compute',{code:'analyze'}), call('respond',{message:'Completed actual analysis'}),
  ], request => {
    if (++inferenceCalls === 4) assert.deepEqual(request.tools?.map(t=>t.name),['analysis_checkpoint','respond']);
  }), tools:{execute:async(name,args)=>{executed.push(name+JSON.stringify(args));return {value:1};}},save(){}});
  assert.match(await agent.turn(current,'Compute and deliver an analysis'),/Completed/);
  assert.equal(executed.filter(v=>v.includes('inspect')).length,2);
  assert.ok(executed.some(v=>v.includes('analyze')));
  assert.ok(current.messages.some(m=>m.content.includes('Repeated identical action stopped')));
});

test('live job polling remains available even when arguments are identical', async () => {
  const current=session();let reads=0;
  const agent=new ConversationAgent({inference:provider([call('run_status'),call('run_status'),call('run_status'),call('respond',{message:'Finished'})]),tools:{execute:async()=>({percent:++reads})},save(){}});
  await agent.turn(current,'Check running computation');assert.equal(reads,3);
});

test('notebook is scoped to the selected study and is explicitly unverified', async () => {
  const current=session();current.runId='new'; current.analysisNotebooks={old:{objective:'old secret claim',candidates:[],completed:[],openQuestions:[],artifacts:[],nextAction:'old'}};
  const agent=new ConversationAgent({inference:provider([call('respond',{message:'New study'})],request=>{
    const state=JSON.parse((request.input as {messages:Array<{content:string}>}).messages.at(-1)!.content);
    assert.equal(state.analysisNotebook.value,null); assert.match(state.analysisNotebook.provenance,/not verified/);
  }),tools:{execute:async()=>null},save(){}});await agent.turn(current,'New study');
});

test('public narrative survives alongside a tool call, while decorations stay out of inference history', async () => {
  const current=session();let n=0;
  const agent=new ConversationAgent({inference:{id:'fixture',infer:async()=>({id:'fixture',output:++n===1?call('dataset_read',{operation:'overview'}):call('respond',{message:'Actual conclusion'}),metadata:{assistantText:'Check the corpus before interpreting it.'}})},
    tools:{execute:async()=>({rows:5}),decorateAnswer:(_s,m)=>m+' artifact-link'.repeat(20000)},save(){}});
  await agent.turn(current,'Research');
  assert.equal(current.messages[1].content,'Check the corpus before interpreting it.');
  assert.equal(boundedHistory(current.messages).at(-1)!.content,'Actual conclusion');
  assert.ok(current.messages.at(-1)!.content.length>100000);
});

test('a long user receipt cannot evict the latest complete computation pair', () => {
  const messages: ProductSession['messages']=[{role:'user',content:'receipt'.repeat(3500)},
    {role:'assistant',content:'',metadata:{toolCalls:[{id:'latest',name:'calculate',arguments:{}}]}},
    {role:'tool',content:'observed'.repeat(2000),metadata:{toolCallId:'latest'}}];
  const history=boundedHistory(messages);
  assert.equal(history[0].content,messages[0].content);assert.equal(history.at(-1)?.metadata?.toolCallId,'latest');
  assert.equal(history.at(-2)?.role,'assistant');
});

test('completion prose asking for consent is repaired into a real report card without training', async () => {
  const current = session(); const executed: string[] = [];
  const agent = new ConversationAgent({ inference: provider([
    call('respond', { message: '训练已完成。我先申请结果读取授权，可以吗？' }),
    call('results_read', { view: 'report' }),
  ]), tools: { execute: async (name) => { executed.push(name); return { needsUser: true, summary: '报告确认卡' }; } }, save() {} });
  assert.equal(await agent.turn(current, '主机监控事件：任务已完成', undefined, { userIntent: '' }), '报告确认卡');
  assert.deepEqual(executed, ['results_read']);
});

test('web completion asking whether to request approval produces a real card', async () => {
  const current = session(); const executed: string[] = [];
  const agent = new ConversationAgent({ inference: provider([
    call('respond', { message: '训练已完成。是否现在申请这次结果读取确认？' }),
    call('results_read', { view: 'report' }),
  ]), tools: { execute: async (name) => { executed.push(name); return { needsUser: true, summary: '报告确认卡' }; } }, save() {} });
  assert.equal(await agent.turn(current, '主机监控事件：任务已完成', undefined, { userIntent: '' }), '报告确认卡');
  assert.deepEqual(executed, ['results_read']);
});

for (const mode of ['interactive','research'] as const) test(`${mode} approved synthesis uses interpretation instructions and supplied evidence`, async () => {
  const current = session(); current.messages.push({ role: 'assistant', content: 'obsolete interpretation numbers' }); current.pendingSynthesis = { id: 'synthesis', question: '解释图表', jobIds: ['job'], contextHash: 'hash' };
  const agent = new ConversationAgent({ mode, inference: provider([call('respond', { message: '图表解释' })], request => {
    assert.ok(request.tools?.every(tool => ['knowledge_list', 'knowledge_search', 'knowledge_read', 'respond'].includes(tool.name))); assert.equal(request.options?.extra?.toolChoice, 'auto');
    assert.equal(request.agentId, 'agent.theta.result-interpretation');
    const instructions=(request.input as {instructions:string}).instructions;
    assert.match(instructions,/独立授权的研究结果解读 Agent/);
    assert.match(instructions,/不是接口故障/);
    assert.doesNotMatch(instructions,/autonomous research mode/);
    assert.equal(JSON.stringify(request.input).includes('obsolete interpretation numbers'), false);
  }), tools: { execute: async () => { throw new Error('No mutation during synthesis'); } }, save() {} });
  assert.equal(await agent.turn(current, '已批准的证据'), '图表解释');
});

test('retry after inference failure retains the approved synthesis evidence boundary', async () => {
  let fail = true;
  const current = session(); current.messages.push({ role: 'assistant', content: 'obsolete numbers' });
  current.pendingSynthesis = { id: 'synthesis', question: '解释图表', jobIds: ['job'], contextHash: 'hash' };
  const agent = new ConversationAgent({ inference: provider([call('respond', { message: '基于原始证据完成解读' })], request => {
    if (fail) throw new Error('response interrupted');
    const input = JSON.stringify(request.input);
    assert.match(input, /approved sourceRow 180/);
    assert.doesNotMatch(input, /obsolete numbers/);
  }), tools: { execute: async () => { throw new Error('No new computation'); } }, save() {} });
  await assert.rejects(agent.turn(current, 'approved sourceRow 180'), /response interrupted/);
  fail = false;
  assert.equal(await agent.turn(current, '重试解读'), '基于原始证据完成解读');
});

test('latest observed job state follows historical dialogue in the model input', async () => {
  const current = session(); current.messages.push({ role: 'assistant', content: '训练进行中' });
  current.interpretations = [{ id: 'done', question: 'research', jobIds: ['job'], contextHash: 'hash', documentPath: '/saved.md' }];
  const agent = new ConversationAgent({ inference: provider([call('respond', { message: '训练已完成' })], request => {
    const messages = (request.input as { messages: Array<{ content: string }> }).messages;
    assert.equal(JSON.parse(messages.at(-1)!.content).currentResearch.lastObservedJob.status, 'completed');
    assert.equal(JSON.parse(messages.at(-1)!.content).completedInterpretations[0].documentPath, '/saved.md');
  }), tools: { readState: () => ({ lastObservedJob: { status: 'completed' } }), execute: async () => null }, save() {} });
  assert.equal(await agent.turn(current, '训练过程中能问问题吗'), '训练已完成');
});

test('conceptual conversation needs neither data nor a run; host approvals are absent from model tools', async () => {
  const current = session();
  const agent = new ConversationAgent({ inference: provider([call('respond', { message: '主题模型帮助概括文本集合中的主题。' })], (request) => { assert.ok(!request.tools?.some((tool) => tool.name.includes('approve'))); }), tools: { execute: async () => { throw new Error('No tool should execute'); } }, save() {}, activity() {} });
  assert.match(await agent.turn(current, '什么是主题模型？'), /概括/);
  assert.equal(current.runId, undefined);
});

test('agent chooses evidence tools and can continue result interpretation after completion', async () => {
  const current = session(); current.runId = 'completed-run';
  const executed: string[] = [];
  const agent = new ConversationAgent({
    inference: provider([call('results_read', { view: 'summary' }), call('respond', { message: '主题一涉及公共交通，指标为 0.52；这不能证明因果关系。' }), call('models_inspect', { modelId: 'lda' }), call('respond', { message: '可以比较 LDA，并在新研究中验证。' })]),
    tools: { execute: async (name) => { executed.push(name); return name === 'results_read' ? { evidence: { topics: ['公交', '地铁'], coherence: 0.52 } } : { modelId: 'lda' }; } }, save() {}, activity() {},
  });
  assert.match(await agent.turn(current, '解释结果'), /0.52/);
  assert.match(await agent.turn(current, '还能试什么模型？'), /LDA/);
  assert.deepEqual(executed, ['results_read', 'models_inspect']);
  const history = boundedHistory(current.messages);
  for (let index = 0; index < history.length; index++) if (history[index].role === 'tool') assert.equal(history[index - 1].role, 'assistant');
});

test('agent discovers existing data, waits for selection, then discusses business content', async () => {
  const current = session(); const executed: string[] = [];
  const agent = new ConversationAgent({
    inference: provider([call('datasets_discover'), call('respond', { message: '已有客服反馈数据，要使用它分析服务改进方向吗？' }),
      call('dataset_use', { catalogId: 'catalog-fixture' }), call('dataset_understand', { datasetRef: 'dataset-fixture' }),
      call('respond', { message: '样本涉及物流延迟与退款等待。你更希望改善配送还是售后处理？' })], (request) => {
        assert.ok(request.tools?.some(tool => tool.name === 'datasets_discover'));
        assert.match(String((request.input as { instructions: string }).instructions), /索要上传前用 datasets_discover/);
      }),
    tools: { execute: async (name) => {
      executed.push(name);
      if (name === 'datasets_discover') return { datasets: [{ catalogId: 'catalog-fixture', name: '客服反馈.csv' }] };
      if (name === 'dataset_use') { current.datasetRefs.push('dataset-fixture'); return { datasetRef: 'dataset-fixture' }; }
      return { excerpts: [{ sampleId: 'sample-1', text: '物流延迟，退款处理等待时间长' }] };
    } }, save() {}, activity() {},
  });
  await agent.turn(current, '帮我分析业务问题');
  assert.deepEqual(executed, ['datasets_discover']);
  assert.deepEqual(current.datasetRefs, []);
  assert.match(await agent.turn(current, '使用客服反馈'), /配送还是售后/);
  assert.deepEqual(executed, ['datasets_discover', 'dataset_use', 'dataset_understand']);
});

test('completion can trigger business interpretation of tables and figure data in the same conversation', async () => {
  const current = session(); current.runId = 'completed'; const views: unknown[] = [];
  const agent = new ConversationAgent({ inference: provider([
    call('results_read', { view: 'summary' }), call('results_read', { view: 'tables' }),
    call('results_read', { view: 'figures' }), call('respond', { message: '配送主题权重为60%，退款为40%；这是文本主题权重，并非客户占比。主题表支持优先抽查配送反馈。图表依据底层数据解释，未读取图像像素。' }),
  ]), tools: { execute: async (_name, input) => { views.push((input as { view: string }).view); return { tables: [{ relativePath: 'topic_table.csv', rows: [{ topic: '配送', strength: 0.6 }] }], figures: [{ basis: 'underlying_results_not_image_pixels' }] }; } }, save() {}, activity() {} });
  assert.match(await agent.turn(current, '主机事件：训练完成，业务目标是改善客服体验'), /并非客户占比/);
  assert.deepEqual(views, ['summary', 'tables', 'figures']);
});

test('failed calls return observations for recovery and repeated failure stops autonomous work', async () => {
  const current = session(); let calls = 0;
  const agent = new ConversationAgent({ inference: provider(Array.from({ length: 4 }, () => call('runtime_check', { modelId: 'theta' }))), tools: { execute: async () => { calls++; throw new Error('Missing model assets'); } }, save() {}, activity() {} });
  assert.match(await agent.turn(current, '检查模型'), /没有取得进展/);
  assert.equal(calls, 3);
  assert.ok(current.messages.some((message) => message.role === 'tool' && message.content.includes('Missing model assets')));
});

test('long tool conversations stay bounded without orphaned tool results', () => {
  const current = session();
  current.messages.push({ role: 'user', content: '继续分析数据' });
  for (let index = 0; index < 20; index++) {
    current.messages.push(
      { role: 'assistant', content: '', metadata: { toolCalls: [{ id: `call-${index}`, name: 'dataset_read', arguments: { operation: 'overview' } }] } },
      { role: 'tool', content: 'evidence'.repeat(3000), metadata: { toolCallId: `call-${index}` } },
    );
  }
  const history = boundedHistory(current.messages);
  assert.ok(JSON.stringify(history).length < 120000);
  assert.equal(history[0].content, '继续分析数据');
  assert.equal(history.at(-1)?.metadata?.toolCallId, 'call-19');
  for (let index = 0; index < history.length; index++) {
    if (history[index].role === 'tool') {
      assert.equal(history[index - 1].role, 'assistant');
      assert.equal((history[index - 1].metadata?.toolCalls as Array<{ id: string }>)[0].id, history[index].metadata?.toolCallId);
    }
  }
});

test('a model cannot forge approval through the tool executor', async () => {
  const { LocalProductTools } = await import('../tools/local-tools.js');
  const tools = new LocalProductTools({ runtimeDb: '/unused', uploadDir: '/unused', inferenceFactory: () => undefined });
  await assert.rejects(tools.execute('checkpoint_approve', {}, { session: session(), userMessage: 'hello', save() {} }), /Unknown tool/);
  await assert.rejects(tools.approve(session(), '确认'), /先展示/);
});

test('action confirmation yields immediately; the model cannot continue other tool calls behind the prompt', async () => {
  const current = session(); const executed: string[] = [];
  const agent = new ConversationAgent({ inference: provider([{ kind: 'tool_calls', toolCalls: [
    { id: 'effect', name: 'training_advance', arguments: {} }, { id: 'next', name: 'runtime_check', arguments: { modelId: 'lda' } },
  ] }]), tools: { execute: async (name) => { executed.push(name); return { needsUser: true, summary: '允许发送文本并运行训练？' }; } }, save() {}, activity() {} });
  assert.match(await agent.turn(current, '分析数据'), /允许发送/);
  assert.deepEqual(executed, ['training_advance']);
  assert.ok(current.messages.some((message) => message.role === 'tool'));
});

test('successful repeated reads execute once then permit recovery instead of disabling all work', async () => {
  const current = session(); let reads = 0; let requests = 0;
  const agent = new ConversationAgent({ inference: provider([
    call('dataset_read', { operation: 'overview' }), call('dataset_read', { operation: 'overview' }),
    call('dataset_read', { operation: 'overview' }), call('respond', { message: '已有数据主要记录客服反馈，先确认希望改善的业务环节。' }),
  ], request => { if (++requests === 4) assert.deepEqual(request.tools?.map(tool => tool.name), ['analysis_checkpoint','respond']); }),
  tools: { execute: async () => { reads++; return { rows: 100 }; } }, save() {}, activity() {} });
  assert.match(await agent.turn(current, '我的数据已经上传，做文本分析'), /客服反馈/);
  assert.equal(reads, 1);
  assert.equal(current.executionEvents?.filter(e => e.kind === 'tool' && e.status === 'cached').length, 2);
  for (const event of current.executionEvents ?? []) {
    assert.ok(event.durationMs >= 0); assert.equal(event.version, 1);
    if (event.status !== 'started') assert.ok(current.executionEvents?.some(e => e.callId === event.callId && e.status === 'started'));
  }
});

test('file-only host receipt preserves the prior user intent', async () => {
  const current = session(); current.lastUserIntent = '研究售后投诉的改进机会';
  await new ConversationAgent({ inference: provider([call('respond', { message: '继续研究售后投诉。' })]), tools: { execute: async () => ({}) }, save() {}, activity() {} })
    .turn(current, '用户上传了文件，无额外指令', undefined, { userIntent: '' });
  assert.equal(current.lastUserIntent, '研究售后投诉的改进机会');
});

test('a business understanding answer is handed to persistent context before the turn completes', async () => {
  const current = session(); let saved = '';
  const agent = new ConversationAgent({ inference: provider([call('dataset_understand'), call('respond', { message: '样本主要涉及物流延迟，目标是改善售后体验。' })]),
    tools: { execute: async () => ({ excerpts: ['物流延迟'] }), recordUnderstanding: (_session, text) => { saved = text; }, readContext: () => ({ markdown: '用户目标：改善售后' }) }, save() {} });
  const answer = await agent.turn(current, '理解数据'); assert.equal(saved, answer);
});

test('a batch of distinct checks also obeys the turn budget and ends with a real answer', async () => {
  let executions = 0; let inferenceCalls = 0;
  const agent = new ConversationAgent({ maxToolCalls: 2,
    inference: { id: 'fixture', infer: async request => {
      inferenceCalls++;
      if (request.tools?.length === 0) return { id: 'summary', output: call('respond', { message: '已检查两项证据，请先明确业务目标。' }) };
      return { id: 'checks', output: { kind: 'tool_calls', toolCalls: Array.from({ length: 12 }, (_, index) => ({ id: String(index), name: 'runtime_check', arguments: { modelId: String(index) } })) } };
    } }, tools: { execute: async () => { executions++; return { ready: true }; } }, save() {}, activity() {},
  });
  assert.match(await agent.turn(session(), '做文本分析'), /两项证据/);
  assert.equal(executions, 2); assert.equal(inferenceCalls, 2);
});


test('a prose training confirmation is repaired into a host action card before asking the user', async () => {
  const current = session(); const executed: string[] = [];
  const agent = new ConversationAgent({ inference: provider([
    call('plan_propose', {}), call('respond', { message: '训练方案确认：回复确认后开始训练。' }),
    call('training_advance'),
  ]), tools: { execute: async name => { executed.push(name); return name === 'training_advance' ? { needsUser: true, summary: '宿主确认：限定本次计算' } : { plan: {} }; } }, save() {} });
  assert.equal(await agent.turn(current, '请给我训练确认'), '宿主确认：限定本次计算');
  assert.deepEqual(executed, ['plan_propose', 'training_advance']);
  assert.ok(!current.messages.some(message => message.role === 'assistant' && message.content.includes('回复确认后开始训练')));
});


test('statistical interpretation isolates current evidence and rejects unoffered mutations', async () => {
  const current = session(); current.analysisMode = 'free';
  current.messages.push({role:'assistant',content:'obsolete evidence: DW 1.937'});
  current.pendingStatisticalInterpretation = 'stats-current';
  current.statisticalReports = [{analysisId:'stats-current',runId:current.id,status:'completed',reportPath:'/current/report.html',files:[],results:[{method:'ols',metrics:{nobs:180,inference_distribution:'t'}}]}];
  const executed: string[] = [];
  const agent = new ConversationAgent({mode:'research', inference:provider([
    call('statistics_plan',{}), call('statistics_results',{}), call('respond',{message:'当前模型使用 t 推断，N=180；未计算 DW。'})
  ], request => {
    const input = request.input as {messages:unknown[]};
    assert.doesNotMatch(JSON.stringify(input.messages),/obsolete evidence/);
    assert.match(JSON.stringify(input.messages),/stats-current/);
    assert.match(JSON.stringify(input.messages),/180/);
    assert.ok(request.tools?.every(tool => ['knowledge_list','knowledge_search','knowledge_read','statistics_results','respond'].includes(tool.name)));
    assert.equal(request.agentId,'agent.theta.statistical-interpretation');
  }),tools:{execute:async name=>{executed.push(name);return {analysisId:'stats-current'};},
    readContext:()=>({markdown:'obsolete evidence'}),readState:()=>({notes:'obsolete evidence'})},save(){}});
  assert.match(await agent.turn(current,'解释本次批准的计算'),/N=180/);
  assert.deepEqual(executed,['statistics_results']);
  assert.ok(current.messages.some(m=>m.role==='tool' && m.content.includes('统计解读只允许')));
});


test('requested statistical revision enters evidence isolation before claiming delivery', async () => {
  const current=session();current.analysisMode='free';current.runId='research-current';
  current.messages.push({role:'assistant',content:'obsolete shape claim'});
  current.statisticalReports=[{analysisId:'stats-delivered',runId:'research-current',status:'complete',reportPath:'/report.html',files:[],results:[]}];
  let requestCount=0;const executed:string[]=[];
  const agent=new ConversationAgent({mode:'research',inference:provider([
    call('respond',{message:'直接沿用旧文字。'}),call('statistics_synthesize',{analysisId:'stats-delivered'}),call('respond',{message:'当前证据的修订解读。'})
  ],request=>{
    requestCount++;
    if(requestCount===2)assert.match(JSON.stringify(request.input),/First call statistics_synthesize/);
    if(requestCount===3){assert.equal(request.agentId,'agent.theta.statistical-interpretation');assert.doesNotMatch(JSON.stringify(request.input),/obsolete shape claim/);}
  }),tools:{execute:async name=>{executed.push(name);current.pendingStatisticalInterpretation='stats-delivered';return {analysisId:'stats-delivered'};}},save(){}});
  assert.equal(await agent.turn(current,'请重新解读统计结果并保存修订文件'), '当前证据的修订解读。');
  assert.deepEqual(executed,['statistics_synthesize']);
});
