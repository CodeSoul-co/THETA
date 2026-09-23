import { INSTRUCTIONS, INTERPRETATION_INSTRUCTIONS, RESEARCH_INSTRUCTIONS, FREE_ANALYSIS_INSTRUCTIONS, STATISTICAL_INTERPRETATION_INSTRUCTIONS, STATISTICAL_EVIDENCE_DISCIPLINE } from './prompts.js';
import { randomUUID } from 'node:crypto';
import { METRIC_INSTRUCTIONS } from './metric-instructions.js';
import { contentHash } from '../domain/research.js';
import { ExecutionTrace, type ExecutionEvent } from './execution-events.js';
import type { InferenceProvider, PromptMessage } from '../providers/types.js';
import { z } from 'zod';
import { productTools, toolDescriptor } from '../tools/tool-catalog.js';
import type { ProductToolExecutor } from '../tools/local-tools.js';
import { notebookKey, type ProductSession } from '../memory/session-store.js';

// Providers may return one inspection per model. The execution loop still enforces
// the turn budget and approval boundaries for every individual call.
const callsSchema = z.object({ kind: z.literal('tool_calls'), toolCalls: z.array(z.object({ id: z.string().min(1), name: z.string().min(1), arguments: z.record(z.unknown()) })).min(1).max(64) });
const answerSchema = z.object({ message: z.string().min(1).max(16000) }).strict();
const answerTool = { id: 'respond', name: 'respond', description: 'Speak to the user: answer, explain evidence, ask a focused question or describe a blocker. Ends this turn, keeping the conversation open.', inputSchema: { type: 'object', properties: { message: { type: 'string', minLength: 1 } }, required: ['message'], additionalProperties: false } };

const knowledgeTools = ['knowledge_list', 'knowledge_search', 'knowledge_read'];

const explainsApprovalBlocker = (message: string): boolean => /(?:无法|不能|未能|尚不能).{0,24}(?:确认卡|授权卡|训练|执行)|(?:缺少|缺失|未配置|配置不完整|未完整配置)/u.test(message);

export class ConversationAgent {
  constructor(private readonly options: {
    inference: InferenceProvider; tools: ProductToolExecutor; save(): void;
    activity?(name: string): void; onEvent?(event: ExecutionEvent): void;
    trace?: ExecutionTrace;
    maxRounds?: number; maxToolCalls?: number; timeoutMs?: number; maxOutputTokens?: number;
    /** Explicit host opt-in; cancellation and repeated-action recovery remain active. */
    unboundedResearch?: boolean;
    mode?: 'interactive' | 'research';
  }) {}

  async turn(session: ProductSession, userMessage: string, inputSignal?: AbortSignal, input: { userIntent?: string } = {}): Promise<string> {
    const deadline = this.options.unboundedResearch ? undefined : AbortSignal.timeout(this.options.timeoutMs ?? 180000);
    const signal = deadline ? (inputSignal ? AbortSignal.any([inputSignal, deadline]) : deadline) : (inputSignal ?? new AbortController().signal);
    const toolLimit = this.options.unboundedResearch ? Infinity : (this.options.maxToolCalls ?? 12);
    const roundLimit = this.options.unboundedResearch ? Infinity : (this.options.maxRounds ?? 8);
    const trace = this.options.trace ?? new ExecutionTrace(event => {
      session.executionEvents = [...(session.executionEvents ?? []), event].slice(-1000);
      this.options.save(); this.options.onEvent?.(event);
    });
    const intent = input.userIntent ?? userMessage;
    session.pendingUnderstanding = undefined;
    if (intent.trim()) session.lastUserIntent = intent;
    else session.lastUserIntent ??= '待确认用户业务目标';
    const turnStart = session.messages.length;
    if (session.pendingSynthesis && session.pendingSynthesis.evidenceMessageStart === undefined) {
      // Upgrade an interrupted web interpretation saved before evidence boundaries existed.
      let receiptIndex = -1;
      session.messages.forEach((message, index) => {
        if (!userMessage.includes('主机执行回执：{"resumedAction":"results.synthesize"')
          && message.role === 'user' && message.metadata?.webUserText === '确认'
          && message.content.includes('主机执行回执：{"resumedAction":"results.synthesize"')
          && session.pendingSynthesis!.jobIds.every(id => message.content.includes(id))) receiptIndex = index;
      });
      session.pendingSynthesis.evidenceMessageStart = receiptIndex >= 0 ? receiptIndex : turnStart;
    }
    session.messages.push({ role: 'user', content: userMessage });
    if (session.title === '新的研究对话') session.title = userMessage.slice(0, 60);
    this.options.save();
    const cache = new Map<string, unknown>(); const failures = new Map<string, number>();
    let calls = 0; let duplicateReads = 0; let proposedPlan = false; let answerRepairs = 0; let repairApproval = false;
    let lastFingerprint = '', consecutiveCalls = 0, blockedFingerprint = '', recoveryNeeded = false, assistantText = '';
    let checkpointAt = 0;
    let deliveryRepairs = 0; let statisticalRepairs = 0; let readStatisticalEvidence = false;
    const requestsReinterpretation = /(?:重新|修订|重写).{0,12}解读/u.test(intent) && !/(?:不要|不必|无需).{0,6}(?:重新|修订|重写).{0,12}解读/u.test(intent);

    const recentFingerprints: string[] = [];
    const requestsConfirmation = /授权卡|确认卡|展示.{0,8}确认|给.{0,12}确认|训练.{0,12}(?:确认|授权)/u.test(intent);
    const infer = async (summarize = false) => {
      const synthesis = !!session.pendingSynthesis;
      const statisticalSynthesis = !!session.pendingStatisticalInterpretation;
      const remaining = toolLimit - calls;
      if (this.options.mode === 'research' && remaining > 4 && !synthesis && !statisticalSynthesis && !repairApproval && !session.pendingConfirmation && calls - checkpointAt >= 8) recoveryNeeded = true;
      const offered = productTools.filter(tool => tool.name !== 'training_advance' && (this.options.tools.toolAvailable?.(tool.name,session) ?? true) && (!synthesis || knowledgeTools.includes(tool.name)) && (!statisticalSynthesis || [...knowledgeTools,'statistics_results'].includes(tool.name)) && (!repairApproval || ['training_request_approval', 'results_read', 'results_synthesize'].includes(tool.name)) && (!recoveryNeeded || tool.name === 'analysis_checkpoint'));
      const end = trace.start('inference', this.options.inference.id);
      let attemptEnd: ReturnType<ExecutionTrace['start']> | undefined;
      try {
        const result = await this.options.inference.infer({
          runId: session.runId ?? session.id, stepId: `dialogue:${randomUUID()}`, agentId: statisticalSynthesis ? 'agent.theta.statistical-interpretation' : synthesis ? 'agent.theta.result-interpretation' : 'agent.theta.conversation', modelAlias: 'runtime-selected',
          input: { instructions: METRIC_INSTRUCTIONS + (synthesis ? INSTRUCTIONS + INTERPRETATION_INSTRUCTIONS : this.options.mode === 'research' ? RESEARCH_INSTRUCTIONS : INSTRUCTIONS) + STATISTICAL_INTERPRETATION_INSTRUCTIONS + STATISTICAL_EVIDENCE_DISCIPLINE + (statisticalSynthesis ? '\n本次为当前 approvedStatisticalReport 的只读解读。仅使用当前回执及 statistics_results 读取的同一批证据。不可沿用历史数字、诊断或预处理结论。无需新计划、确认或执行；完整正文由宿主保存。' : '') + (session.analysisMode === 'free' ? FREE_ANALYSIS_INSTRUCTIONS : '') + (summarize ? '\nThe configured tool budget is exhausted for this turn. Report only actually completed results and exact remaining blockers. Do not claim unexecuted actions or a host denial that did not occur.' : ''),
            messages: [...(synthesis ? session.messages.slice(session.pendingSynthesis?.evidenceMessageStart ?? turnStart) : statisticalSynthesis ? session.messages.slice(turnStart) : this.options.mode === 'research' ? researchHistory(session.messages, notebookKey(session)) : boundedHistory(session.messages)), { role: 'system', content: JSON.stringify({ session: session.id, approvedStatisticalReport: session.statisticalReports?.find(r=>r.analysisId===session.pendingStatisticalInterpretation) ?? null, analysisMode: session.analysisMode ?? 'topic', statisticalExecution:session.statisticalExecution ?? null, statisticalPlan: session.statisticalPlans?.[notebookKey(session)] ?? null, statisticalReports: session.statisticalReports?.filter(r=>r.runId===notebookKey(session)).map(r=>({analysisId:r.analysisId,status:r.status,reportPath:r.reportPath})) ?? [], selectedRun: session.runId ?? null, availableRuns: session.runIds ?? [],
              ...(this.options.mode === 'research' ? {originalUserObjective:session.lastUserIntent} : {}),
              availableDatasets: session.datasetRefs, pendingConfirmation: session.pendingConfirmation ?? null,
              authorizationStatus: session.pendingConfirmation ? 'A host action is pending.' : (summarize ? '本轮调用预算已用完，尚未建立可点击的确认请求；这是本轮预算限制，不是宿主拒绝训练授权。准确报告最后的失败回执，不能把失败的计划说成已保存或把拟修复项说成已修复。' : '尚无待确认请求。可以先修复计划校验错误，再通过 training_request_approval 创建真实确认卡。创建请求无需事先批准，不执行训练或外部嵌入；执行仍必须等待用户确认。'),
              knowledgeCatalog: this.options.tools.knowledgeCatalog?.() ?? null,
              currentResearch: statisticalSynthesis ? null : this.options.tools.readState?.(session) ?? null,
              approvedSynthesis: session.pendingSynthesis ?? null,
              deliveredReports: session.reports?.map(({ jobId, reportPath }) => ({ jobId, reportPath })) ?? [],
              completedInterpretations: statisticalSynthesis ? [] : session.interpretations?.slice(-5).map(({ id, question, jobIds, documentPath }) => ({ id, question, jobIds, documentPath })) ?? [],
              analysisNotebook: {provenance: 'Agent-authored working notes, not verified evidence or authorization', value: statisticalSynthesis ? null : session.analysisNotebooks?.[notebookKey(session)] ?? null},
              analysisProgress: statisticalSynthesis ? null : this.options.tools.analysisProgress?.(session) ?? null,
              deliveryBudget: this.options.unboundedResearch ? {unbounded:true,instruction:'No cumulative time or tool-call budget. Continue substantive research, preserve evidence and deliver when the objective is met. Repeating unchanged actions is not progress.'} : {remainingToolCalls:remaining,reserveForDelivery:Math.max(3,Math.ceil(toolLimit*0.2)),instruction:remaining <= Math.max(3,Math.ceil(toolLimit*0.2)) ? 'Converge now: execute one complete candidate analysis, verify required files and deliver. Report unsupported hypotheses honestly; do not open new exploration branches or weaken user requirements.' : 'Prefer reusable computations and saved evidence over repeated reconnaissance.'},
              executionProgress: {toolCalls: calls, toolLimit: this.options.unboundedResearch ? null : toolLimit, recoveryNeeded,
                instruction: recoveryNeeded ? 'Progress checkpoint required (periodic or repeated-action recovery). Record actual evidence, completed work and one concrete next action in analysis_checkpoint, then continue. No new execution authorization is granted.' : 'Preserve progress in analysis_checkpoint after substantive discoveries and finish the requested deliverable.'},
              researchContext: statisticalSynthesis ? null : this.options.tools.readContext?.(session) ?? null }) }] },
          tools: summarize ? [] : [...offered.map(toolDescriptor), answerTool],
          options: { temperature: 0.2, maxTokens: synthesis ? 8192 : (this.options.maxOutputTokens ?? 8192), extra: { toolChoice: summarize ? 'none' : 'auto', allowTextResponse: true, signal,
            onProviderAttempt: (status: 'started' | 'completed' | 'failed' | 'cancelled', attempt: number) => {
              if (status === 'started') attemptEnd = trace.start('provider', this.options.inference.id, attempt);
              else { attemptEnd?.(status); attemptEnd = undefined; }
            } } },
        });
        assistantText = typeof result.metadata?.assistantText === 'string' ? result.metadata.assistantText.slice(0, 8000) : '';
        end('completed'); return callsSchema.parse(result.output).toolCalls;
      } catch (error) { attemptEnd?.(signal.aborted ? 'cancelled' : 'failed'); end(signal.aborted ? 'cancelled' : 'failed'); throw error; }
    };
    try {
      for (let round = 0; round < roundLimit; round++) {
        signal.throwIfAborted();
        if (calls >= toolLimit) break;
        const toolCalls = await infer();
        if (toolCalls.length === 1 && toolCalls[0].name === 'respond') {
          const message = answerSchema.parse(toolCalls[0].arguments).message;
          if(requestsReinterpretation && (readStatisticalEvidence || /统计|频数|相关系数/u.test(intent)) && !session.pendingSynthesis && !session.pendingStatisticalInterpretation && !session.pendingConfirmation && session.statisticalReports?.some(r=>r.runId===notebookKey(session)&&r.status==='complete') && statisticalRepairs++<2){
            session.messages.push({role:'system',content:'Delivery check: this user explicitly requested a revised statistical interpretation. First call statistics_synthesize for the chosen delivered batch, then use its current evidence and statistics_results as needed. This isolates historical assistant claims and saves the revision. Reading tables or writing prose alone does not save a revised interpretation. No new computation or approval is needed.'});
            continue;
          }
          const progress = this.options.tools.analysisProgress?.(session) as {plan?:unknown;workspace?:{delivered?:boolean}} | undefined;
          if (this.options.mode==='research' && !session.pendingSynthesis && !session.pendingStatisticalInterpretation && !session.pendingConfirmation && progress?.plan && progress.workspace && !progress.workspace.delivered && deliveryRepairs++ < 2) {
            session.messages.push({role:'system',content:'Delivery check: the current analysis plan has no verified archived deliverable. A proposed script or promise is not a file. Use the available analysis tools to execute/verify and deliver your actual required files. If evidence is insufficient, persist an honest limitations/abstention artifact in the user-required format. Do not invent findings or weaken the original task. A genuine remaining blocker may be reported after these bounded recovery attempts.'});
            continue;
          }
          if (!session.pendingSynthesis && !session.pendingStatisticalInterpretation && !session.pendingConfirmation && !explainsApprovalBlocker(message) && ((proposedPlan || requestsConfirmation) && /(?:待授权|授权卡|授权方案|确认卡)|(?:回复|输入).{0,16}(?:确认|approve)|(?:训练|计算).{0,8}确认|确认后.{0,12}(?:执行|训练|开始)/u.test(message) || /请确认是否允许|请确认是否.{0,12}(?:读取|解读|生成)|是否允许我.{0,12}(?:读取|解读|生成)|(?:是否|要不要|可以|需要).{0,8}(?:申请|请求|发起).{0,16}(?:结果|读取|解读).{0,8}(?:确认|授权)|(?:申请|请求).{0,12}(?:结果|读取|解读).{0,12}授权[，,、。\s]*(?:可以吗|好吗|是否可以)/u.test(message)) && answerRepairs++ < 2) {
            repairApproval = true;
            session.messages.push({ role: 'system', content: 'Host response validation: NO pending execution approval exists. Do not fabricate a confirmation card in prose. Create the host request for the action you just asked about: training_request_approval for training, results_read for report/basic interpretation, results_synthesize for business interpretation. Requesting a card does not execute the action and needs no prior permission. Otherwise respond without asking the user to approve a nonexistent action.' });
            continue;
          }
          return this.finish(session, (proposedPlan || requestsConfirmation) && !session.pendingConfirmation && answerRepairs > 2 ? '当前没有可批准的执行请求，也未启动新计算。请先处理本轮工具反馈中的阻碍，再请求确认。' : message);
        }
        for (const call of toolCalls) {
          signal.throwIfAborted();
          if (call.name === 'respond') continue;
          if (calls >= toolLimit) break;
          const fingerprint = contentHash({ name: call.name, args: call.arguments, run: session.runId, datasets: session.datasetRefs });
          consecutiveCalls = fingerprint === lastFingerprint ? consecutiveCalls + 1 : 1;
          lastFingerprint = fingerprint;
          recentFingerprints.push(fingerprint); if (recentFingerprints.length > 8) recentFingerprints.shift();
          if ((failures.get(fingerprint) ?? 0) >= 3) return this.finish(session, '同一步骤暂时没有取得进展，我已暂停自动执行。可以查看状态、调整要求或稍后继续。', false);
          calls++;
          this.options.activity?.(call.name);
          const detail = [call.arguments.operation, call.arguments.modelId, call.arguments.column, call.arguments.query, call.arguments.documentId,
            ...(Array.isArray(call.arguments.columns) ? call.arguments.columns : [])].filter((value): value is string => typeof value === 'string').join(' · ');
          const end = trace.start('tool', call.name, undefined, detail);
          let observation: unknown;
          const cacheable = ['datasets_discover', 'dataset_read', 'dataset_understand', 'models_list', 'models_inspect', 'runtime_check', 'runtime_config', ...knowledgeTools].includes(call.name);
          if (cacheable && cache.has(fingerprint)) {
            duplicateReads++;
            if (duplicateReads >= 2 && !session.pendingSynthesis && !session.pendingStatisticalInterpretation) recoveryNeeded = true;
            observation = { ok: true, data: cache.get(fingerprint), cached: true, instruction: recoveryNeeded
              ? '已有读取成功但发生重复。现在用 analysis_checkpoint 保存已知证据和一个不同的下一步，再继续研究；工具并未失效，不能声称宿主拒绝了未尝试的操作。'
              : '相同读取已成功，复用本轮证据；选择一个能推进当前目标的不同操作，不要重复检查。' }; end('cached');
          } else {
            try {
              if (session.pendingSynthesis && !knowledgeTools.includes(call.name)) throw new Error('独立解读只允许读取参考知识，不能调用其他能力或改变研究。');
              if (session.pendingStatisticalInterpretation && ![...knowledgeTools, 'statistics_results'].includes(call.name)) throw new Error('统计解读只允许读取当前批次结果和参考知识，不能改变研究或启动新计算。');
              const polling = ['run_status', 'training_advance', 'training_request_approval'].includes(call.name);
              if (!cacheable && !polling && call.name !== 'analysis_checkpoint' && (consecutiveCalls >= 3 || recentFingerprints.filter(value => value === fingerprint).length >= 3 || fingerprint === blockedFingerprint)) {
                blockedFingerprint = fingerprint; recoveryNeeded = true;
                throw new Error('Repeated identical action stopped before execution. Save analysis_checkpoint with actual progress, evidence and a DIFFERENT next action, or explain the blocker. Repeating a successful computation is not progress. Notes do not authorize actions.');
              }
              if (recoveryNeeded && call.name !== 'analysis_checkpoint') throw new Error('First save an analysis_checkpoint to diagnose the repeated-action loop.');
              const data = await this.options.tools.execute(call.name, call.arguments, { session, userMessage, signal, save: this.options.save });
              if (call.name === 'analysis_checkpoint') { recoveryNeeded = false; duplicateReads = 0; checkpointAt = calls; recentFingerprints.length = 0; }
              else if (fingerprint !== blockedFingerprint) blockedFingerprint = '';
              if (call.name === 'plan_propose') proposedPlan = true;
              if (call.name === 'statistics_results') readStatisticalEvidence = true;
              observation = { ok: true, data }; if (cacheable) cache.set(fingerprint, data); failures.delete(fingerprint); end('completed');
            } catch (error) {
              failures.set(fingerprint, (failures.get(fingerprint) ?? 0) + 1);
              observation = { ok: false, error: error instanceof Error ? error.message : String(error), instruction: 'Explain or repair this error. Do not claim success or blindly retry.' };
              end(signal.aborted ? 'cancelled' : 'failed');
            }
          }
          session.messages.push({ role: 'assistant', content: assistantText, metadata: { toolCalls: [call] } },
            { role: 'tool', content: observationText(observation), metadata: { toolCallId: call.id, runId:session.runId ?? null, analysisScope:notebookKey(session) } });
          assistantText = '';
          this.options.save();
          const outcome = observation as { ok: boolean; data?: { needsUser?: boolean; summary?: string } };
          if (outcome.ok && outcome.data?.needsUser) return this.finish(session, outcome.data.summary ?? '这次操作需要你的确认。', false);
        }
      }
      const response = await infer(true);
      if (response.length === 1 && response[0].name === 'respond') {
        const message = answerSchema.parse(response[0].arguments).message;
        if ((proposedPlan || requestsConfirmation) && !session.pendingConfirmation && !explainsApprovalBlocker(message) && /授权卡|确认卡|训练.{0,8}确认|回复.{0,12}确认/u.test(message)) {
          return this.finish(session, '本轮尚未建立可批准的执行请求，也没有启动新计算。已有证据保留；请先解决工具反馈中的阻碍。', false);
        }
        return this.finish(session, message);
      }
      return this.finish(session, `本轮已达到读取上限（${calls} 次工具请求），已停止继续调用。已有证据保留，请指定最关心的业务问题。`, false);
    } catch (error) {
      if (signal.aborted) return this.finish(session, deadline?.aborted && !inputSignal?.aborted ? '本轮等待已达到时间上限，已中止新调用。已有证据与调用记录保留；可继续提问。' : '本轮已暂停。已有证据与调用记录保留，后台训练不受影响。', false);
      throw error;
    }
  }

  private finish(session: ProductSession, message: string, saveUnderstanding = true): string {
    const contextText = message;
    if (saveUnderstanding) {
      const synthesis = session.pendingSynthesis;
      this.options.tools.recordUnderstanding?.(session, message);
      message = this.options.tools.decorateAnswer?.(session, message) ?? message;
      if (synthesis) {
        const document = session.interpretations?.find(item => item.id === synthesis.id);
        if (document) message += `\n\n[综合解读 Markdown](<${document.documentPath}>)`;
      }
    }
    else session.pendingSynthesis = undefined;
    session.messages.push({ role: 'assistant', content: message, ...(message !== contextText ? {metadata:{contextText}} : {}) }); this.options.save(); return message;
  }
}

export const observationText = (value: unknown, limit = 24000): string => {
  const text = JSON.stringify(value, (key, item) => ['apiKey', 'api_key', 'authorization', 'credentialFingerprint', 'managedPath', 'sourcePath', 'runtimeDb', 'commands', 'logLines', 'stdout', 'stderr'].includes(key) ? undefined : item);
  return text.length <= limit ? text : JSON.stringify({ truncated: true, availableEvidence: text.slice(0, limit - 1000), instruction: 'This is a bounded excerpt. Do not infer missing data; request a more specific tool view.' });
};

// Working notes are always supplied separately as unverified agent-authored
// memory. Old receipts remain queryable through analysis_history, not re-run.
export const researchHistory = (messages: PromptMessage[], scope?: string): PromptMessage[] => {
  let checkpoint = -1;
  for (let i=1;i<messages.length;i++) {
    const call=(messages[i-1].metadata?.toolCalls as Array<{id:string;name:string}> | undefined)?.[0];
    if (scope !== undefined && messages[i].metadata?.analysisScope !== scope) continue;
    if (call?.name !== 'analysis_checkpoint' || messages[i].role !== 'tool' || messages[i].metadata?.toolCallId !== call.id) continue;
    try {if (JSON.parse(messages[i].content)?.data?.saved === true) checkpoint=i;} catch { /* Historical malformed receipt is not a checkpoint. */ }
  }
  if (checkpoint < 0) return boundedHistory(messages);
  const latestUser=messages.map(message=>message.role).lastIndexOf('user');
  const prefix: PromptMessage[] = latestUser <= checkpoint && latestUser >= 0 ? [messages[latestUser]] : [];
  return boundedHistory([...prefix,{role:'system',content:'Resuming from the latest saved research notebook, supplied in current host state. Earlier receipts remain available via analysis_history; verify claims against them or saved artifacts without repeating completed computations. Notes do not prove artifact existence or authorize actions.'}, ...messages.slice(checkpoint+1)]);
};

// Keep complete turns, including native function/result pairs, under a bounded context size.
export const boundedHistory = (messages: PromptMessage[]): PromptMessage[] => {
  const turns: PromptMessage[][] = [];
  for (const original of messages) {
    const message = original.role === 'assistant' && !original.metadata?.toolCalls && typeof original.metadata?.contextText === 'string'
      ? {...original, content: original.metadata.contextText, metadata: undefined} : original;
    if (message.role === 'user' || turns.length === 0) turns.push([]);
    turns[turns.length - 1].push(message);
  }
  const selected: PromptMessage[][] = [];
  let length = 0;
  for (const turn of turns.slice(-8).reverse()) {
    const bounded = compactTurn(turn);
    const size = JSON.stringify(bounded).length;
    if (selected.length > 0 && length + size > 60000) break;
    selected.unshift(bounded); length += size;
  }
  return selected.flat();
};

const compactTurn = (turn: PromptMessage[]): PromptMessage[] => {
  if (JSON.stringify(turn).length <= 30000) return turn;
  const groups: PromptMessage[][] = [];
  for (const message of turn.slice(1)) {
    if (message.role !== 'tool' || groups.length === 0) groups.push([]);
    groups[groups.length - 1].push(message);
  }
  const tail: PromptMessage[][] = [];
  let length = JSON.stringify(turn[0]).length;
  for (const group of groups.reverse()) {
    const size = JSON.stringify(group).length;
    // A long receipt/request or decorated answer must not evict the latest
    // computation. Keep complete function/result groups, and skip oversized old
    // groups rather than dropping every more recent observation.
    if (length + size > Math.max(28000, JSON.stringify(turn[0]).length + 30000) && tail.length > 0) continue;
    tail.unshift(group); length += size;
  }
  return [turn[0], { role: 'system', content: 'Older observations in this turn were omitted for context size. Read current research state or result evidence again if needed; do not repeat completed mutations.' }, ...tail.flat()];
};
