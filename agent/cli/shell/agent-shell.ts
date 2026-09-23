import path from 'node:path';
import type { ComputeJob } from '../../src/domain/research.js';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { parseAttachmentInput } from './attachment-input.js';
import { ExecutionTrace } from '../../src/conversation/execution-events.js';
import type { InferenceProvider } from '../../src/providers/types.js';
import { createConfiguredProvider, configuredProviderSummaries, selectConfiguredProvider, providerIdForBaseUrl } from '../../src/providers/configured-provider.js';
import { ReadlineInteractiveTerminal, type InteractiveTerminal } from './terminal-io.js';
import { ConversationAgent, observationText } from '../../src/conversation/conversation-agent.js';
import { LocalProductTools, isExplicitApproval, isExplicitDenial, explicitEmbeddingChoice } from '../../src/tools/local-tools.js';
import { ProductSessionStore, type ProductSession } from '../../src/memory/session-store.js';
import { AgentTerminalView } from '../ui/terminal-view.js';

export interface AgentShellOptions {
  terminal?: InteractiveTerminal;
  directory?: string;
  sessionId?: string;
  inferenceFactory?: () => InferenceProvider | undefined;
}

export class AgentShell {
  private readonly terminal: InteractiveTerminal;
  private readonly store: ProductSessionStore;
  private readonly view: AgentTerminalView;
  private readonly tools: LocalProductTools;
  private readonly inferenceFactory: () => InferenceProvider | undefined;
  private session: ProductSession;
  private controller?: AbortController;
  private background?: Promise<void>;
  private lastProgress?: string;
  private observation?: Promise<void>;
  private monitorErrorShown = false;

  constructor(options: AgentShellOptions = {}) {
    const directory = path.resolve(options.directory ?? process.env.THETA_AGENT_HOME ?? path.join(process.cwd(), '.theta_agent'));
    // Only the standalone local product selects local persistence by default.
    process.env.THETA_MEMORY_BACKEND ??= 'sqlite';
    this.terminal = options.terminal ?? new ReadlineInteractiveTerminal();
    this.view = new AgentTerminalView(this.terminal);
    this.store = new ProductSessionStore(directory);
    this.session = options.sessionId ? this.store.get(options.sessionId) : this.store.create();
    this.inferenceFactory = options.inferenceFactory ?? createConfiguredProvider;
    this.tools = new LocalProductTools({ runtimeDb: path.join(directory, 'research.sqlite'), uploadDir: path.join(directory, 'uploads'), inferenceFactory: this.inferenceFactory, onJobObserved: job => this.view.trainingResult(job) });
  }

  async run(): Promise<number> {
    const interrupt = (): void => { this.controller?.abort(); };
    process.on('SIGINT', interrupt);
    const stopListening = this.terminal.onPendingInput?.(() => {
      this.view.notice('已收到补充指令，正在停止当前对话调用后继续。后台训练不受影响。'); interrupt();
    });
    const monitor = setInterval(() => {
      if (!this.session.monitorTraining || this.observation || this.background) return;
      if (this.controller) {
        const session = this.session; const runId = session.runId;
        this.observation = this.tools.observeTraining(session).then(job => {
          if (this.session.id === session.id && this.session.runId === runId && this.session.monitorTraining && job) {
            this.view.training(job); this.monitorErrorShown = false;
          }
        }).catch(() => { if (this.session.id === session.id && this.session.runId === runId) this.monitorError(); }).finally(() => { this.observation = undefined; });
      } else {
        this.background = this.pollTraining().catch(() => this.monitorError()).finally(() => { this.background = undefined; });
      }
    }, 3000);
    try {
      this.view.welcome(this.session, this.modelLabel(), process.cwd());
      if (this.session.messages.length) {
        const last = [...this.session.messages].reverse().find((message) => message.role === 'assistant' && message.content);
        if (last && last.content !== this.session.pendingConfirmation?.summary) this.view.reply(last.content);
        if (this.session.pendingConfirmation) this.view.confirmation(this.session.pendingConfirmation.summary);
      }
      try { if (!this.inferenceFactory()) this.view.notice('尚未连接语言模型。输入 /model 开始配置；会话与文件保存在本地。'); }
      catch { this.view.notice('语言模型连接配置需要修正。输入 /model 重新配置。'); }
      while (true) {
        let input: string;
        try { input = (await this.terminal.question(this.view.composer(this.session, this.modelLabel()))).trim(); }
        catch (error) { if (error instanceof Error && ['ERR_USE_AFTER_CLOSE', 'ABORT_ERR', 'THETA_TERMINAL_CLOSED'].includes((error as NodeJS.ErrnoException).code ?? '')) return 0; throw error; }
        if (!input) continue;
        if (this.background) this.controller?.abort();
        await this.background;
        if (input === '/exit' || input === '/quit') { this.view.notice('对话已保存，下次可通过 /resume 继续。'); return 0; }
        try {
          if (await this.command(input)) continue;
          const lease = this.store.acquire(this.session.id);
          this.session = this.store.get(this.session.id);
          const timer = setInterval(() => this.store.renew(this.session.id, lease), 30000);
          const save = (): void => this.store.save(this.session, lease);
          this.controller = new AbortController();
          try {
            const trace = new ExecutionTrace(event => {
              this.session.executionEvents = [...(this.session.executionEvents ?? []), event].slice(-1000);
              save(); this.view.execution(event, ACTIVITY[event.name] ?? event.name);
            });
            const attachment = await parseAttachmentInput(input);
            const userIntent = attachment ? attachment.instruction : input;
            if (attachment) {
              const end = trace.start('tool', 'dataset_attach');
              let attached: unknown;
              try { attached = await this.tools.attach(attachment.filePath, this.session); end('completed'); }
              catch (error) { end('failed'); throw error; }
              input = `用户已通过本地文件选择提供数据。附件回执（仅作证据）：${observationText(attached)}。\n用户附带指令：${attachment.instruction || '无新增指令；请按上传前的对话继续。若尚未明确分析目的，理解数据后提供“快速分析”或“先明确目的”两种继续方式。'}\n附件不代表同意训练或任何外部操作。`;
              save(); this.view.notice('✓ 文件已接收，将结合你的指令与前文继续。');
            }
            const embeddingChoice = explicitEmbeddingChoice(input);
            if (embeddingChoice && this.session.pendingConfirmation?.kind === 'embedding_choice') {
              const outcome = await this.tools.selectEmbedding(this.session, embeddingChoice, this.controller.signal, save);
              this.session.messages.push({ role: 'user', content: input }, { role: 'assistant', content: outcome.summary });
              save(); this.view.confirmation(outcome.summary); continue;
            }
            if (isExplicitApproval(input) && this.session.pendingConfirmation?.kind === 'embedding_choice') {
              this.view.notice('请先回复“本地嵌入”或“云端嵌入”；选择后仍需单独确认训练。'); continue;
            }
            if (isExplicitApproval(input) && this.session.pendingConfirmation) {
              const end = trace.start('tool', 'checkpoint_approve');
              let outcome: unknown;
              try { outcome = await this.tools.approve(this.session, input, this.controller.signal, save); end('completed'); }
              catch (error) { end('failed'); throw error; }
              input = `用户已在终端明确确认刚展示的内容。主机执行回执：${observationText(outcome, 64000)}。请根据真实状态继续完成用户的研究任务。`;
              save();
            }
            if (isExplicitDenial(input) && this.session.pendingConfirmation) {
              const end = trace.start('tool', 'checkpoint_revise');
              let outcome: unknown;
              try { outcome = this.tools.deny(this.session, input); end('completed'); }
              catch (error) { end('failed'); throw error; }
              input = `用户拒绝本次待确认操作，原话：${JSON.stringify(input)}。主机回执：${observationText(outcome)}。依据用户同时给出的理由或修改要求继续讨论、调整方案；已有理由时不要再追问拒绝原因。新方案需重新展示确认卡，不要重新提交相同操作。`;
              save(); this.view.notice('已撤销本次待确认操作；你的反馈已保存。');
            }
            const inference = this.inferenceFactory();
            if (!inference) { this.view.notice('请先输入 /model 连接语言模型。文件和会话已保存在本地。'); continue; }
            const agent = new ConversationAgent({ inference, tools: this.tools, save, trace, activity() {} });
            const answer = await agent.turn(this.session, input, this.controller.signal, { userIntent });
            if (answer !== this.session.pendingConfirmation?.summary) this.view.reply(answer);
            this.view.executionSummary(this.session.executionEvents ?? []);
            if (this.session.contextId) this.view.notice('研究上下文已保存，可用 /context 查看、/contexts 选择复用。');
            if (this.session.pendingConfirmation) this.view.confirmation(this.session.pendingConfirmation.summary);
          } finally { this.terminal.activity?.(); if (!this.session.monitorTraining) this.view.training(); clearInterval(timer); this.controller = undefined; this.store.release(this.session.id, lease); }
        } catch (error) { this.view.error(`${error instanceof Error ? error.message : String(error)}\n\n会话已保留，你可以修正后继续。`); }
      }
    } finally { clearInterval(monitor); await this.observation; await this.background; stopListening?.(); process.removeListener('SIGINT', interrupt); this.terminal.close(); this.store.close(); }
  }

  private monitorError(): void {
    this.view.trainingNotice('状态查询失败，当前进度未知 · 3秒后自动重试');
    if (!this.monitorErrorShown) this.view.notice('暂时无法查询任务状态，监控会重试；不会重复提交训练。');
    this.monitorErrorShown = true;
  }

  private async pollTraining(): Promise<void> {
    const sessionId = this.session.id;
    let lease: string;
    try { lease = this.store.acquire(sessionId); } catch { return; }
    this.controller = new AbortController();
    const renew = setInterval(() => this.store.renew(sessionId, lease), 30000);
    const save = (): void => this.store.save(this.session, lease);
    try {
      this.session = this.store.get(sessionId);
      if (!this.session.monitorTraining) return;
      const result = await this.tools.execute('run_status', {}, { session: this.session, userMessage: '', save }) as { job: ComputeJob; summary?: string };
      this.view.training(result.job); this.monitorErrorShown = false;
      const progress = `${result.job.id}:${result.job.status}:${result.job.phase}:${result.job.telemetry?.activity}`;
      if (progress !== this.lastProgress) {
        this.lastProgress = progress;
        this.view.notice(`${result.summary ?? '正在更新训练状态'} · ${result.job.phase}（详细状态自动刷新）`);
      }
      if (!this.session.monitorTraining) {
        try {
          const inference = this.inferenceFactory();
          if (inference) {
            const agent = new ConversationAgent({ inference, tools: this.tools, save, activity() {}, onEvent: event => this.view.execution(event, ACTIVITY[event.name] ?? event.name) });
            const answer = await agent.turn(this.session, `主机监控事件（不是新的用户授权）：${observationText(result)}。完成时说明真实状态并用 results_read 提出结果整理与解读的独立确认，未确认前不要解读；综合业务解读另用 results_synthesize 确认；若失败，说明实际原因和恢复步骤。不要自动开始新的研究或训练。`, this.controller.signal, { userIntent: '' });
            if (answer !== this.session.pendingConfirmation?.summary) this.view.reply(answer);
            this.view.executionSummary(this.session.executionEvents ?? []);
            if (this.session.pendingConfirmation) this.view.confirmation(this.session.pendingConfirmation.summary);
          }
        } catch {
          this.view.notice(this.controller.signal.aborted
            ? '任务状态已更新，已停止自动说明以处理你的输入。'
            : '任务状态已更新，但自动说明未完成；可以继续询问状态或申请读取结果。');
        }
      }
    } finally { clearInterval(renew); this.controller = undefined; this.store.release(sessionId, lease); }
  }

  private modelLabel(): string {
    try {
      const provider = this.inferenceFactory();
      if (!provider) return '未连接模型';
      return 'model' in provider && typeof provider.model === 'string' ? provider.model : provider.id;
    } catch { return '模型配置待修正'; }
  }

  private async command(input: string): Promise<boolean> {
    if (/^\/mode(?:\s|$)/u.test(input)) {
      const mode = input.slice(5).trim();
      if (!['topic','free'].includes(mode)) { this.view.reply('用 /mode topic 选择主题分析，或 /mode free 选择自由分析。'); return true; }
      const lease = this.store.acquire(this.session.id);
      try {
        this.session = this.store.get(this.session.id);
        if (this.session.pendingConfirmation || this.session.pendingSynthesis || this.session.pendingStatisticalInterpretation) throw new Error('先处理当前确认或解读，再切换分析模式。');
        this.session.analysisMode = mode as 'topic' | 'free'; this.store.save(this.session, lease);
        this.view.notice(mode === 'free' ? '自由分析：按研究问题选择统计、建模、文本或优化方法。' : '主题分析：先探索主题，再围绕主题挖掘。');
      } finally { this.store.release(this.session.id, lease); }
      return true;
    }
    if (input === '/reports' || /^\/open(?:\s|$)/u.test(input)) {
      this.session = this.store.get(this.session.id);
      const reports = [...(this.session.reports ?? []), ...(this.session.statisticalReports ?? []).map(r=>({...r,jobId:r.analysisId,reportStatus:'complete' as const}))];
      if (input === '/reports') {
        this.view.reply(reports.length ? reports.map((report, i) => `${i + 1}. ${report.jobId}\n[${report.reportStatus === 'incomplete' ? '已有产物与诊断（整理未完成）' : '完整报告'}](<${report.reportPath}>)\n打开：/open ${i + 1}`).join('\n\n') : '本会话尚无已生成报告。可以向 Agent 请求整理已完成的结果。');
      } else {
        const selection = input.slice(5).trim();
        const index = selection ? Number(selection) - 1 : reports.length - 1;
        if (!Number.isInteger(index) || !reports[index]) throw new Error('请用 /reports 查看报告编号，再输入 /open 编号。');
        const report = reports[index];
        if (!path.isAbsolute(report.reportPath) || path.extname(report.reportPath) !== '.html') throw new Error('报告路径无效。');
        if (!['darwin', 'linux'].includes(process.platform)) { this.view.reply(`[请在浏览器打开报告](<${report.reportPath}>)`); return true; }
        await promisify(execFile)(process.platform === 'darwin' ? 'open' : 'xdg-open', [report.reportPath]);
        this.view.notice('已在默认浏览器打开报告。');
      }
      return true;
    }
    if (input === '/knowledge') {
      const catalog = this.tools.knowledgeCatalog() as { available: boolean; total?: number; nextOffset?: number | null; items?: Array<{ title: string; summary: string; date: string; useWhen: string[] }> };
      this.view.reply(catalog.available ? (catalog.items?.map(item => `${item.title}（${item.date}）\n${item.summary}\n适用：${item.useWhen.join('；')}`).join('\n\n') || '知识库暂为空。') : '知识目录暂不可用，请检查知识目录文件。');
      this.view.notice(`直接提问即可按需查阅知识。${catalog.nextOffset != null ? `当前仅展示前20条，共${catalog.total}条；可让 Agent 继续列出。` : ''}`);
      return true;
    }
    if (input === '/trace') { this.view.executionSummary(this.session.executionEvents ?? []); return true; }
    if (input === '/contexts' || input === '/context' || input.startsWith('/context ')) {
      const lease = this.store.acquire(this.session.id);
      try {
        this.session = this.store.get(this.session.id);
        const save = (): void => this.store.save(this.session, lease);
        if (input === '/contexts') {
          const values = await this.tools.execute('contexts_list', {}, { session: this.session, userMessage: input, save }) as Array<{ id: string; title: string; goal: string }>;
          this.view.reply(values.map(value => `${value.title}\n${value.goal}\n/context ${value.id}`).join('\n\n') || '还没有研究上下文，完成一次数据理解后会自动保存。');
        } else {
          if (input.startsWith('/context ')) await this.tools.execute('context_select', { contextId: input.slice(9).trim() }, { session: this.session, userMessage: input, save });
          if (!this.session.contextId) { this.view.notice('还没有研究上下文。'); return true; }
          const value = this.tools.readContext(this.session) as { markdown: string; documentPath: string };
          this.view.reply(value.markdown); this.view.notice(`Markdown：${value.documentPath}`);
        }
      } finally { this.store.release(this.session.id, lease); }
      return true;
    }
    if (input === '/help') {
      this.view.reply('直接输入自然语言即可。\n\n/attach "文件路径" 可选指令：上传后理解、提问或按前文继续\n/mode free 或 /mode topic：切换自由/主题分析\n/new：独立研究对话\n/sessions：查看最近会话\n/resume 会话标识：恢复对话\n/context：查看研究 Markdown\n/contexts：列出可复用上下文\n/context 标识：选用上下文副本\n/reports：列出已生成报告\n/open 编号：在浏览器打开报告\n/knowledge：查看持久知识目录\n/trace：查看上轮逐次调用与耗时\n/models：查看供应商配置\n/model deepseek：切换供应商\n/model：自定义连接\n/approve 或 确认：批准刚展示的操作\n/deny 理由 或 拒绝，修改要求：撤销本次操作并将反馈交给 Agent\n/exit：保存并退出；后台训练继续执行');
      return true;
    }
    if (input === '/new') { this.view.training(); this.lastProgress = undefined; this.session = this.store.create(); this.view.welcome(this.session, this.modelLabel(), process.cwd()); return true; }
    if (input === '/sessions') { this.view.sessions(this.store.list(), this.session.id); return true; }
    if (input.startsWith('/resume')) {
      const id = input.slice(7).trim() || (await this.terminal.question('会话标识：')).trim();
      this.session = this.store.get(id);
      this.view.training(); this.lastProgress = undefined;
      this.view.notice(`已恢复：${this.session.title}`);
      const last = [...this.session.messages].reverse().find((message) => message.role === 'assistant' && message.content);
      if (last && last.content !== this.session.pendingConfirmation?.summary) this.view.reply(last.content);
      if (this.session.pendingConfirmation) this.view.confirmation(this.session.pendingConfirmation.summary);
      return true;
    }
    if (input === '/models') {
      this.view.reply(`可用供应商\n\n${configuredProviderSummaries().map((provider) => `${provider.name} · ${provider.configured ? '已配置' : '未配置'} · ${provider.model}\n切换：/model ${provider.id}`).join('\n\n')}\n\n自定义连接：/model。切换仅影响当前终端，现有会话会保留。`);
      return true;
    }
    if (input.startsWith('/model ')) {
      const provider = selectConfiguredProvider(input.slice(7).trim());
      this.view.notice(`✓ 已切换到 ${provider.model}，下一条消息使用此连接。`);
      return true;
    }
    if (input === '/model') {
      this.view.notice('配置自定义连接。已有本地配置可用 /models 查看、/model deepseek 等直接切换。');
      const baseUrl = (await this.terminal.question('API Base URL（兼容 Chat Completions）：')).trim();
      const model = (await this.terminal.question('模型名称：')).trim();
      const apiKey = await (this.terminal.secret ? this.terminal.secret('API Key（不显示、不写入文件）：') : this.terminal.question('API Key：'));
      const previous = { base: process.env.THETA_INFERENCE_BASE_URL, model: process.env.THETA_INFERENCE_MODEL, key: process.env.THETA_INFERENCE_API_KEY, provider: process.env.THETA_INFERENCE_PROVIDER };
      try {
        process.env.THETA_INFERENCE_BASE_URL = baseUrl; process.env.THETA_INFERENCE_MODEL = model; process.env.THETA_INFERENCE_API_KEY = apiKey.trim();
        process.env.THETA_INFERENCE_PROVIDER = providerIdForBaseUrl(baseUrl);
        if (!baseUrl || !model || !apiKey.trim() || !createConfiguredProvider()) throw new Error('API 地址、模型名和密钥都需要填写。');
      } catch (error) {
        for (const [key, value] of Object.entries({ THETA_INFERENCE_BASE_URL: previous.base, THETA_INFERENCE_MODEL: previous.model, THETA_INFERENCE_API_KEY: previous.key, THETA_INFERENCE_PROVIDER: previous.provider })) {
          if (value === undefined) delete process.env[key]; else process.env[key] = value;
        }
        throw error;
      }
      this.view.notice(`✓ 已选择 ${model}。下一条消息将使用此连接。`); return true;
    }
    if (input.startsWith('/') && !input.startsWith('/attach ') && !isExplicitApproval(input) && !isExplicitDenial(input) && !await parseAttachmentInput(input)) {
      this.view.notice('未识别该操作。输入 /help 查看帮助，也可以直接描述需求。'); return true;
    }
    return false;
  }
}

const ACTIVITY: Record<string, string> = {
  statistics_methods:'查找统计方法', statistics_inspect:'核对方法规格', statistics_plan:'准备实证分析方案', statistics_request_approval:'展示统计分析确认卡', statistics_results:'读取统计分析证据',
  knowledge_list: '查看知识目录', knowledge_search: '检索参考知识', knowledge_read: '阅读知识章节',
  checkpoint_approve: '确认并提交本次操作',
  context_save: '保存研究上下文', context_read: '读取研究上下文', contexts_list: '查看可复用上下文', context_select: '复用研究上下文', dataset_attach: '导入文件',
  datasets_discover: '查看已有数据集', dataset_use: '选用已有数据集', dataset_understand: '阅读少量文本样本，理解业务内容',
  runs_list: '查找已有研究', models_list: '查看可用模型', models_inspect: '了解模型能力', runtime_check: '检查模型运行环境', run_create: '建立研究任务', run_update: '更新研究目标', run_status: '查看当前进度', research_read: '读取研究依据', research_continue: '推进研究分析', research_answer: '更新研究信息', dataset_read: '分析数据', checkpoint_review: '整理待确认内容', checkpoint_revise: '根据反馈调整方案', training_prepare: '检查训练准备情况', training_request_approval: '生成训练确认卡', training_advance: '准备训练授权', training_cancel: '准备停止训练', results_read: '整理结果与解读确认', results_synthesize: '业务综合解读确认',
};
