import { mkdirSync, readFileSync, writeFileSync, existsSync, renameSync, readdirSync } from 'node:fs';
import path from 'node:path';
import { contentHash } from '../src/domain/research.js';
import { ConversationAgent, observationText } from '../src/conversation/conversation-agent.js';
import { KnowledgeBase } from '../src/knowledge/knowledge-base.js';
import type { InferenceProvider } from '../src/providers/types.js';
import type { CapabilityWorker } from '../src/adapters/python-worker.js';
import type { ProductSession } from '../src/memory/session-store.js';

export interface AnalysisReportState {
  status: 'idle' | 'running' | 'complete' | 'failed'; phase: string; updatedAt?: string;
  error?: string; markdown?: boolean; pdf?: boolean; provider?: string;
}
const chapters = [
  ['2. 数据分析', '分析语料内容、分布、数据质量、异常、代表性与局限。引用实际统计和有编号的文本摘录，不能把摘录频率当全量频率。'],
  ['3. 任务分析', '根据真实任务目标及参数解释研究问题、分析单位、数据划分、比较目标和可回答范围。未提供的业务背景明确标为未知，不补造用户意图。'],
  ['4. 建模描述', '解释实际使用的模型、主题数、参数、预处理、训练及评估过程；核对模型知识与本次证据。分别说明全量、训练、验证、测试指标，默认全量测试不代表泛化成绩。'],
  ['5. 结论分析', '结合真实主题词、分布、指标、图表数据与前述分析综合结论，区分事实、解释与假设；说明局限和可执行后续建议。不虚构因果、显著性或模型优劣。'],
];

/** One durable, explicitly requested report per exact owned job. GET never runs inference. */
export class AnalysisReports {
  private readonly active = new Map<string, { controller: AbortController; work: Promise<void> }>();
  private readonly root: string;
  constructor(private readonly home: string, private readonly worker: CapabilityWorker) {
    this.root = path.join(home, 'analysis-reports');
    if (existsSync(this.root)) for (const id of readdirSync(this.root)) {
      try {
        const file = path.join(this.root, id, 'status.json');
        const state = JSON.parse(readFileSync(file, 'utf8')) as AnalysisReportState;
        if (state.status === 'running') writeFileSync(file, JSON.stringify({ ...state, status: 'failed', phase: '生成已中断', error: '上次应用关闭或服务重启中断了报告，请点击重试。' }));
      } catch { /* Ignore unrelated/incomplete directories. */ }
    }
  }
  private directory(key: string) { return path.join(this.root, contentHash(key)); }
  state(key: string): AnalysisReportState {
    try { return JSON.parse(readFileSync(path.join(this.directory(key), 'status.json'), 'utf8')); }
    catch { return { status: 'idle', phase: '尚未生成' }; }
  }
  file(key: string, format: string) {
    if (!['md', 'pdf'].includes(format)) throw new Error('不支持的报告格式');
    const state = this.state(key);
    if (!(format === 'md' ? state.markdown : state.pdf)) throw new Error('该格式尚未生成');
    return path.join(this.directory(key), `analysis.${format}`);
  }
  start(key: string, jobId: string, provider: InferenceProvider | undefined, objective = '') {
    const existing = this.state(key);
    if (this.active.has(key) || existing.status === 'complete') return existing;
    if (!provider) throw new Error('请先在“设置 → 模型 API”填写服务地址、模型名称和 API Key，再生成分析报告。');
    if (this.active.size >= 2) throw new Error('已有两个报告正在生成，请等待其中一个完成后重试。');
    const directory = this.directory(key); mkdirSync(directory, { recursive: true });
    const controller = new AbortController();
    let state: AnalysisReportState = { status: 'running', phase: '正在读取数据描述与结果证据', provider: provider.id };
    const save = (patch: Partial<AnalysisReportState>) => {
      state = { ...state, ...patch, updatedAt: new Date().toISOString() };
      const file = path.join(directory, 'status.json'); writeFileSync(file + '.tmp', JSON.stringify(state)); renameSync(file + '.tmp', file);
    };
    save({ markdown: existing.markdown });
    const work = (async () => {
      const signal = AbortSignal.any([controller.signal, AbortSignal.timeout(25 * 60_000)]);
      // A PDF-only retry must not spend more model API calls or overwrite the saved Markdown.
      if (!existing.markdown) {
        const evidence = await this.worker.call<any>('analysis_report.evidence', { home: this.home, jobId }, signal);
        writeFileSync(path.join(directory, 'evidence.json'), JSON.stringify(evidence));
        const knowledge = new KnowledgeBase();
        const session: ProductSession = { id: `report-${contentHash(key)}`, title: '完整分析报告', datasetRefs: [evidence.dataset.datasetRef], messages: [], updatedAt: new Date().toISOString() };
        const allowed = new Set(['knowledge_list', 'knowledge_search', 'knowledge_read', 'models_inspect', 'dataset_read', 'dataset_understand', 'results_read']);
        const tools = {
          toolAvailable: (name: string) => allowed.has(name),
          knowledgeCatalog: () => knowledge.catalog(),
          execute: async (name: string, args: any) => {
            if (!allowed.has(name)) throw new Error('本次仅生成分析报告，不执行训练或改变数据');
            if (name === 'knowledge_list') return knowledge.list(args);
            if (name === 'knowledge_search') return knowledge.search(args.query, args.documentId, args.limit);
            if (name === 'knowledge_read') return knowledge.read(args);
            if (name === 'models_inspect') {
              if (args.modelId !== evidence.plan.modelId) throw new Error('请读取当前任务的模型');
              return this.worker.call('models.inspect', args, signal);
            }
            if (name === 'results_read') {
              if (args.view === 'report' || (args.jobId && args.jobId !== jobId)) throw new Error('只可读取当前任务的现有结果，无需再次生成原生图表');
              return this.worker.call('compute.results', { home: this.home, jobId, view: args.view, offset: args.offset }, signal);
            }
            if (args.datasetRef && args.datasetRef !== evidence.dataset.datasetRef) throw new Error('数据集不属于本报告');
            return this.worker.call(name === 'dataset_read' ? 'dataset.profile' : 'dataset.understand', { ...args, dataset: evidence.dataset }, signal);
          },
        };
        const { dataset, ...publicEvidence } = evidence;
        const evidenceContext = `本次任务背景（仅作背景，不作为结果证据）：${objective.slice(0, 6000)}\n已验证证据：${observationText(publicEvidence, 36000)}`;
        let markdown = `# THETA 完整分析报告\n\n生成时间：${new Date().toISOString()}\n\n任务：${jobId}\n\n结果校验：${evidence.resultHash}\n\n${evidence.description}\n`;
        for (const [title, instruction] of chapters) {
          save({ phase: `Agent 正在撰写${title.slice(3)}` });
          const agent = new ConversationAgent({ inference: provider, tools, mode: 'report', save: () => {}, maxRounds: 16, maxToolCalls: 24, timeoutMs: 600000 });
          const text = await agent.turn(session, `${evidenceContext}\n请完成“${title}”章节：${instruction} 输出完整可交付正文，不要只说准备开展。`, signal);
          if (text.trim().length < 400) throw new Error(`${title} 内容过短，未形成完整论述，请检查模型响应后重试`);
          markdown += `\n## ${title}\n\n${text}\n`;
        }
        markdown += '\n---\n\n本报告由 Agent 基于保存的数据与模型结果生成；统计描述与模型解释有不同的证据强度。请结合原始结果审阅后使用。\n';
        writeFileSync(path.join(directory, 'analysis.md'), markdown, { mode: 0o600 });
        save({ markdown: true });
      }
      save({ phase: '正在排版并导出 PDF' });
      await this.worker.call('analysis_report.pdf', { directory }, signal);
      save({ status: 'complete', phase: '报告已生成', pdf: true });
    })().catch(error => save({ status: 'failed', phase: '报告生成未完成', error: error instanceof Error ? error.message : String(error) }))
      .finally(() => this.active.delete(key));
    this.active.set(key, { controller, work });
    return state;
  }
  async close() { for (const { controller } of this.active.values()) controller.abort(); await Promise.allSettled([...this.active.values()].map(item => item.work)); }
}
