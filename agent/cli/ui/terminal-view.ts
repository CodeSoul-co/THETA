import { stripVTControlCharacters } from 'node:util';
import type { InteractiveTerminal } from '../shell/terminal-io.js';
import type { ProductSession } from '../../src/memory/session-store.js';
import type { ComputeJob } from '../../src/domain/research.js';
import type { ExecutionEvent } from '../../src/conversation/execution-events.js';

const segmenter = new Intl.Segmenter(undefined, { granularity: 'grapheme' });
/** Treat model/file content as text, including escape sequences and carriage returns. */
export const terminalText = (value: string): string => stripVTControlCharacters(value).replace(/[\x00-\x08\x0b-\x1f\x7f]/gu, '').replace(/\t/gu, '  ');
export const cellWidth = (value: string): number => [...segmenter.segment(terminalText(value))].reduce((sum, { segment }) => {
  if (/^[\p{Mark}\u200d\ufe0f]+$/u.test(segment)) return sum;
  const code = segment.codePointAt(0) ?? 0;
  return sum + (/\p{Extended_Pictographic}|\p{Regional_Indicator}/u.test(segment) || code >= 0x1100 && (code <= 0x115f || code >= 0x2e80 && code <= 0xa4cf || code >= 0xac00 && code <= 0xd7a3 || code >= 0xf900 && code <= 0xfaff || code >= 0xfe10 && code <= 0xfe6f || code >= 0xff01 && code <= 0xff60 || code >= 0x20000) ? 2 : 1);
}, 0);

export const wrapCells = (value: string, width: number): string[] => terminalText(value).split('\n').flatMap((line) => {
  const lines: string[] = []; let current = ''; let used = 0;
  for (const { segment } of segmenter.segment(line)) {
    const size = cellWidth(segment);
    if (used + size > width && current) { lines.push(current); current = ''; used = 0; }
    current += segment; used += size;
  }
  lines.push(current); return lines;
});

/** Percent is a pipeline marker. Only expose observed progress, never an ETA. */
export const trainingStatusText = (job: ComputeJob): string => {
  const t = job.telemetry;
  const age = (seconds: number): string => seconds < 60 ? `${Math.floor(seconds)}秒` : `${Math.floor(seconds / 60)}分${Math.floor(seconds % 60)}秒`;
  const phase = ({ preparing: '整理输入', downloading: '准备输入', preparing_data: '数据预处理', training: '模型执行', uploading: '整理产物' } as Record<string, string>)[job.phase] ?? '执行中';
  const state = job.resultWarning ? '执行已结束 · 结果导出不完整' : job.status === 'running' ? t?.activity === 'visualizing' ? '生成训练附属图表' : phase : ({ queued: '排队等待 worker 领取', completed: '训练已完成', failed: '训练失败', cancelled: '训练已取消' })[job.status];
  const lines = [`${state}${t?.iteration ? ` · 已报告迭代 ${t.iteration.current}/${t.iteration.total}` : ''}${t?.elapsedSeconds != null ? ` · 已运行 ${age(t.elapsedSeconds)}` : ''}`];
  if (job.status === 'running' || job.status === 'queued') {
    if (!t) lines.push('暂无细粒度进度与心跳信息 · 每3秒查询');
    else {
      if (t.phaseElapsedSeconds != null) lines.push(`本阶段 ${age(t.phaseElapsedSeconds)}`);
      lines.push(`${t.heartbeatAgeSeconds == null ? '心跳未知' : `心跳 ${age(t.heartbeatAgeSeconds)}前`}${t.lastLogAgeSeconds == null ? ' · 暂无日志' : ` · 日志 ${age(t.lastLogAgeSeconds)}前`} · 每3秒查询`);
      if (t.health === 'unresponsive') lines.push('超过30秒未收到心跳，状态不确定；不会自动重训');
      else if (job.status === 'running' && (t.lastLogAgeSeconds == null || t.lastLogAgeSeconds >= 30)) lines.push('worker 仍响应，暂无新日志；尚不能判断算法是否推进');
      else if (job.status === 'running') lines.push('有近期活动记录；阶段标记不代表完成比例');
    }
  }
  return lines.join('\n');
};

export class AgentTerminalView {
  private readonly shownResultDirectories = new Set<string>();
  private readonly callNumbers = new Map<string, number>();
  constructor(private readonly terminal: InteractiveTerminal) {}
  private get width(): number { return Math.max(12, Math.min(96, (this.terminal.columns ?? 80) - 2)); }
  private ink(value: string, color: number): string { return this.terminal.isTTY && process.env.NO_COLOR === undefined && process.env.TERM !== 'dumb' ? `\x1b[${color}m${value}\x1b[0m` : value; }
  private text(value: string, color?: number): string { const safe = terminalText(value); return color === undefined ? safe : this.ink(safe, color); }
  private line(value = ''): string { return wrapCells(value, this.width - 2).map((part) => `  ${part}`).join('\n'); }
  private box(title: string, content: string, color = 36): string {
    const inside = this.width - 4;
    const rows = wrapCells(content, inside);
    const label = wrapCells(` ${title} `, this.width - 4)[0];
    return [this.ink(`╭─${label}${'─'.repeat(Math.max(0, this.width - 3 - cellWidth(label)))}╮`, color),
      ...rows.map((row) => `${this.ink('│', color)} ${row}${' '.repeat(Math.max(0, inside - cellWidth(row)))} ${this.ink('│', color)}`),
      this.ink(`╰${'─'.repeat(this.width - 2)}╯`, color)].join('\n');
  }
  welcome(session: ProductSession, model: string, directory: string): void {
    this.terminal.write(`\n${this.box('Θ  THETA', `以主题建模为核心的数据分析 Agent\n\n从一个问题，到有证据支持的分析。\n讨论目标 · 理解数据 · 构建模型 · 解读结果\n\n模型  ${model}\n会话  ${session.title}\n目录  ${directory}`)}\n`);
    this.terminal.write(this.line('描述你的研究或业务问题，也可以说“看看已有数据集”或拖入文件。'));
    this.terminal.write(this.ink(this.line('/model 连接模型   /sessions 历史   /help 帮助'), 90));
  }
  composer(session: ProductSession, model: string): string {
    this.terminal.activity?.();
    const state = session.pendingConfirmation ? '等待确认' : session.monitorTraining ? '任务监控中' : '就绪';
    this.terminal.write(`\n${this.ink('─'.repeat(this.width), 90)}\n${this.ink(this.line(`${model} · ${state} · ${session.datasetRefs.length} 个数据集`), 90)}\n${this.ink(this.line('Enter 发送 · ↑ 历史 · /help 帮助 · Ctrl+C 中断/退出'), 90)}`);
    return this.ink('❯ ', 36);
  }
  reply(message: string): void {
    this.terminal.activity?.();
    let code = false;
    const rendered = terminalText(message).split('\n').map((line) => {
      if (line.startsWith('```')) { code = !code; return this.ink(this.line(code ? `┌ ${line.slice(3) || '代码'}` : '└'), 90); }
      if (code) return wrapCells(line, this.width - 4).map((part) => `  ${this.ink('│', 90)} ${part}`).join('\n');
      const artifact = line.match(/^\s*(?:-\s*)?\[([^\]]+)\]\(<?(\/[^>]+?)>?\)\s*$/u);
      if (artifact) return `${this.line(artifact[1])}\n  ${artifact[2]}`;
      if (/^#{1,6}\s/u.test(line)) return this.ink(this.line(line.replace(/^#{1,6}\s/u, '')), 1);
      return this.line(line).replace(/\*\*([^*]+)\*\*/gu, (_all, bold: string) => this.ink(bold, 1)).replace(/`([^`]+)`/gu, (_all, literal: string) => this.ink(literal, 36));
    }).join('\n');
    this.terminal.write(`\n${this.ink('● THETA', 36)}\n${rendered}`);
  }
  confirmation(summary: string): void {
    this.terminal.activity?.();
    this.terminal.write(`\n${this.box('本次操作授权', `${summary}\n\n回复“确认”执行；可直接输入“拒绝，换成 LDA”或“/deny 理由”，一次提交拒绝与修改要求；也可继续提问。`, 33)}`);
  }
  activity(message: string): void {
    if (this.terminal.activity) this.terminal.activity(this.text(message));
    else this.terminal.write(this.line(`◌ ${message}`));
  }
  execution(event: ExecutionEvent, label = event.name): void {
    if (!this.callNumbers.has(event.callId)) this.callNumbers.set(event.callId, this.callNumbers.size + 1);
    const number = this.callNumbers.get(event.callId);
    const name = (event.kind === 'inference' ? `模型推理 · ${event.name}` : event.kind === 'provider' ? `HTTP 请求 · 尝试 ${event.attempt}` : label) + (event.detail ? ` · ${event.detail}` : '');
    if (event.status === 'started') {
      if (event.kind === 'provider') this.notice(`  ↳ #${number} ${name}`);
      else this.activity(`#${number} ${name}`);
      return;
    }
    if (event.kind !== 'provider') this.terminal.activity?.();
    const status = ({ completed: '✓', cached: '↪ 缓存', failed: '× 失败', cancelled: '■ 中断', started: '→' })[event.status];
    this.notice(`${status} #${number} ${name} · ${(event.durationMs / 1000).toFixed(1)}s`);
  }
  executionSummary(events: ExecutionEvent[]): void {
    const turn = events.at(-1)?.turnId;
    if (!turn) return;
    const current = events.filter(event => event.turnId === turn);
    this.notice('本轮调用记录（模型与 HTTP 耗时有重叠，不累加）');
    for (const event of current.filter(item => item.status !== 'started')) {
      this.notice(`${event.kind === 'provider' ? '  ↳' : '•'} ${event.name}${event.detail ? ` · ${event.detail}` : ''}${event.attempt ? ` · 尝试 ${event.attempt}` : ''} · ${event.status} · ${(event.durationMs / 1000).toFixed(1)}s`);
    }
  }
  trainingResult(job?: ComputeJob): void {
    if (job && ['completed', 'failed', 'cancelled'].includes(job.status) && job.resultDir) {
      const key = `${job.id}:${job.resultDir}`;
      if (!this.shownResultDirectories.has(key)) {
        this.shownResultDirectories.add(key);
        // Persist the worker's actual path in the transcript, independently of model inference.
        // Keep the path on one logical line so terminal wrapping does not break copying.
        this.terminal.write(`\n${job.status === 'completed' && !job.resultWarning ? '训练结果目录' : '已有训练产物目录（结果可能不完整）'}\n任务：${terminalText(job.id)}\n${terminalText(job.resultDir)}\n`);
      }
    }
  }
  training(job?: ComputeJob): void {
    this.trainingResult(job);
    this.trainingNotice(job ? trainingStatusText(job) : undefined);
  }
  trainingNotice(message?: string): void {
    this.terminal.trainingProgress?.(message ? this.line(message) : undefined);
  }
  notice(message: string): void { this.terminal.write(this.ink(this.line(message), 90)); }
  error(message: string): void { this.terminal.activity?.(); this.terminal.writeError(`\n${this.box('本轮需要处理', message, 33)}`); }
  sessions(sessions: Array<{ id: string; title: string; updatedAt: string }>, selected: string): void {
    this.terminal.write(`\n${this.ink('研究对话', 1)}`);
    for (const session of sessions) {
      this.terminal.write(this.line(`${session.id === selected ? '●' : '○'} ${session.title}`));
      this.terminal.write(this.ink(this.line(`  ${session.id}  ${session.updatedAt.slice(0, 10)}`), 90));
    }
    this.notice('输入 /resume 会话标识 继续研究。');
  }
}
