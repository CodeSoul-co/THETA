import { createInterface, type Interface } from 'node:readline/promises';
import { Writable } from 'node:stream';
import { clearScreenDown, cursorTo, moveCursor } from 'node:readline';

export interface InteractiveTerminal {
  readonly columns?: number;
  readonly isTTY?: boolean;
  activity?(message?: string): void;
  trainingProgress?(message?: string): void;
  onPendingInput?(listener: () => void): () => void;
  write(message: string): void;
  writeError(message: string): void;
  question(prompt: string): Promise<string>;
  secret?(prompt: string): Promise<string>;
  close(): void;
}

export class ReadlineInteractiveTerminal implements InteractiveTerminal {
  private readonly terminal: Interface;
  private muted = false;
  private closed = false;
  private rejectQuestion?: (error: Error) => void;
  private spinner?: ReturnType<typeof setInterval>;
  private spinnerLines = 0;
  private trainingMessage = '';
  private questionPrompt = '';
  private lastPlainProgress = 0;
  private readonly pendingLines: string[] = [];
  private pendingInputListener?: () => void;
  get columns(): number { return process.stdout.columns || 80; }
  get isTTY(): boolean { return Boolean(process.stdout.isTTY && process.stdin.isTTY && process.env.TERM !== 'dumb'); }

  constructor() {
    const output = new Writable({ write: (chunk, _encoding, done) => { if (!this.muted) process.stdout.write(chunk); done(); } });
    Object.defineProperty(output, 'columns', { get: () => this.columns });
    this.terminal = createInterface({ input: process.stdin, output, terminal: process.stdin.isTTY });
    this.terminal.on('line', (line: string) => {
      if (!this.rejectQuestion && !this.muted && line.trim()) {
        this.pendingLines.push(line); this.pendingInputListener?.();
      }
    });
    let historyBeforeSecret: string[] = [];
    this.terminal.on('history', (history) => {
      if (this.muted) history.splice(0, history.length, ...historyBeforeSecret);
      else historyBeforeSecret = [...history];
    });
    this.terminal.on('close', () => {
      this.closed = true;
      this.rejectQuestion?.(Object.assign(new Error('终端已关闭'), { code: 'THETA_TERMINAL_CLOSED' }));
    });
    this.terminal.on('SIGINT', () => { if (this.rejectQuestion) this.close(); else process.emit('SIGINT'); });
  }

  write(message: string): void {
    this.print(message, false);
  }

  writeError(message: string): void {
    this.print(message, true);
  }

  private clearSpinner(): void {
    if (this.spinnerLines) { moveCursor(process.stdout, 0, 1 - this.spinnerLines); cursorTo(process.stdout, 0); clearScreenDown(process.stdout); this.spinnerLines = 0; }
  }

  private print(message: string, error: boolean): void {
    this.clearSpinner();
    // Background training notifications must not overwrite an in-progress input.
    const restore = this.isTTY && this.rejectQuestion && !this.muted;
    let rows = 0;
    if (restore) {
      rows = this.terminal.getCursorPos().rows;
      moveCursor(process.stdout, 0, -rows); cursorTo(process.stdout, 0); clearScreenDown(process.stdout);
    }
    (error ? process.stderr : process.stdout).write(`${message}\n`);
    if (restore) {
      // Readline still owns the previous cursor offset. Reserve that many rows below
      // the notification so its redraw moves back to the new prompt, not into history.
      process.stdout.write('\n'.repeat(rows));
      this.terminal.prompt(true);
    }
  }

  trainingProgress(message = ''): void {
    this.trainingMessage = message;
    if (!this.isTTY) {
      if (message && Date.now() - this.lastPlainProgress >= 15000) {
        this.lastPlainProgress = Date.now(); this.write(message);
      }
      return;
    }
    if (this.rejectQuestion && !this.muted) {
      // prompt(true) already moves to the old prompt's start and clears it.
      // Doing this manually as well erases transcript rows on every heartbeat.
      this.terminal.setPrompt(this.withProgress(this.questionPrompt));
      this.terminal.prompt(true);
    }
  }

  private withProgress(prompt: string): string {
    return this.trainingMessage && !this.muted ? `${this.trainingMessage}\n${prompt}` : prompt;
  }

  activity(message?: string): void {
    if (this.spinner) clearInterval(this.spinner);
    this.spinner = undefined; this.clearSpinner();
    if (!message) return;
    if (!this.isTTY) { this.write(`  → ${message}`); return; }
    const started = Date.now(); let frame = 0;
    const frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
    const draw = (): void => {
      if (this.rejectQuestion || this.closed) return;
      this.clearSpinner();
      // Keep the live line short enough to avoid terminal wrapping.
      const label = [...message].slice(0, Math.max(1, Math.floor((this.columns - 24) / 2))).join('');
      const text = `${frames[frame++ % frames.length]} ${label} · ${Math.floor((Date.now() - started) / 1000)}s`;
      process.stdout.write(process.env.NO_COLOR === undefined ? `\x1b[36m${text}\x1b[0m` : text);
      if (this.trainingMessage) process.stdout.write(`\n${this.trainingMessage}`);
      this.spinnerLines = 1 + (this.trainingMessage ? this.trainingMessage.split('\n').length : 0);
    };
    draw(); this.spinner = setInterval(draw, 100); this.spinner.unref();
  }

  async question(prompt: string): Promise<string> {
    this.activity();
    if (this.pendingLines.length) return this.pendingLines.shift()!;
    if (this.closed) throw Object.assign(new Error('终端已关闭'), { code: 'THETA_TERMINAL_CLOSED' });
    this.questionPrompt = prompt;
    try {
      return await Promise.race([this.terminal.question(this.withProgress(prompt)), new Promise<string>((_resolve, reject) => { this.rejectQuestion = reject; })]);
    } finally { this.rejectQuestion = undefined; this.questionPrompt = ''; }
  }
  onPendingInput(listener: () => void): () => void {
    this.pendingInputListener = listener;
    return () => { this.pendingInputListener = undefined; };
  }

  async secret(prompt: string): Promise<string> {
    process.stdout.write(prompt);
    this.muted = true;
    try { return await this.question(''); }
    finally { this.muted = false; process.stdout.write('\n'); }
  }

  close(): void {
    this.activity();
    this.terminal.close();
  }
}

export const askChoice = async (
  terminal: InteractiveTerminal,
  prompt: string,
  allowed: readonly string[],
): Promise<string> => {
  const choices = new Set(allowed);
  while (true) {
    const answer = (await terminal.question(prompt)).trim();
    if (choices.has(answer)) return answer;
    terminal.writeError(`请输入 ${allowed.join('、')} 中的一个选项。`);
  }
};

export const askNonEmpty = async (
  terminal: InteractiveTerminal,
  prompt: string,
): Promise<string> => {
  while (true) {
    const answer = (await terminal.question(prompt)).trim();
    if (answer) return answer;
    terminal.writeError('请输入内容，或输入 /cancel 返回。');
  }
};
