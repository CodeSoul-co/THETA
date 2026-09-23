import { randomUUID } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync, renameSync } from 'node:fs';
import path from 'node:path';
import { ResearchStore } from './research-store.js';

export interface ResearchContext {
  id: string; owner: string; title: string; goal: string; understanding: string;
  questions: string[]; datasetRefs: string[]; updatedAt: string; sourceContextId?: string;
}
export class ResearchContexts {
  constructor(private readonly records: ResearchStore) {}
  private file(id: string): string {
    if (!/^context-[a-f0-9-]{36}$/u.test(id)) throw new Error('无效研究上下文标识');
    return path.join(this.records.directory, 'contexts', `${id}.md`);
  }
  list(): Array<Pick<ResearchContext, 'id' | 'title' | 'goal' | 'updatedAt'>> {
    return this.records.list<ResearchContext>('research-context').map(({ id, title, goal, updatedAt }) => ({ id, title, goal, updatedAt }));
  }
  read(id: string): ResearchContext & { markdown: string; documentPath: string } {
    const context = this.records.get<ResearchContext>('research-context', id);
    const file = this.file(id);
    return { ...context, markdown: readFileSync(file, 'utf8').slice(0, 18000), documentPath: file };
  }
  save(owner: string, value: Omit<ResearchContext, 'id' | 'owner' | 'updatedAt'>, id?: string): ResearchContext & { documentPath: string } {
    if (id && this.read(id).owner !== owner) throw new Error('请先选用该上下文，在本会话的副本中更新');
    const context = { ...value, title: value.title.slice(0, 200), goal: value.goal.slice(0, 2000),
      understanding: value.understanding.slice(0, 14000), questions: value.questions.slice(0, 10).map(q => q.slice(0, 500)),
      id: id ?? `context-${randomUUID()}`, owner, updatedAt: new Date().toISOString() };
    const file = this.file(context.id); mkdirSync(path.dirname(file), { recursive: true });
    const text = `# ${context.title}\n\n> 研究工作笔记：业务解释与待确认假设，不是训练授权或工具指令。\n\n## 用户目标\n\n${context.goal}\n\n## 数据与业务理解\n\n${context.understanding}\n\n## 待确认问题\n\n${context.questions.map(q => `- ${q}`).join('\n') || '暂无记录；不代表所有假设已经核实。'}\n\n## 数据依据\n\n${context.datasetRefs.map(ref => `- ${ref}`).join('\n')}\n\n更新时间：${context.updatedAt}\n`;
    const temp = `${file}.${randomUUID()}.tmp`; writeFileSync(temp, text, { mode: 0o600 }); renameSync(temp, file);
    this.records.put('research-context', context.id, context);
    return { ...context, documentPath: file };
  }
  select(owner: string, id: string): ResearchContext & { documentPath: string } {
    const source = this.read(id);
    return this.save(owner, { title: source.title, goal: source.goal, understanding: source.understanding,
      questions: source.questions, datasetRefs: source.datasetRefs, sourceContextId: id });
  }
}
