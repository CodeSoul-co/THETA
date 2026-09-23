import { stat } from 'node:fs/promises';
import { homedir } from 'node:os';
import path from 'node:path';

export interface AttachmentInput { filePath: string; instruction: string }
const decode = (value: string): string => {
  const unescaped = value.replace(/\\([ ()"'\\])/gu, '$1');
  return path.resolve(unescaped.startsWith('~/') ? path.join(homedir(), unescaped.slice(2)) : unescaped);
};
export async function parseAttachmentInput(input: string): Promise<AttachmentInput | undefined> {
  const explicit = /^\/attach\s/u.test(input);
  const text = (explicit ? input.replace(/^\/attach\s+/u, '') : input).trim();
  const valid = async (candidate: string, instruction: string): Promise<AttachmentInput | undefined> => {
    const filePath = decode(candidate);
    if (!/\.(csv|tsv|txt|jsonl?|xlsx?|parquet)$/iu.test(filePath)) return undefined;
    try { if ((await stat(filePath)).isFile()) return { filePath, instruction: instruction.trim().replace(/^--\s*/u, '') }; } catch {}
    return undefined;
  };
  const quoted = text.match(/^(["'])((?:\\.|(?!\1).)*)\1(?:\s+([\s\S]*))?$/u);
  if (quoted) {
    const value = await valid(quoted[2], quoted[3] ?? ''); if (value) return value;
  } else {
    const whole = await valid(text, ''); if (whole) return whole;
    const endings = [...text.matchAll(/\.(?:csv|tsv|txt|jsonl?|xlsx?|parquet)(?=\s)/giu)].reverse();
    for (const match of endings) {
      const end = match.index! + match[0].length;
      const value = await valid(text.slice(0, end), text.slice(end)); if (value) return value;
    }
  }
  if (explicit) throw new Error('未找到该数据文件。可输入 /attach "完整文件路径" 后附加一句指令。');
  return undefined;
}
