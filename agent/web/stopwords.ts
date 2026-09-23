import { readFileSync, readdirSync } from 'node:fs';
import path from 'node:path';

export const STOPWORD_BYTES = 512 * 1024;
export function parseStopwords(bytes: Uint8Array): string[] {
  if (bytes.byteLength > STOPWORD_BYTES) throw new Error('停用词表不能超过 512 KB');
  let text: string;
  try { text = new TextDecoder('utf-8', { fatal: true }).decode(bytes); }
  catch { throw new Error('请上传 UTF-8 编码的 TXT 停用词表'); }
  if (/[\x00-\x08\x0b\x0c\x0e-\x1f]/u.test(text)) throw new Error('停用词表包含无效字符');
  const words = [...new Set(text.split(/\r?\n/u).map(line => line.normalize('NFKC').trim().toLowerCase()).filter(line => line && !line.startsWith('#')))];
  if (!words.length) throw new Error('停用词表为空，请每行填写一个词');
  if (words.length > 20000 || words.some(word => word.length > 100)) throw new Error('停用词表最多 20000 个词，每个词不超过 100 个字符');
  return words;
}

/** Export the actual engine resources, not a second frontend-maintained list. */
export function exportBuiltinStopwords(root: string): string {
  const folder = path.join(root, 'src/models/resources/stopwords');
  return '# THETA 内置停用词表：各语言与通用词表合集\n# UTF-8，每行一个词；以 # 开头的行是注释。\n'
    + readdirSync(folder).filter(name => /^[a-z]+\.txt$/u.test(name)).sort().map(name => `\n# ${name}\n${readFileSync(path.join(folder, name), 'utf8').trim()}\n`).join('');
}
