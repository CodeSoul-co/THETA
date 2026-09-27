export interface ReportExhibit { id: string; kind: 'figure' | 'table'; title: string; scope: string; block: string; [key: string]: unknown }

/** Host owns chapter numbering; models may return a redundant heading or wrapper. */
export function normalizeChapter(text: string, title: string): string {
  text = text.trim().replace(/^```(?:markdown|md)?\s*\n([\s\S]*?)\n```\s*$/i, '$1');
  const chapterName = title.replace(/^\d+\.\s*/, '');
  return text.split('\n').filter(line => {
    const label = line.trim().replace(/^#{1,6}\s*/, '').replace(/^\*\*|\*\*$/g, '').replace(/^(?:第[一二三四五\d]+章\s*|\d+[.、．]?\s*)/, '');
    return label !== chapterName && label !== 'THETA 完整分析报告';
  }).map(line => line.replace(/^#{1,2}\s+/, '### ')).join('\n').trim();
}

export function citationProblem(text: string, catalog: ReportExhibit[], requireEvidence: boolean, inserted: ReadonlySet<string> = new Set()): string | undefined {
  const ids = new Set(catalog.map(item => item.id));
  const cited = [...text.matchAll(/【((?:图|表)\d+)】/g)].map(match => match[1]);
  const mentioned = [...text.matchAll(/(?:图|表)\s*\d+/g)].map(match => match[0].replace(/\s/g, ''));
  if (mentioned.some(id => !ids.has(id))) return '引用了不存在的图表编号，只能使用证据目录中给出的编号。';
  if (/!\[[^\]]*\]\(|^\s*\|/m.test(text)) return '不要自行生成图片路径或表格；实际图表由宿主按独立一行的 [[图1]] 或 [[表1]] 插入。';
  if (requireEvidence) for (const kind of ['figure', 'table']) {
    if (catalog.some(item => item.kind === kind) && !catalog.some(item => item.kind === kind && cited.includes(item.id))) return '本章节需用【图N】和【表N】引用现有证据，并解释具体数值、比较与研究问题的关系，不能只罗列编号。';
  }
  const placed = new Set<string>();
  const paragraphs = text.trim().split(/\n\s*\n/);
  for (let i = 0; i < paragraphs.length; i++) {
    const paragraph = paragraphs[i].trim();
    if (!paragraph.includes('[[')) continue;
    const marker = paragraph.match(/^\[\[((?:图|表)\d+)\]\]$/);
    if (!marker) return '每个图表占位符必须独立成段，每处只放一张图或一张表。';
    const id = marker[1];
    if (!ids.has(id) || !cited.includes(id)) return '插入图表前，先在论述中使用其【编号】说明它支撑的具体判断。';
    if (inserted.has(id) || placed.has(id)) return '已经展示过的图表只在正文引用，不要重复插入。';
    const before = paragraphs[i - 1]?.trim() || '', after = paragraphs[i + 1]?.trim() || '';
    if ([before, after].some(part => part.length < 60 || /^(?:#|\[\[|\||[-*] )/.test(part))) return '每张图或表前后都需要实质性分析段落（至少 60 字）：前文提出判断，后文分析具体数据、解释及研究含义。不得相邻堆放图表、用一句图注替代解读或在章末罗列。';
    placed.add(id);
  }
  if (cited.some(id => !inserted.has(id) && !placed.has(id))) return '首次引用的图表需要在合适的论证段落之间，以独立一行的 [[图N]] 或 [[表N]] 插入；插入后继续深入解读，不能只引用编号或集中罗列。';
}

/** Placement is explicit: prose, one exhibit, then substantive interpretation. */
export function insertExhibits(text: string, catalog: ReportExhibit[], inserted: Set<string>): string {
  const byId = new Map(catalog.map(item => [item.id, item]));
  return text.replace(/^\[\[((?:图|表)\d+)\]\]$/gm, (_, id: string) => {
    const exhibit = byId.get(id);
    if (!exhibit || inserted.has(id)) return '';
    inserted.add(id);
    return exhibit.block;
  }).replace(/【((?:图|表)\d+)】/g, '$1');
}
