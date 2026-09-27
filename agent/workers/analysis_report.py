"""On-demand evidence and portable report export. Never starts model training."""
import html
import json
import re
from pathlib import Path


def evidence(payload):
    from .local_compute import database, results
    from .capabilities import dataset_profile, verify_dataset
    from .dataset.business import understand
    # This validates the immutable result hash before reading any analysis evidence.
    summary = results({**payload, 'view': 'summary'})
    with database(payload['home']) as db:
        row = db.execute('SELECT request,value FROM jobs WHERE id=?', (payload['jobId'],)).fetchone()
    request, job = map(json.loads, row)
    dataset, plan = request['dataset'], request['plan']
    # Independent uploads must each appear in the report; the primary descriptor
    # alone refers to the training file and is not a profile of the full corpus.
    options = plan.get('dataSplit') or {}
    sources = options.get('sources') if options.get('enabled') and options.get('mode') == 'upload' else None
    sources = sources or {'all': {'dataset': dataset, 'textColumn': plan['textColumn']}}
    labels = {'all': '全量来源', 'train': '训练来源', 'validation': '验证来源', 'test': '测试来源'}
    lines = ['## 1. 数据描述性统计', '']
    profiles, contexts = {}, {}
    def cell(value):
        return str(value if value is not None else '未记录').replace('|', '\uFF5C').replace('\n', ' ')
    for role, source in sources.items():
        descriptor = source['dataset']
        profile = dataset_profile({'dataset': descriptor})
        profiles[role] = profile
        contexts[role] = understand({'dataset': descriptor, 'column': source.get('textColumn') or plan['textColumn']})
        policy = profile.get('samplePolicy', {})
        lines += [f'### {labels.get(role, role)}', '', '| 项目 | 数值 |', '| --- | --- |',
                  f'| 数据文件 | {cell(descriptor["fileName"])} |',
                  f'| 原始记录数 | {profile.get("rowCount", "未记录")} |',
                  f'| 字段数 | {len(profile.get("columns", []))} |',
                  f'| 正文字段 | {cell(source.get("textColumn") or plan["textColumn"])} |',
                  f'| 字段统计样本数 | {policy.get("profileRows", "未记录")} |', '',
                  ('字段统计来自确定性抽样，样本统计不等同于全量统计。' if policy.get('profileTruncated') else '字段统计覆盖本文件全部记录。'), '',
                  '| 字段 | 类型 | 非空数 | 缺失比例 | 平均字符数 |', '| --- | --- | --- | --- | --- |']
        for col in profile.get('columnProfiles', []):
            lines.append('| ' + ' | '.join(cell(col.get(k)) for k in ('name', 'inferredType', 'nonEmptyCount', 'missingRatio', 'averageLength')) + ' |')
        numeric = numeric_description(descriptor, profile)
        profiles[role]['numericDescription'] = numeric
        if numeric:
            lines += ['', '| 数值字段 | 有限值数 | 均值 | 样本标准差 | 最小值 | 中位数 | 最大值 |', '| --- | --- | --- | --- | --- | --- | --- |']
            lines += ['| ' + ' | '.join(cell(item.get(k)) for k in ('name', 'count', 'mean', 'std', 'min', 'median', 'max')) + ' |' for item in numeric]
        verify_dataset(descriptor)
        lines.append('')
    root = Path(job['resultDir'])
    split_files = list(root.rglob('split_results.json'))
    if len(split_files) > 1:
        raise ValueError('结果包含多个划分清单，请选择具体训练任务后生成报告')
    splits = json.loads(split_files[0].read_text(encoding='utf-8')) if split_files else None
    if splits:
        lines += ['| 分组 | 有效正文数 |', '| --- | --- |']
        group_labels = {'all': '全量', 'train': '训练', 'validation': '验证', 'test': '测试'}
        lines += [f'| {group_labels.get(key, key)} | {group["count"]} |' for key, group in splits['groups'].items()]
        if splits.get('testUsesAllData'):
            lines += ['', '**测试集使用全量数据，包含训练及验证记录，不是独立留出测试。**']
    return {'description': '\n'.join(lines), 'dataset': dataset, 'profiles': profiles,
            'plan': plan, 'context': contexts, 'summary': summary, 'splits': splits,
            'resultHash': job['resultHash'], 'jobId': job['id']}


def numeric_description(dataset, profile):
    import math
    from statistics import mean, median, stdev
    from .capabilities import verify_dataset
    from .dataset.readers import load_dataset
    columns = [item['name'] for item in profile.get('columnProfiles', []) if item['inferredType'] == 'number']
    if not columns: return []
    # Identical seed and reservoir size to dataset.profile; explicit sample scope.
    table = load_dataset(verify_dataset(dataset), seed=profile['sampleSeed'])
    output = []
    for name in columns:
        values = []
        for row in table.rows:
            try: value = float(row.get(name))
            except (TypeError, ValueError): continue
            if math.isfinite(value): values.append(value)
        if values:
            rounded = lambda value: float(f'{value:.8g}')
            output.append({'name': name, 'count': len(values), 'mean': rounded(mean(values)),
                           'std': rounded(stdev(values)) if len(values) > 1 else None,
                           'min': min(values), 'median': median(values), 'max': max(values)})
    return output


def markdown_html(markdown):
    """Small safe renderer for the report contract; never loads external resources."""
    blocks, table, paragraph = [], [], []
    def inline(text):
        text = html.escape(text)
        text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
        text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
        return text
    def flush():
        if paragraph:
            blocks.append('<p>' + '<br>'.join(map(inline, paragraph)) + '</p>'); paragraph.clear()
        if table:
            rows = [re.split(r'(?<!\\)\|', line.strip().strip('|')) for line in table]
            rows = [row for row in rows if not all(re.fullmatch(r'\s*:?-+:?\s*', x) for x in row)]
            blocks.append('<table>' + ''.join('<tr>' + ''.join(f'<{"th" if i == 0 else "td"}>' + inline(x.strip().replace('\\|', '|')) + f'</{"th" if i == 0 else "td"}>' for x in row) + '</tr>' for i, row in enumerate(rows)) + '</table>'); table.clear()
    fenced = False
    for line in markdown.splitlines():
        if line.startswith('```'):
            flush(); fenced = not fenced; continue
        if not fenced and line.startswith('|'):
            if paragraph: flush()
            table.append(line); continue
        if table: flush()
        heading = re.match(r'^(#{1,4})\s+(.+)', line) if not fenced else None
        if heading:
            flush(); level = len(heading[1]); blocks.append(f'<h{level}>' + inline(heading[2]) + f'</h{level}>')
        elif not line.strip(): flush()
        else: paragraph.append(line)
    flush()
    return '<html><body>' + ''.join(blocks) + '</body></html>'


def export_pdf(payload):
    import pymupdf
    directory = Path(payload['directory']).resolve()
    markdown = (directory / 'analysis.md').read_text(encoding='utf-8')
    target = directory / 'analysis.pdf'
    temporary = directory / 'analysis.partial.pdf'
    css = 'body{font-family:sans-serif;font-size:10pt;line-height:1.6;color:#172338}h1{font-size:22pt}h2{font-size:15pt;margin-top:22pt}h3{font-size:12pt}table{width:100%;border-collapse:collapse;margin:12pt 0;font-size:8pt}td,th{border:0.5pt solid #cdd5df;padding:5pt;text-align:left}th{background:#eef2f8}code{font-size:9pt}p{overflow-wrap:anywhere}'
    story = pymupdf.Story(html=markdown_html(markdown), user_css=css)
    box = pymupdf.paper_rect('a4'); content = box + (42, 42, -42, -42)
    writer = pymupdf.DocumentWriter(str(temporary))
    try:
        more, pages = True, 0
        while more:
            pages += 1
            if pages > 150: raise ValueError('报告超过 150 页，请缩短内容后重试')
            device = writer.begin_page(box)
            more, _ = story.place(content)
            story.draw(device); writer.end_page()
    finally: writer.close()
    with pymupdf.open(temporary) as pdf:
        if not len(pdf) or not ''.join(page.get_text() for page in pdf).strip():
            raise ValueError('PDF 未生成可读取的正文')
        for i, page in enumerate(pdf):
            page.insert_text((42, box.height - 22), f'THETA | {i + 1} / {len(pdf)}', fontsize=8, color=(0.4, 0.45, 0.5))
        pdf.save(target)
    temporary.unlink(missing_ok=True)
    return {'pages': pages, 'file': str(target)}
