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
    summary = results({**payload, 'view': 'analysis'})
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
    from .report_exhibits import collect_exhibits
    summary['_resultHash'] = job['resultHash']
    catalog = collect_exhibits(summary, root, Path(payload['directory'])) if payload.get('directory') else []
    # Large matrix excerpts can crowd out the actual figures and tabular evidence.
    summary = {key: value for key, value in summary.items() if key not in {'matrices', 'figures', 'tables', 'availableArtifacts'}}
    return {'catalog': catalog, 'description': '\n'.join(lines), 'dataset': dataset, 'profiles': profiles,
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


def markdown_html(markdown, assets=None):
    """CommonMark + tables, with HTML disabled and only verified local images."""
    from markdown_it import MarkdownIt
    renderer = MarkdownIt('commonmark', {'html': False, 'breaks': False}).enable('table')
    assets = assets or {}
    def image(tokens, index, options, env):
        token = tokens[index]
        src = token.attrGet('src')
        if src not in assets:
            return '<span>[图片未包含在报告中]</span>'
        width, height = assets[src]
        # Bound both dimensions to keep a figure and caption readable on A4.
        scale = min(500 / width, 430 / height, 1)
        return f'<img src="{html.escape(src, quote=True)}" width="{int(width * scale)}" height="{int(height * scale)}" />'
    renderer.renderer.rules['image'] = image
    # Source links are readable text in the PDF; no network or file URI navigation.
    renderer.renderer.rules['link_open'] = lambda *args: '<span>'
    renderer.renderer.rules['link_close'] = lambda *args: '</span>'
    return '<html><body>' + renderer.render(markdown) + '</body></html>'


def report_blocks(markdown):
    """Keep captions with exhibits and repeat headers for long source tables."""
    from markdown_it import MarkdownIt
    tokens = MarkdownIt('commonmark', {'html': False}).enable('table').parse(markdown)
    lines = markdown.splitlines()
    blocks = []
    for token in tokens:
        if token.level != 0 or not token.map or token.type.endswith('_close'): continue
        block = '\n'.join(lines[token.map[0]:token.map[1]])
        if token.type == 'table_open':
            rows = block.splitlines()
            for offset in range(2, len(rows), 14):
                blocks.append(('table', '\n'.join(rows[:2] + rows[offset:offset + 14])))
        else: blocks.append((token.type, block))
    groups = []
    i = 0
    while i < len(blocks):
        kind, block = blocks[i]; i += 1
        if block.startswith('!['):
            while i < len(blocks) and (blocks[i][1].startswith('**图') or blocks[i][1].startswith('来源：')):
                block += '\n\n' + blocks[i][1]; i += 1
        elif block.startswith('**表') and i < len(blocks) and blocks[i][0] == 'table':
            block += '\n\n' + blocks[i][1]; i += 1
        elif kind == 'heading_open' and i < len(blocks):
            block += '\n\n' + blocks[i][1]; i += 1
        groups.append(block)
    return groups


def export_pdf(payload):
    import pymupdf
    from io import BytesIO
    directory = Path(payload['directory']).resolve()
    markdown = (directory / 'analysis.md').read_text(encoding='utf-8')
    target = directory / 'analysis.pdf'
    buffer = BytesIO()
    assets = {}
    archive = pymupdf.Archive()
    for file in sorted((directory / 'assets').glob('*.png')):
        if file.is_symlink(): raise ValueError('报告图片不能是符号链接')
        pix = pymupdf.Pixmap(file.read_bytes())
        name = 'assets/' + file.name
        assets[name] = (pix.width, pix.height)
        archive.add((file.read_bytes(), name))
    css = 'body{font-family:sans-serif;font-size:10pt;line-height:1.6;color:#172338}h1{font-size:22pt}h2{font-size:15pt;margin-top:22pt;page-break-after:avoid}h3{font-size:12pt;page-break-after:avoid}h4{font-size:10pt;page-break-after:avoid}img{display:block;margin:8pt auto}thead{display:table-header-group}tr{page-break-inside:avoid}table{page-break-inside:avoid;width:100%;border-collapse:collapse;margin:12pt 0;font-size:8pt}td,th{border:0.5pt solid #cdd5df;padding:5pt;text-align:left}th{font-weight:bold}code{font-size:9pt}p{overflow-wrap:anywhere}'
    box = pymupdf.paper_rect('a4'); content = box + (42, 42, -42, -42)
    writer = pymupdf.DocumentWriter(buffer)
    pages, device, y = 1, writer.begin_page(box), content.y0
    try:
        for block in report_blocks(markdown):
            rendered = markdown_html(block, assets)
            story = pymupdf.Story(html=rendered, user_css=css, archive=archive)
            more, filled = story.place(pymupdf.Rect(content.x0, y, content.x1, content.y1))
            # Tables and image/caption groups start on a fresh page if they do
            # not fit. CSS page-break-inside is not honored by all Story builds.
            if more and y > content.y0:
                writer.end_page(); pages += 1; device = writer.begin_page(box); y = content.y0
                story = pymupdf.Story(html=rendered, user_css=css, archive=archive)
                more, filled = story.place(content)
            while True:
                if pages > 150: raise ValueError('报告超过 150 页，请缩短内容后重试')
                story.draw(device)
                if not more:
                    y = filled[3] + 5
                    break
                writer.end_page(); pages += 1; device = writer.begin_page(box); y = content.y0
                more, filled = story.place(content)
        writer.end_page()
    finally: writer.close()
    with pymupdf.open(stream=buffer.getvalue(), filetype='pdf') as pdf:
        if not len(pdf) or not ''.join(page.get_text() for page in pdf).strip():
            raise ValueError('PDF 未生成可读取的正文')
        for i, page in enumerate(pdf):
            page.insert_text((42, box.height - 22), f'THETA | {i + 1} / {len(pdf)}', fontsize=8, color=(0.4, 0.45, 0.5))
        # Write after rendering to memory: DocumentWriter can retain native file
        # handles on Windows even after close(), preventing temporary-file removal.
        temporary = directory / 'analysis.partial.pdf'
        temporary.write_bytes(pdf.tobytes(garbage=3, deflate=True))
    temporary.replace(target)
    import zipfile
    with zipfile.ZipFile(directory / 'analysis.zip', 'w', zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(directory / 'analysis.md', 'analysis.md')
        for file in sorted((directory / 'assets').glob('*')):
            if file.is_file() and not file.is_symlink() and file.suffix in {'.png', '.csv'}:
                bundle.write(file, 'assets/' + file.name)
    return {'pages': pages, 'file': str(target)}
