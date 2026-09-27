"""Numbered report exhibits copied from hash-verified training presentation files."""
import csv
import hashlib
import math
from statistics import mean
from io import StringIO
from pathlib import Path

# The native visualizer writes these exact CSV/figure pairs. No pixel inference.
EXHIBITS = [
    ('evaluation_metrics', '模型评估指标', ('7项核心指标', '7 Core Metrics')),
    ('topic_proportions', '平均主题权重', ('主题占比分布', 'Topic Proportion Distribution')),
    ('training_curves', '训练损失', ('训练损失', 'Training Loss')),
    ('主题表', '主题关键词', ('topic_table_1',)),
    ('topic_coherence', '主题一致性', ('主题一致性', 'Topic Coherence')),
    ('topic_exclusivity', '主题排他性', ('主题排他性', 'Topic Exclusivity')),
]
LABELS = {'all': '全量', 'train': '训练集', 'validation': '验证集', 'test': '测试集'}


def collect_exhibits(summary, result_root, directory):
    root = Path(summary['reportPath']).parent / 'native' if summary.get('reportPath') else result_root
    root = root.resolve()
    # Native report paths originate from the already-verified result manifest.
    verified = {item['relativePath']: item['sha256'] for key in ('tables', 'figures') for item in summary.get(key, [])}
    # presentation_evidence is bounded; additional CSVs still have to be recorded
    # in a verified native manifest, or belong to the immutable training tree.
    if summary.get('reportPath'):
        import json
        manifest = json.loads((root.parent / 'manifest.json').read_text(encoding='utf-8'))
        verified = {Path(item['path']).resolve().relative_to(root).as_posix(): item['sha256']
                    for item in manifest['files'] if Path(item['path']).resolve().is_relative_to(root)}
    assets = directory / 'assets'; assets.mkdir(parents=True, exist_ok=True)
    catalog = []
    def read(file):
        if file.is_symlink() or not file.resolve().is_relative_to(root):
            raise ValueError('报告图表路径超出本任务范围')
        raw = file.read_bytes()
        expected = verified.get(file.relative_to(root).as_posix())
        if expected and hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('报告图表在读取期间发生变化')
        if summary.get('reportPath') and not expected:
            raise ValueError('报告图表不在已验证清单中')
        return raw
    def cell(value):
        return str(value or '').replace('|', '／').replace('\n', ' ').replace('`', '').replace('*', '')[:150]
    tables = sorted(root.rglob('*.csv'))
    figures = sorted(root.rglob('*.png'))
    figure_number = table_number = 0
    for group in LABELS:
        for stem, title, image_names in EXHIBITS if group == 'all' else EXHIBITS[:2]:
            def matches(file):
                parts = file.relative_to(root).parts
                if 'en' in parts and any('zh' in x.relative_to(root).parts for x in tables): return False
                if group == 'all': return not any(role in parts for role in ('train', 'validation', 'test'))
                return group in parts
            candidates = [f for f in tables if f.stem in ({'主题表', 'topic_table'} if stem == '主题表' else {stem}) and matches(f)]
            if len(candidates) != 1: continue
            source = candidates[0]
            if source.stat().st_size > 4 * 1024 * 1024: continue
            raw = read(source)
            reader = csv.DictReader(StringIO(raw.decode('utf-8-sig')))
            rows = list(reader); columns = (reader.fieldnames or [])[:6]
            if not rows or not columns: continue
            table_number += 1; identity = f'表{table_number}'
            name = f'assets/table-{table_number}.csv'; (directory / name).write_bytes(raw)
            scope = LABELS[group]
            # Shared training history is not a held-out loss curve.
            if stem == 'training_curves': scope = '训练过程（范围以记录字段为准）'
            display = rows[:12]
            caption = f'{identity}　{title}（{scope}）'
            block = f'**{caption}**\n\n| ' + ' | '.join(map(cell, columns)) + ' |\n| ' + ' | '.join('---' for _ in columns) + ' |\n'
            block += '\n'.join('| ' + ' | '.join(cell(row.get(col)) for col in columns) + ' |' for row in display)
            note = f'来源：{source.relative_to(root).as_posix()}。展示前 {len(display)} / {len(rows)} 行、{len(columns)} / {len(reader.fieldnames)} 列；数值保持原始口径。'
            block += f'\n\n{note} [完整数据表]({name})\n'
            statistics = {}
            # Only summarize columns with one consistent quantity. Aggregating
            # evaluation_metrics.value would mix unrelated metrics and units.
            numeric_columns = [c for c in columns if c not in {'topic_id', 'topic', 'epoch'}] if stem in {'topic_proportions', 'training_curves'} else []
            for column in numeric_columns:
                numbers = []
                for row in rows:
                    try: value = float(row.get(column))
                    except (TypeError, ValueError): continue
                    if math.isfinite(value): numbers.append(value)
                if numbers: statistics[column] = {'count': len(numbers), 'min': min(numbers), 'max': max(numbers), 'mean': mean(numbers), 'first': numbers[0], 'last': numbers[-1]}
            item = {'statistics': statistics, 'id': identity, 'kind': 'table', 'title': title, 'scope': scope, 'source': source.relative_to(root).as_posix(),
                    'sha256': hashlib.sha256(raw).hexdigest(), 'rows': display, 'rowCount': len(rows), 'columns': columns, 'block': block}
            catalog.append(item)
            images = [f for f in figures if f.parent == source.parent and f.stem in image_names]
            if not images: continue
            file = images[0]; image_raw = read(file)
            figure_number += 1; identity = f'图{figure_number}'
            name = f'assets/figure-{figure_number}.png'; (directory / name).write_bytes(image_raw)
            caption = f'{identity}　{title}（{scope}）'
            catalog.append({'id': identity, 'kind': 'figure', 'title': title, 'scope': scope,
                            'source': file.relative_to(root).as_posix(), 'sha256': hashlib.sha256(image_raw).hexdigest(),
                            'dataTable': item['id'], 'basis': '同目录的原生图表及其绘图数据；依据表中数值分析，不推测未读取的图像特征。',
                            'block': f'![{caption}]({name})\n\n**{caption}**\n\n来源：{file.relative_to(root).as_posix()}。[绘图原始数据](assets/table-{table_number}.csv)\n'})
    # Protect the non-native fallback from mutation during reads/copying as well.
    if not summary.get('reportPath'):
        from .results_reader import tree_hash
        if summary.get('_resultHash') and tree_hash(result_root) != summary['_resultHash']:
            raise ValueError('训练结果在读取图表期间发生变化')
    return catalog
