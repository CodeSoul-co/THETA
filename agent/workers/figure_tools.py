"""Adjust selected figures through the same native code that produced the report."""
from contextlib import redirect_stdout, redirect_stderr
import json
from pathlib import Path
import sys
from uuid import uuid4

from .capabilities import engine_root
from .results_reader import file_hash, tree_hash


def adjust(payload: dict) -> dict:
    import matplotlib
    matplotlib.use('Agg')
    sys.path.insert(0, str(engine_root() / 'src/models'))
    from visualization import run_visualization as native
    from visualization.publication import figure_edit

    report_dir = Path(payload['reportDir']).resolve()
    figure = Path(payload['figure'])
    figure = (figure if figure.is_absolute() else report_dir / figure).resolve()
    if report_dir not in figure.parents or not figure.is_file():
        raise ValueError('选中图不属于该报告，未生成替代图')
    spec = payload.get('spec') or {}
    if set(spec) != {'title'} or not isinstance(spec['title'], str) or not 1 <= len(spec['title'].strip()) <= 200:
        raise ValueError('当前原生绘图调整支持 title 参数；不接受 CSV 或 kind 重建替代图')
    manifest = report_dir / 'manifest.json'
    if not manifest.is_file():
        raise ValueError('缺少原生绘图记录，请先通过 results_read 重新整理原生报告；未生成替代图')
    report = json.loads(manifest.read_text())
    if not report.get('resultDir') or not report.get('nativeRunner'):
        raise ValueError('缺少原生绘图记录，请先通过 results_read 重新整理原生报告；未生成替代图')
    original = figure
    # Repeated edits replay the same native recipe, not an adjusted raster image.
    if report_dir / 'adjustments' in figure.parents:
        receipt = json.loads(figure.with_suffix('.json').read_text())
        original = (report_dir / receipt['originalFigure']).resolve()
    native_root = report_dir / 'native'
    if native_root not in original.parents:
        raise ValueError('缺少选中图的原生绘图记录，未生成替代图')
    relative = original.relative_to(native_root)
    matches = []
    for name in ('chart-status.json', 'additional-chart-status.json'):
        status_file = native_root / name
        if not status_file.is_file():
            continue
        for item in json.loads(status_file.read_text()):
            for output in item.get('files', []):
                path = Path(output)
                path = (path if path.is_absolute() else native_root / path).resolve()
                if path == original and item.get('status') == 'generated':
                    matches.append(item['chart'])
    if len(set(matches)) != 1:
        raise ValueError('无法唯一定位选中图的原生绘图函数，未生成替代图')
    chart = matches[0]
    root = Path(report['resultDir'])
    if tree_hash(root) != report['resultHash']:
        raise ValueError('原始模型结果已变化，不能用不同数据重画引用图')
    # Code can be patched; model inputs must still be the ones the report used.
    signatures = report.get('inputSignatures', {})
    for filename, digest in signatures.items():
        path = Path(filename)
        if path.suffix in {'.json', '.npy', '.npz', '.csv', '.tsv', '.xlsx', '.parquet'}:
            if not path.is_file() or file_hash(path) != digest:
                raise ValueError('原报告输入已变化，不能保证复现选中图')
    model = report['modelId']
    params = report.get('trainingPlan', {}).get('params', {})
    context = report.get('renderContext')
    if context:
        options = context['loaderOptions']
    elif model == 'theta':
        options = dict(result_dir=str(root.parents[3]), dataset=root.parents[2].name,
                       model_size=root.parents[1].name, model_exp=root.name,
                       mode=params.get('mode', 'zero_shot'), model_type='theta')
    else:
        vocab_dirs = {str(Path(p).parent) for p in signatures if Path(p).name == 'vocab.json'}
        if len(vocab_dirs) > 1:
            raise ValueError('原报告词表来源不唯一，无法安全复现')
        options = dict(result_dir=str(root), dataset=root.parent.parent.name, model=model,
                       num_topics=int(params.get('num_topics', 3)), workspace_dir=next(iter(vocab_dirs), None))
    # Older reports do not persist the aligned metadata used by time/group charts.
    if not context and (chart in {'doc_volume', 'representative_topic_evolution', 'kl_divergence',
            'vocab_evolution', 'topic_similarity_evolution', 'all_topics_strength_table',
            'dimension_heatmap', 'domain_topic_distribution'} or ':T' in chart):
        raise ValueError('旧报告缺少该图的对齐时间/分组参数，需先重新整理原生报告；未生成替代图')
    output_dir = report_dir / 'adjustments' / uuid4().hex[:12]
    output_dir.mkdir(parents=True)
    with (output_dir / 'render.log').open('w') as log, redirect_stdout(log), redirect_stderr(log):
        loader = native.load_visualization_data if model == 'theta' else native.load_baseline_data
        data = loader(**options)
        if context:
            import pandas as pd
            data['timestamps'] = pd.to_datetime(context['timestamps']).tolist() if context.get('timestamps') is not None else None
            data['dimension_values'] = context.get('dimensionValues')
        runner = native.run_all_visualizations if model == 'theta' else native.run_baseline_visualization
        with figure_edit(chart, relative, output_dir, spec) as edit:
            runner(**options, data=data, output_dir=output_dir, language='zh', dpi=300)
    files = list(dict.fromkeys(edit['files']))
    if not files:
        raise ValueError('选中图的原生函数没有成功导出，未生成替代图；请检查绘图日志')
    for filename in files:
        Path(filename).with_suffix('.json').write_text(json.dumps({
            'originalFigure': str(original.relative_to(report_dir)), 'chart': chart,
            'spec': spec, 'resultHash': report['resultHash'],
            'nativeRunner': report['nativeRunner'],
        }, ensure_ascii=False, indent=2))
    preview = next((filename for filename in files if filename.endswith('.png')), files[0])
    return {'path': preview, 'paths': files, 'name': Path(preview).name, 'kind': 'native',
            'originalFigure': str(original.relative_to(report_dir)), 'renderer': chart,
            'note': '通过选中图的原生绘图函数修改标题并另存；原始模型数据、图类型、页码和绘图参数未改动。'}
