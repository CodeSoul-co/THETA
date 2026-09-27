"""Render held-out views from verified saved rows, without refitting the model."""
import csv
import json
from pathlib import Path
import shutil
import numpy as np


def attach_split_report(data):
    matrix_dir = data.get('matrix_dir')
    if matrix_dir is None:
        return
    directory = Path(matrix_dir)
    for parent in [directory, *list(directory.parents)[:3]]:
        file = parent / 'split_results.json'
        if file.is_file():
            report = json.loads(file.read_text(encoding='utf-8'))
            if report.get('groups', {}).get('all', {}).get('count') != len(data['theta']):
                raise ValueError('划分结果与模型矩阵行数不一致')
            data['split_report_path'] = file
            data['metrics'] = report['groups']['all']['metrics']
            return


def split_view(data, report_path, role):
    report_path = Path(report_path)
    report = json.loads(report_path.read_text(encoding='utf-8'))
    folder = report_path.parent / 'splits' / role
    with (folder / 'documents.csv').open(encoding='utf-8-sig', newline='') as stream:
        source_rows = [int(row['source_row']) - 1 for row in csv.DictReader(stream)]
    original = np.asarray(data.get('source_rows', np.arange(len(data['theta']))), dtype=int)
    lookup = {int(row): i for i, row in enumerate(original)}
    if len(lookup) != len(original) or len(set(source_rows)) != len(source_rows):
        raise ValueError('划分结果的原始行映射重复')
    indices = [lookup[row] for row in source_rows]
    saved = np.load(folder / 'theta.npy', allow_pickle=False)
    values = data['theta'][indices]
    if len(indices) != report['groups'][role]['count'] or not np.array_equal(saved, values):
        raise ValueError('划分结果与保存的主题矩阵不一致，不能绘图')
    view = {**data, 'theta': values, 'metrics': report['groups'][role]['metrics']}
    view.pop('split_report_path', None)
    # Row-dependent inputs must all use exactly the same selection.
    for key in ['bow_matrix', 'timestamps', 'dimension_values', 'document_topics', 'covariates', 'source_rows', 'source_frame']:
        value = data.get(key)
        if value is None:
            continue
        count = value.shape[0] if hasattr(value, 'shape') else len(value)
        if count != len(original):
            raise ValueError(f'{key} 与模型行数不一致，不能绘制划分结果')
        view[key] = value.iloc[indices].copy() if key == 'source_frame' else value[indices] if hasattr(value, 'shape') else [value[i] for i in indices]
    if role != 'train':
        view['training_history'] = None
    view['plot_scope'] = f'{role}: {len(indices)} model rows; shared fitted beta, no retraining.'
    return view


def render_split_views(data, output_dir, render):
    report_path = data.get('split_report_path')
    if report_path is None:
        return
    output_dir = Path(output_dir)
    report = json.loads(Path(report_path).read_text(encoding='utf-8'))
    # Snapshot the full-data files before rendering any subset to avoid recursive copies.
    full_files = [file for file in output_dir.rglob('*') if file.is_file() and 'splits' not in file.relative_to(output_dir).parts]
    statuses = {}
    for role in ('train', 'validation', 'test'):
        target = output_dir / 'splits' / role
        try:
            view = split_view(data, report_path, role)
            if role == 'test' and report.get('testUsesAllData'):
                for source in full_files:
                    destination = target / source.relative_to(output_dir)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, destination)
            else:
                render(view, target)
            statuses[role] = {'status': 'complete', 'count': len(view['theta'])}
            print(f'[数据集可视化] {role}: {len(view["theta"])} 条正文', flush=True)
        except Exception as error:
            statuses[role] = {'status': 'error', 'reason': str(error)}
            print(f'[Error] 数据集可视化 {role}: {error}', flush=True)
    (output_dir / 'split-visualizations.json').write_text(json.dumps(statuses, ensure_ascii=False, indent=2), encoding='utf-8')
