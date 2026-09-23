"""Deliver native THETA visualizations and evidence; no plotting or modeling implementation."""
from contextlib import redirect_stdout, redirect_stderr
import html
import json
from pathlib import Path
import sys
from uuid import uuid4

from .results_reader import file_hash, tree_hash, read_result_evidence
from .capabilities import engine_root, verify_dataset
from .dataset.readers import load_dataset
from .dataset.business import excerpt
from .runtime_environments import identity


def write_delivery_tables(target: Path, data: dict) -> None:
    """Export the two tables the manual workbench always offers for download.

    topic_keywords.csv: topic x keyword weights taken from the model's own beta.
    document_topic.csv: document x topic weights taken from the model's own theta.
    No value is recomputed, rescaled or imputed.
    """
    import numpy as np
    tables = target / 'tables'
    tables.mkdir(parents=True, exist_ok=True)
    theta = data.get('theta')
    if theta is not None and np.asarray(theta).ndim == 2:
        matrix = np.asarray(theta, dtype=float)
        labels = ['T%d' % (index + 1) for index in range(matrix.shape[1])]
        dominant = matrix.argmax(axis=1) + 1
        lines = ['matrix_row,' + ','.join(labels) + ',dominant_topic']
        lines.extend(str(row) + ',' + ','.join('%.6f' % value for value in matrix[row]) + ',' + str(int(dominant[row]))
                     for row in range(matrix.shape[0]))
        (tables / 'document_topic.csv').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    beta = data.get('beta')
    vocab = data.get('vocab')
    if beta is not None and vocab is not None and len(vocab):
        weights = np.asarray(beta, dtype=float)
        words = list(vocab)
        if weights.ndim == 2 and weights.shape[1] == len(words):
            lines = ['topic_id,rank,keyword,weight']
            for topic in range(weights.shape[0]):
                order = np.argsort(weights[topic])[::-1][:50]
                for rank, index in enumerate(order, start=1):
                    keyword = str(words[int(index)]).replace(',', ' ').replace('\n', ' ').strip()
                    lines.append('%d,%d,%s,%.6f' % (topic + 1, rank, keyword, float(weights[topic][int(index)])))
            (tables / 'topic_keywords.csv').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def generate_report(root, job, destination, *, workspace=None, dataset=None, prepared=None, worker_log=None):
    runtime = identity('reports')
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    sys.path.insert(0, str(engine_root() / 'src/models'))
    from visualization import run_visualization as native
    from row_provenance import verify_lineage

    root = Path(root).resolve()
    if tree_hash(root) != job['resultHash']:
        raise ValueError('Result artifact changed before report generation')
    model = job.get('plan', {}).get('modelId', 'unknown')
    plan = job.get('plan', {})
    params = plan.get('params', {})
    source = verify_dataset(dataset) if dataset else None
    metadata_dir = native.resolve_theta_data_dir(root, params.get('mode', 'zero_shot')) if model == 'theta' else Path(workspace) if workspace else None
    # The worker entry configures the visualization runtime (writable numba cache for
    # UMAP's cached kernels); a changed entry must invalidate cached reports.
    inputs = [Path(__file__), Path(__file__).with_name('__main__.py'), Path(native.__file__), Path(native.__file__).with_name('visualization_generator.py'),
              Path(native.__file__).with_name('topic_visualizer.py'),
              Path(native.__file__).with_name('publication.py'),
              Path(native.__file__).parents[1] / 'artifact_utils.py', Path(__file__).with_name('result_analysis.py'),
              Path(__file__).with_name('results_reader.py'), Path(__file__).with_name('diagnostics.py'),
              engine_root() / 'src/models/row_provenance.py']
    if source:
        inputs.append(source)
    if prepared: inputs.append(Path(prepared))
    if worker_log and Path(worker_log).is_file(): inputs.append(Path(worker_log))
    if workspace:
        inputs.extend(f for f in Path(workspace).iterdir() if f.name in
                      {'vocab.json', 'bow_matrix.npy', 'bow_matrix.npz', 'covariates.npy', 'covariate_names.json',
                       'covariate_encoding.json', 'texts.json', 'source_rows.npy', 'row_provenance.json', 'time_indices.npy', 'time_slices.json'})
    if model == 'theta' and metadata_dir:
        inputs.extend(f for f in metadata_dir.rglob('*') if f.is_file() and f.name in {'texts.json', 'source_rows.npy', 'row_provenance.json', 'vocab.json', 'vocab.txt', 'bow_matrix.npy', 'bow_matrix.npz'})
    signatures = {str(f.resolve()): file_hash(f) for f in inputs}
    parent = Path(destination) / job['id']
    for manifest in sorted(parent.glob('report-*/manifest.json'), reverse=True):
        cached = json.loads(manifest.read_text())
        if (cached.get('schemaVersion') == 'theta.result-report.v2' and cached.get('resultHash') == job['resultHash']
                and cached.get('inputSignatures') == signatures and cached.get('runtime') == runtime
                and all(Path(f['path']).is_file() and file_hash(Path(f['path'])) == f['sha256'] for f in cached['files'])):
            return cached
    target = parent / ('report-' + uuid4().hex)
    target.mkdir(parents=True, exist_ok=False)
    log = target / 'visualization.log'
    source_evidence = None
    missing = [job['resultWarning']] if job.get('resultWarning') else []
    data = None
    render_context = None
    try:
        with log.open('w', encoding='utf-8') as output, redirect_stdout(output), redirect_stderr(output):
            if model == 'theta':
                options = dict(result_dir=str(root.parents[3]), dataset=root.parents[2].name,
                               model_size=root.parents[1].name, model_exp=root.name,
                               mode=params.get('mode', 'zero_shot'), model_type='theta')
                data = native.load_visualization_data(**options)
            else:
                options = dict(result_dir=str(root), dataset=root.parent.parent.name, model=model,
                               num_topics=int(params.get('num_topics', 3)), workspace_dir=workspace)
                data = native.load_baseline_data(**options)
            # Fail rather than silently shipping word_0 placeholders or a mismatched vocabulary.
            if not data.get('vocab') or len(data['vocab']) != data['beta'].shape[1]:
                raise ValueError('Native visualization vocabulary does not match beta')
            if workspace and not (Path(workspace) / 'vocab.json').is_file() and model not in {'theta', 'bertopic'}:
                raise ValueError('Training vocabulary is unavailable; cannot generate truthful visualizations')
            for name in ['theta', 'beta']:
                value = data[name]
                if value.ndim != 2 or not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all() or ((value < 0).any() and not (model == 'nvdm' and name == 'theta')):
                    raise ValueError(f'Invalid {name} matrix')
            if source:
                table = load_dataset(source, profile_limit=1_000_000)
                if table.rows_truncated:
                    raise ValueError('Source metadata exceeds the existing full-data reader limit')
                text_column = plan['textColumn']
                columns = list(dict.fromkeys([text_column, *plan.get('covariates', []),
                    *[c for c in ['timestamp', 'year', 'date', '日期', 'channel'] if c in table.columns]]))
                rows = table.rows
                aligned = False
                indices = np.arange(len(rows))
                # A job hash binds the normalized input; exported row indices/texts prove any filtering.
                if prepared:
                    prepared = Path(prepared)
                    normalized = load_dataset(prepared, profile_limit=1_000_000)
                    original_matches = (file_hash(prepared) == job.get('preparedHash') and not normalized.rows_truncated
                        and len(rows) == len(normalized.rows)
                        and all(str(a.get(text_column) or '') == str(b.get('text') or '') for a, b in zip(rows, normalized.rows)))
                    if metadata_dir and (metadata_dir / 'source_rows.npy').is_file() and (metadata_dir / 'texts.json').is_file():
                        indices = np.load(metadata_dir / 'source_rows.npy', allow_pickle=False)
                        texts = json.loads((metadata_dir / 'texts.json').read_text())
                        indices_valid = (indices.ndim == 1 and np.issubdtype(indices.dtype, np.integer)
                            and len(indices) == len(data['theta']) == len(texts) and len(set(indices.tolist())) == len(indices)
                            and (indices >= 0).all() and (indices < len(normalized.rows)).all())
                        exact_texts = indices_valid and all(str(normalized.rows[int(i)].get('text') or '') == text
                            for i, text in zip(indices, texts))
                        lineage_file = metadata_dir / 'row_provenance.json'
                        lineage_matches = False
                        if lineage_file.is_file() and indices_valid:
                            lineage_matches = verify_lineage(json.loads(lineage_file.read_text()), file_hash(prepared),
                                [str(row.get('text') or '') for row in normalized.rows], texts, indices)
                        aligned = (original_matches and indices_valid
                            and (lineage_matches if lineage_file.is_file() else exact_texts))
                    elif model in {'lda', 'stm', 'btm', 'hdp', 'etm', 'ctm', 'nvdm', 'gsm', 'prodlda'}:
                        aligned = original_matches and len(rows) == len(data['theta'])
                display_rows = [rows[int(i)] for i in indices] if aligned else rows
                source_evidence = {'datasetRef': dataset['datasetRef'], 'datasetHash': dataset['sha256'],
                    'textColumn': text_column, 'rowCount': table.row_count, 'matrixRowsAligned': bool(aligned),
                    'excludedSourceRows': len(rows) - len(indices) if aligned else None,
                    'coverage': 'all model rows' if aligned and len(display_rows) <= 40 else 'first 40 rows, not representative',
                    'rows': [{'sourceRow': int(indices[i]) + 1 if aligned else i + 1, **{c: excerpt(row.get(c), 500) for c in columns},
                             **({('latentCoordinates' if model == 'nvdm' else 'membershipWeights' if model == 'bertopic' else 'topicWeights'): data['theta'][i].tolist()} if aligned else {}),
                             **({'assignedTopic': int(data['document_topics'][i])} if aligned and model == 'bertopic' else {})}
                             for i, row in enumerate(display_rows[:40])]}
                if aligned:
                    frame = pd.DataFrame(display_rows)
                    data['source_frame'] = frame
                    if plan.get('covariates'):
                        column = plan['covariates'][0]
                        data['source_covariate_column'] = column
                        data['dimension_values'] = frame[column].fillna('unknown').astype(str).to_numpy()
                    column = plan.get('timeColumn') or next((c for c in ['timestamp', 'date', '日期', 'year'] if c in frame), None)
                    if column:
                        dates = pd.to_datetime(frame[column].astype(str), errors='coerce')
                        if dates.notna().all():
                            data['timestamps'] = dates.tolist()
                else:
                    missing.append('Original rows are not proven aligned to theta; no row/topic or temporal association inferred.')
            render_context = {'loaderOptions': {key: str(value) if isinstance(value, Path) else value for key, value in options.items()},
                'timestamps': [str(value) for value in data['timestamps']] if data.get('timestamps') is not None else None,
                'dimensionValues': np.asarray(data['dimension_values']).tolist() if data.get('dimension_values') is not None else None}
            if model == 'theta':
                native.run_all_visualizations(**options, data=data, output_dir=target / 'native', language='zh', dpi=300)
            else:
                native.run_baseline_visualization(**options, data=data, output_dir=target / 'native', language='zh', dpi=300)
            try:
                write_delivery_tables(target, data)
            except Exception as error:  # a missing export must not fail the whole report
                missing.append(f'可下载表格导出失败：{error}')
            # The manual workbench shows the metric table itself; the chart is not produced.
            for chart in sorted((target / 'native').rglob('7 Core Metrics.*')):
                try:
                    chart.unlink()
                except OSError:
                    pass
    except Exception as error:
        import traceback
        with log.open('a', encoding='utf-8') as output:
            traceback.print_exc(file=output)
        missing.append(f'原生结果整理失败：{error}。仅交付已有文件与诊断，不代表完整分析结果；未重训或推断缺失标签。')
        data = None
        source_evidence = None
    # Native runner attempts its complete chart set; report its skips/errors honestly.
    missing.extend(line.strip()[:400] for line in log.read_text().splitlines()
                   if any(marker in line.lower() for marker in ['[skip]', '[error]', 'skipped', 'placeholder vocab', 'visualizations error', '⚠']))
    if tree_hash(root) != job['resultHash'] or any(file_hash(Path(f)) != digest for f, digest in signatures.items()):
        raise ValueError('Result or source data changed during report generation')
    files = []
    for base, prefix in [(root, 'training'), *([(target / 'native', 'native')] if (target / 'native').is_dir() else []),
                          *([(target / 'tables', 'tables')] if (target / 'tables').is_dir() else [])]:
        for file in sorted(base.rglob('*')):
            if not file.is_file():
                continue
            relative = file.relative_to(base).as_posix()
            # Prior training plots remain in the original tree; deliver the corrected native
            # regeneration and original numerical/model artifacts without mixing obsolete charts.
            if prefix == 'training' and any(part in {'en', 'zh', 'visualization'} for part in file.relative_to(base).parts):
                continue
            kind = 'figure' if file.suffix.lower() in {'.png', '.svg', '.jpg', '.pdf'} else 'table' if file.suffix in {'.csv', '.tsv'} else 'matrix' if file.suffix == '.npy' else 'artifact'
            files.append({'name': f'{prefix}/{relative}', 'path': str(file.resolve()), 'kind': kind,
                          'sha256': file_hash(file), 'sizeBytes': file.stat().st_size})
    evidence = {'trainingRunId': job['id'], 'evidence': [], 'tables': [], 'figures': [], 'matrices': [], 'limitations': missing}
    if data is not None:
        evidence = read_result_evidence({'trainingRunId': job['id'], 'modelId': model, 'artifacts': [
            {'kind': 'results', 'path': str(root), 'sha256': job['resultHash'], 'artifactId': job['id']}]})
        # The old training presentation may contain placeholder labels. Only regenerated native
        # tables and figures belong to the interpretation evidence for this report.
        from .result_analysis import presentation_evidence
        presentation = presentation_evidence(target / 'native', file_hash, evidence['evidence'], model)
        evidence.update(tables=presentation['tables'], figures=presentation['figures'], analysisSkipped=presentation['analysisSkipped'])
        if model == 'stm' and workspace and (Path(workspace) / 'covariate_encoding.json').is_file():
            encoding = json.loads((Path(workspace) / 'covariate_encoding.json').read_text())
            evidence['covariateEncoding'] = {
                'columns': {name: {'sourceColumn': plan.get('covariates', [])[index] if index < len(plan.get('covariates', [])) else None,
                                   'values': values} for index, (name, values) in enumerate(encoding.items())},
                'method': 'sklearn LabelEncoder; sorted categories encoded as integers. More than two nominal categories impose an ordinal score, not separate indicator effects.'}
        if data.get('vocab'):
            evidence['vocabulary'] = data['vocab'][:100]
            evidence['vocabularyOmitted'] = max(0, len(data['vocab']) - 100)
    complete = data is not None
    data = data or {}
    training_log = Path(worker_log).resolve() if worker_log and Path(worker_log).is_file() else None
    body = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>THETA 原生结果报告</title>',
        '<style>body{font:16px/1.65 system-ui;max-width:1100px;margin:40px auto;padding:0 24px}img{max-width:100%}li{margin:8px 0}</style>',
        f'<h1>{html.escape(model.upper())} · 原生可视化与训练产物</h1>',
        '<p>复用 src/models/visualization；未重训、未改写 theta/beta。已有文件见下方。</p>',
        '<h2>结果整理未完成：以下仅为原始产物与诊断，不能视作完整图表报告或研究结论。</h2>' if not complete else '',
        '<p>原始结果目录：' + html.escape(str(root)) + '</p>',
        f'<p><a href="{training_log.as_uri()}">原始训练日志</a></p>' if training_log else '',
        '<p>图表统计范围：' + html.escape(data.get('plot_scope', '全部模型行；原文关联以行序校验为准' if complete else '不可用：缺少通过校验的原生结果')) + '</p>',
        '<h2>未生成项与运行记录</h2><ul>' + ''.join(f'<li>{html.escape(x)}</li>' for x in missing) + '</ul>',
        f'<a href="{log.as_uri()}">完整原生可视化日志</a><h2>已有图表与产物</h2>']
    for item in files:
        uri = Path(item['path']).as_uri()
        body.append(f'<p><a href="{html.escape(uri, quote=True)}">{html.escape(item["name"])}</a></p>')
        if item['kind'] == 'figure' and Path(item['path']).suffix.lower() in {'.png', '.jpg', '.svg'}:
            body.append(f'<img loading="lazy" src="{html.escape(uri, quote=True)}" alt="{html.escape(item["name"], quote=True)}">')
    page = target / 'index.html'; page.write_text('\n'.join(body) + '</html>', encoding='utf-8')
    if training_log:
        files.append({'name': 'worker.log', 'path': str(training_log), 'kind': 'artifact', 'sha256': file_hash(training_log), 'sizeBytes': training_log.stat().st_size})
    files.append({'name': 'visualization.log', 'path': str(log.resolve()), 'kind': 'artifact', 'sha256': file_hash(log), 'sizeBytes': log.stat().st_size})
    files.append({'name': 'index.html', 'path': str(page.resolve()), 'kind': 'report', 'sha256': file_hash(page), 'sizeBytes': page.stat().st_size})
    manifest = target / 'manifest.json'
    report = {'schemaVersion': 'theta.result-report.v2', 'jobId': job['id'], 'modelId': model,
        'resultHash': job['resultHash'], 'inputSignatures': signatures, 'files': files, 'runtime': runtime,
        'trainingPlan': plan, 'renderContext': render_context,
        'reportStatus': 'complete' if complete else 'incomplete', 'resultDir': str(root), 'logPath': str(log.resolve()),
        'trainingLogPath': str(training_log) if training_log else None, 'diagnostics': job.get('diagnostics'),
        'evidence': evidence, 'sourceData': source_evidence, 'missingEvidence': missing,
        'nativeRunner': 'src/models/visualization/run_visualization.py',
        'reportPath': str(page.resolve()), 'manifestPath': str(manifest.resolve()),
        'quality': {'convergence': 'not_assessed', 'thetaSemantics': 'latent_coordinates_not_probabilities' if model == 'nvdm' else 'membership_mass_excluding_outliers' if model == 'bertopic' else 'topic_distribution',
                    'betaSemantics': 'normalized_selected_ctfidf_weights' if model == 'bertopic' else 'last_time_slice_topic_word_distribution' if model == 'dtm' else 'topic_word_distribution',
                    'plotScope': data.get('plot_scope', 'all model rows' if complete else 'unavailable'),
                    'metricDefinitions': {
                        'source': 'src/models/evaluation/topic_metrics.py',
                        'C_V': 'Repository approximation: mean of prefix-averaged pairwise document-level NPMI. It does not implement the standard sliding-window/indirect cosine C_V procedure; negative values are possible. Label as C_V approximation, not standard C_V.',
                        'NPMI': 'Pairwise document-level word co-occurrence, not sentence/sliding-window co-occurrence. Report the corpus and selected top-word scope.',
                        'Significance': 'Legacy field: standard deviation of topic mean weights; Significance_per_topic stores mean weights. Neither is an inferential significance test.',
                        'Exclusivity': 'For each topic: reciprocal of the mean number of top-word lists containing each selected top word; 1 means no overlap among those lists, not zero probability outside the topic. It is not caused by small vocabulary alone.',
                        'PPL': 'Unified evaluator computes exp(-sum(BOW * log(clip(theta @ beta, 1e-12, 1))) / sum(BOW)) on the supplied matrix. This is not held-out validation and need not equal a model library perplexity/variational bound.',
                        'comparison': 'No universal quality thresholds. BERTopic c-TF-IDF and DTM last-slice beta do not define an equivalent generative likelihood to static LDA; do not rank models by this common reconstruction score.'}},
        'summary': {'topicCount': int(data['theta'].shape[1]), 'documentCount': int(data['theta'].shape[0])} if complete else {'status': 'incomplete', 'reason': missing}}
    manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return report
