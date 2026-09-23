"""Interpretation evidence derived from verified results; no fitting or new plots."""
import csv
import re
from .dataset.business import excerpt


def matrix_summary(file, model_id=None):
    """Read bounded original values; all statistical/visual calculations belong to src/models."""
    import numpy as np
    value = np.load(file, allow_pickle=False, mmap_mode='r')
    if value.ndim != 2 or not np.issubdtype(value.dtype, np.number) or np.iscomplexobj(value) or value.size > 4_000_000 or 0 in value.shape:
        raise ValueError('矩阵类型、维度或大小不适合有界解读')
    latent = model_id == 'nvdm' and file.name.startswith('theta')
    if not np.isfinite(value).all() or not latent and ((value < 0).any() or not value.any()):
        raise ValueError('分布矩阵含无效值')
    return {'kind': 'latent_document_representation' if latent else 'topic_distribution' if file.name.startswith('theta') else 'topic_word_distribution',
            'thetaSemantics': 'latent_coordinates_not_probabilities' if latent else 'membership_mass_excluding_outliers' if model_id == 'bertopic' else 'topic_distribution',
            'betaSemantics': 'normalized_selected_ctfidf_weights' if model_id == 'bertopic' else 'last_time_slice_topic_word_distribution' if model_id == 'dtm' else 'topic_word_distribution',
            'shape': list(value.shape), 'values': value[:40, :50].tolist(),
            'omittedRows': max(0, value.shape[0] - 40), 'omittedColumns': max(0, value.shape[1] - 50),
            'method': 'Original matrix values, without normalization or new statistics. theta=document/topic, beta=topic/vocabulary. Indices are zero-based.',
            'limitation': '完整矩阵通过原始NPY文件交付；统计与可视化使用原引擎产物。数值摘录不是图像像素。'}


def presentation_evidence(root, hash_file, json_evidence, model_id=None):
    matrices, tables, figures, skipped = [], [], [], []
    for file in sorted(root.rglob('*')):
        if not file.is_file():
            continue
        name = file.relative_to(root).as_posix()
        if file.suffix.lower() == '.npy' and re.fullmatch(r'(?:theta|beta)(?:_[\w.-]+)?\.npy', file.name):
            if file.stat().st_size > 32 * 1024 * 1024 or len(matrices) >= 6:
                skipped.append(name)
                continue
            try:
                matrices.append({'relativePath': name, 'sha256': hash_file(file), **matrix_summary(file, model_id)})
            except (ValueError, OSError, ImportError) as error:
                skipped.append({'relativePath': name, 'reason': excerpt(error, 200)})
        elif file.suffix.lower() in {'.csv', '.tsv'}:
            if file.stat().st_size > 1024 * 1024 or len(tables) >= 32:
                skipped.append(name)
                continue
            try:
                with file.open(encoding='utf-8-sig', newline='') as handle:
                    reader = csv.DictReader(handle, delimiter='\t' if file.suffix.lower() == '.tsv' else ',')
                    columns = (reader.fieldnames or [])[:12]
                    rows = []
                    count = 0
                    for row in reader:
                        count += 1
                        if len(rows) < 20:
                            rows.append({excerpt(key, 80): excerpt(row.get(key), 200) for key in columns})
                tables.append({'relativePath': name, 'sha256': hash_file(file), 'columns': [excerpt(key, 80) for key in columns],
                               'rows': rows, 'rowCount': count, 'omittedRows': max(0, count - 20),
                               'omittedColumns': max(0, len(reader.fieldnames or []) - 12)})
            except (UnicodeError, csv.Error, OSError) as error:
                skipped.append({'relativePath': name, 'reason': excerpt(error, 200)})
        elif file.suffix.lower() in {'.png', '.jpg', '.jpeg', '.svg', '.pdf', '.html'}:
            if len(figures) >= 240:
                skipped.append(name)
                continue
            figures.append({'relativePath': name, 'sha256': hash_file(file), 'format': file.suffix[1:],
                            'sizeBytes': file.stat().st_size})
    sources = [(entry['relativePath'], entry['kind']) for entry in [*json_evidence, *matrices]]
    for figure in figures:
        name = figure['relativePath'].lower()
        if any(word in name for word in ('similarity', '相似')):
            kind, meaning = 'topic_similarity', '主题词相似度：哪些主题可能重叠'
        elif any(word in name for word in ('umap', 'distance', '分布', '距离')):
            kind, meaning = 'topic_distribution', '主题覆盖与文档分布；现有摘要不提供降维图坐标'
        elif any(word in name for word in ('word', '词')):
            kind, meaning = 'topics', '主题关键词与候选业务命名'
        elif any(word in name for word in ('metric', '指标')):
            kind, meaning = 'metrics', '模型质量与可解释性诊断'
        else:
            kind, meaning = 'unknown', '需结合相应结果数据核实内容'
        # Only associate an unambiguous source. Filenames do not prove image contents.
        candidates = [path for path, category in sources if category == kind or kind in {'topic_similarity', 'topics'} and category == 'topic_word_distribution']
        figure.update({'suggestedReading': meaning, 'basis': 'underlying_results_not_image_pixels',
                       'sourceCandidates': candidates, 'dataSource': candidates[0] if len(candidates) == 1 else None,
                       'limitation': '按文件名推测图表类型；仅解释已核实的数据，不声称看见图片中的颜色、位置或趋势。'})
    return {'matrices': matrices, 'tables': tables, 'figures': figures, 'analysisSkipped': skipped}
