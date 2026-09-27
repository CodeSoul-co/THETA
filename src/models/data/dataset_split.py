"""One row assignment shared by preparation, fitting and result reporting."""
import json
import math
import os
import random
from pathlib import Path

ROLES = ('train', 'validation', 'test')


def assignments(count, options=None, roles=None):
    options = options or {}
    enabled = options.get('enabled', False)
    if not isinstance(enabled, bool):
        raise ValueError('数据划分开关必须是布尔值')
    mode = options.get('mode', 'ratio') if enabled else 'ratio'
    method = options.get('method', 'random') if enabled else 'random'
    seed = options.get('seed', 42) if enabled else 42
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ValueError('随机种子必须为 0–4294967295 的整数')
    if mode not in ('ratio', 'upload') or method not in ('random', 'sequential'):
        raise ValueError('请选择按比例/独立上传，以及随机/顺序划分')
    if mode == 'upload':
        if roles is None or len(roles) != count or any(role not in ROLES for role in roles):
            raise ValueError('独立上传需要完整的训练、验证和测试集')
        groups = {role: [i for i, value in enumerate(roles) if value == role] for role in ROLES}
    else:
        ratios = options.get('ratios', [0.7, 0.2, 0.1]) if enabled else [0.7, 0.3, 0]
        if not isinstance(ratios, list) or len(ratios) != 3 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in ratios) or abs(sum(ratios)-1) > 1e-8:
            raise ValueError('训练、验证、测试比例必须合计为 100%')
        if enabled and any(v <= 0 for v in ratios):
            raise ValueError('开启独立划分时，三份数据比例都必须大于 0')
        indices = list(range(count))
        if method == 'random': random.Random(seed).shuffle(indices)
        n_train, n_val = int(count*ratios[0]), int(count*ratios[1])
        if not enabled: n_val = count-n_train
        groups = dict(zip(ROLES, [indices[:n_train], indices[n_train:n_train+n_val], indices[n_train+n_val:]]))
        if not enabled: groups['test'] = list(range(count))
    if len(groups['train']) < 2 or not groups['validation'] or not groups['test']:
        raise ValueError('划分后训练集至少需要 2 条有效正文，验证集和测试集各至少 1 条；请增加数据或调整比例')
    return {'schemaVersion': 1, 'enabled': enabled, 'mode': mode, 'method': method, 'seed': seed, 'total': count, 'groups': groups, 'testUsesAllData': not enabled}


def plan_options(plan):
    if plan.get('dataSplit') is not None: return plan['dataSplit']
    params = plan.get('params', {})
    if any('config.' + role + '_ratio' in params for role in ('train', 'val', 'test')):
        train = params.get('config.train_ratio', .8)
        val = params.get('config.val_ratio', .1)
        test = params.get('config.test_ratio', 1-train-val)
        return {'enabled': True, 'ratios': [train, 1-train-test, test], 'seed': params.get('pipeline.seed', 42)}
    return {}


def current(count, source_rows=None):
    location = os.environ.get('THETA_DATA_SPLIT_FILE')
    if not location: return None
    manifest = json.loads(Path(location).read_text(encoding='utf-8'))
    if source_rows is None:
        if manifest['total'] != count: raise ValueError('数据行数与划分记录不一致，不能继续训练')
        return manifest
    rows = [int(i) for i in source_rows]
    if len(rows) != count or len(set(rows)) != count or any(i < 0 or i >= manifest['total'] for i in rows):
        raise ValueError('预处理后的数据行映射无效')
    inverse = {row: i for i, row in enumerate(rows)}
    manifest = {**manifest, 'total': count, 'groups': {role: [inverse[i] for i in group if i in inverse] for role, group in manifest['groups'].items()}}
    if len(manifest['groups']['train']) < 2 or any(not manifest['groups'][r] for r in ROLES):
        raise ValueError('清理无效正文或时间后，某份数据集为空或训练集不足 2 条，请调整数据与划分')
    return manifest


def subset(dataset, role, source_rows=None):
    from torch.utils.data import Subset
    spec = current(len(dataset), source_rows)
    return Subset(dataset, spec['groups'][role]) if spec else dataset


def export_results(directory, theta, beta, bow, vocab, model_name, source_rows=None):
    """Evaluate the same fitted model on each declared set; never refit on test rows."""
    import csv
    import numpy as np
    from evaluation.unified_evaluator import UnifiedEvaluator
    spec = current(len(theta), source_rows)
    if not spec: return
    directory = Path(directory)
    report = {**spec, 'groups': {}, 'model': model_name}
    for role, indices in {'all': list(range(len(theta))), **spec['groups']}.items():
        target = directory / 'splits' / role
        target.mkdir(parents=True, exist_ok=True)
        values = np.asarray(theta)[indices]
        if role == 'test' and spec['testUsesAllData']:
            import shutil
            metrics = report['groups']['all']['metrics']
            for file in (directory / 'splits/all').glob('metrics*.json'):
                shutil.copyfile(file, target / file.name)
        else:
            evaluator = UnifiedEvaluator(beta=beta, theta=values, bow_matrix=bow[indices], vocab=vocab,
                                         model_name=model_name, output_dir=str(target), num_topics=values.shape[1])
            metrics = evaluator.compute_all_metrics()
            evaluator.save_metrics()
        weights = values.mean(axis=0)
        report['groups'][role] = {'count': len(indices), 'metrics': metrics, 'topicProportions': weights.tolist(), 'maxTopicStd': float(values.std(axis=0).max())}
        with (target / 'documents.csv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['source_row', 'topic', *[f'topic_{i}' for i in range(values.shape[1])]])
            for original, row in zip(indices, values): writer.writerow([int(source_rows[original]) + 1 if source_rows is not None else original + 1, int(row.argmax()), *row.tolist()])
        np.save(target / 'theta.npy', values)
        print(f"[数据集评估] {role}: {len(indices)} 条正文", flush=True)
    def clean(value):
        if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)): return [clean(v) for v in value]
        if hasattr(value, 'item'): return clean(value.item())
        if isinstance(value, float) and not math.isfinite(value): return None
        return value
    (directory / 'split_results.json').write_text(json.dumps(clean(report), ensure_ascii=False, indent=2), encoding='utf-8')
