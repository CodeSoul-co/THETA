"""One fixed, versioned loading entry point for all trained THETA models.

Only load bundles produced by a trusted training run: joblib/pickle can execute code.
No model is fitted, downloaded or sent to an external service by this module.
"""
from pathlib import Path
import hashlib
import inspect
import json
import platform
from importlib.metadata import version, PackageNotFoundError


def save_trained_model(model, directory, model_id, vocab):
    import joblib
    import torch

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / 'trained_model.joblib'
    # Portable CPU tensor storage without changing the live model's device.
    if isinstance(model, torch.nn.Module):
        import copy
        serialized = copy.deepcopy(model).cpu().eval()
    else:
        serialized = model
    joblib.dump(serialized, target, compress=3)
    (directory / 'vocab.json').write_text(json.dumps(list(vocab), ensure_ascii=False), encoding='utf-8')
    packages = {}
    for package in ['numpy', 'scipy', 'torch', 'scikit-learn', 'joblib', 'gensim', 'bertopic', 'umap-learn', 'hdbscan', 'sentence-transformers', 'jieba', 'langdetect', 'pandas', 'matplotlib', 'seaborn', 'plotly', 'wordcloud']:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            pass
    methods = {}
    for name in ['get_beta', 'get_theta', 'transform', 'forward', 'get_topic_words']:
        method = getattr(model, name, None)
        if callable(method):
            methods[name] = {'signature': str(inspect.signature(method)), 'description': inspect.getdoc(method) or ''}
    manifest = {'schemaVersion': 1, 'modelId': model_id, 'file': target.name,
                'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                'class': f'{type(model).__module__}.{type(model).__name__}',
                'python': platform.python_version(), 'packages': packages, 'methods': methods,
                'warning': '仅加载可信来源的模型；模型包不包含本地嵌入基础模型或 API 密钥。'}
    (directory / 'model_delivery.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    (directory / 'requirements-model.txt').write_text('\n'.join(f'{name}=={value}' for name, value in packages.items()) + '\n', encoding='utf-8')
    return target


def load_trained_model(directory, *, trusted=False):
    """Load every model family with the same API; never silently retrain it."""
    if not trusted:
        raise ValueError('模型反序列化可能执行代码。请核验来源后明确传入 trusted=True。')
    import joblib
    directory = Path(directory)
    manifest = json.loads((directory / 'model_delivery.json').read_text(encoding='utf-8'))
    if manifest.get('schemaVersion') != 1 or manifest.get('file') != 'trained_model.joblib':
        raise ValueError('不支持的模型包格式')
    target = directory / manifest['file']
    if hashlib.sha256(target.read_bytes()).hexdigest() != manifest['sha256']:
        raise ValueError('模型文件校验失败，拒绝加载')
    return joblib.load(target)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='加载并检查可信训练模型，不重新训练')
    parser.add_argument('directory', type=Path)
    parser.add_argument('--trust-model', action='store_true', help='确认此模型来自可信来源')
    args = parser.parse_args()
    model = load_trained_model(args.directory, trusted=args.trust_model)
    print(f'已加载：{type(model).__module__}.{type(model).__name__}')
    print((args.directory / 'model_delivery.json').read_text(encoding='utf-8'))
