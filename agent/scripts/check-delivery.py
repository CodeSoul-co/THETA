"""Validate one trusted, browser-downloaded training delivery without retraining."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import sys
import tempfile
from zipfile import ZipFile

parser = argparse.ArgumentParser(description='检查本次可信训练交付包的目录、模型加载与矩阵')
parser.add_argument('archive', type=Path)
parser.add_argument('--trusted', action='store_true', help='确认模型由可信训练任务生成，允许反序列化')
args = parser.parse_args()
if not args.trusted:
    parser.error('joblib 可执行代码，核验来源后才可使用 --trusted')

with tempfile.TemporaryDirectory(prefix='theta-delivery-verify-') as folder, ZipFile(args.archive) as archive:
    names = archive.namelist()
    for entry in archive.infolist():
        path = PurePosixPath(entry.filename)
        if path.is_absolute() or '..' in path.parts or '\\' in entry.filename or stat.S_ISLNK(entry.external_attr >> 16):
            raise ValueError('压缩包包含不安全路径')
        if path.name.startswith('.env'):
            raise ValueError('压缩包不应包含环境配置文件')
    if sum(entry.file_size for entry in archive.infolist()) > 4 * 1024**3:
        raise ValueError('交付包解压体积超出本次验证上限')
    assert '使用方法.md' in names, '缺少中文使用文档或文件名编码错误'
    assert '代码/src/models/model_delivery.py' in names, '缺少固定加载入口'
    assert any(name.endswith('.png') for name in names), '缺少真实绘图'
    assert any(name.endswith('.csv') for name in names), '缺少绘图数据'
    archive.extractall(folder)
    root = Path(folder)
    manifests = list(root.rglob('model_delivery.json'))
    assert len(manifests) == 1, '单模型交付应包含一份模型清单'
    directory = manifests[0].parent
    manifest = json.loads(manifests[0].read_text(encoding='utf-8'))
    sys.path.insert(0, str(root / '代码/src/models'))
    from model_delivery import load_trained_model
    import numpy as np
    model = load_trained_model(directory, trusted=True)
    beta = model.get_beta()
    if hasattr(beta, 'detach'):
        beta = beta.detach().cpu().numpy()
    beta = np.asarray(beta)
    vocab = json.loads((directory / 'vocab.json').read_text(encoding='utf-8'))
    expected_dims = 3 if manifest['modelId'] == 'dtm' else 2
    assert beta.ndim == expected_dims and beta.shape[-1] == len(vocab) and np.isfinite(beta).all(), '主题词矩阵或词表不一致'
    if manifest['modelId'] == 'dtm':
        first_slice = model.get_beta(time_index=0).detach().cpu().numpy()
        np.testing.assert_allclose(first_slice, beta[0])
    print(json.dumps({'模型': manifest['modelId'], '文件数': len(names), '主题词矩阵': list(beta.shape), '统一加载': '通过', '压缩包SHA256': hashlib.sha256(args.archive.read_bytes()).hexdigest()}, ensure_ascii=False))
