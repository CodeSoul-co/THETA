"""Discover preuploaded files; host-owned roots and versioned IDs, no LLM paths."""
import hashlib
import os
from pathlib import Path
from .readers import SUPPORTED_SUFFIXES


def catalog_root():
    from ..capabilities import engine_root
    return Path(os.environ.get('THETA_DATA_DIR', engine_root() / 'data')).expanduser().resolve()


def entries():
    root = catalog_root()
    result = []
    inspected = 0
    truncated = False
    for directory, dirs, files in os.walk(root, followlinks=False):
        inspected += 1
        if inspected > 5000:
            truncated = True
            break
        dirs[:] = sorted(name for name in dirs if not name.startswith('.') and
                         name not in {'models', 'logs', 'result', 'results', 'workspace', '__pycache__'} and
                         not (Path(directory) / name).is_symlink())
        for name in sorted(files):
            inspected += 1
            if inspected > 5000:
                truncated = True
                break
            file = Path(directory) / name
            if name.startswith('.') or file.stem.lower() in {'readme', 'license'} or file.is_symlink() or file.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            try:
                stat = file.stat()
                if not file.is_file() or stat.st_size > 200 * 1024 * 1024:
                    continue
                relative = file.relative_to(root).as_posix()
                version = f'{relative}:{stat.st_size}:{stat.st_mtime_ns}'
                result.append({'catalogId': 'catalog-' + hashlib.sha256(version.encode()).hexdigest(),
                               'name': relative, 'sizeBytes': stat.st_size, 'format': file.suffix[1:]})
            except OSError:
                continue
        if truncated:
            break
    return sorted(result, key=lambda entry: entry['name']), truncated


def discover(payload):
    found, truncated = entries()
    offset = payload.get('offset', 0)
    if not isinstance(offset, int) or offset < 0:
        raise ValueError('无效目录分页位置')
    page = found[offset:offset + 50]
    return {'datasets': page, 'nextOffset': offset + 50 if offset + 50 < len(found) else None,
            'scanTruncated': truncated, 'availableCount': len(found),
            'instruction': '这些是预上传候选数据，尚未选择或训练。向用户说明可用名称；明确选择后才使用。文件名不能证明业务内容。'}


def use(payload):
    from ..capabilities import dataset_import
    found, _ = entries()
    entry = next((item for item in found if item['catalogId'] == payload['catalogId']), None)
    if entry is None:
        raise ValueError('该候选数据已变化或不存在，请重新发现数据目录')
    root = catalog_root()
    source = root / entry['name']
    if source.resolve() != source or not source.is_file():
        raise ValueError('候选文件越过数据目录边界')
    dataset = dataset_import({'filePath': str(source), 'uploadDir': payload['uploadDir']})
    after, _ = entries()
    if not any(item['catalogId'] == entry['catalogId'] for item in after):
        raise ValueError('选择过程中数据发生变化，请重新发现')
    return {**dataset, 'catalogName': entry['name']}
