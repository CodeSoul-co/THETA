"""Create a traceable document collection from host-verified uploaded datasets."""
import json
import tempfile
from pathlib import Path

TEXT_SUFFIXES = {'.txt', '.md', '.pdf', '.docx'}


def combine(payload):
    from ..capabilities import dataset_import, verify_dataset
    from .readers import load_dataset
    datasets = payload.get('datasets', [])
    if not 1 <= len(datasets) <= 500:
        raise ValueError('文档集合需要 1–500 个文件。')
    if sum(item.get('sizeBytes', 0) for item in datasets) > 200 * 1024 * 1024:
        raise ValueError('文档集合总大小不能超过 200 MiB，请分批上传。')
    root = Path(payload['uploadDir'])
    root.mkdir(parents=True, exist_ok=True)
    count = 0
    with tempfile.TemporaryDirectory(dir=root) as temporary:
        combined = Path(temporary) / '文档合集.jsonl'
        with combined.open('w', encoding='utf-8') as output:
            for dataset in datasets:
                source = verify_dataset(dataset)
                if source.suffix.lower() not in TEXT_SUFFIXES:
                    raise ValueError(f"{dataset['fileName']} 不是正文文档。文件夹支持 TXT、Markdown、PDF、DOCX；表格请单独上传并选择正文列。")
                reader = load_dataset(source, profile_limit=1_000_000)
                if reader.rows_truncated:
                    raise ValueError('单个文档超过 100 万条正文，请拆分文件后上传。')
                if not reader.row_count:
                    raise ValueError(f"{dataset['fileName']} 未读取到正文。扫描文档请先完成文字识别后重新上传。")
                for row in reader.rows:
                    count += 1
                    if count > 1_000_000:
                        raise ValueError('正文记录超过 100 万条，请拆分文档集合。')
                    output.write(json.dumps({**row, 'source_file': dataset['fileName'], 'source_dataset': dataset['datasetRef']}, ensure_ascii=False) + '\n')
        result = dataset_import({'filePath': str(combined), 'uploadDir': str(root)})
    return {**result, 'inputKind': 'text', 'documentCount': len(datasets), 'recordCount': count}
