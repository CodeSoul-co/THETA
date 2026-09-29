"""Combine every source row without sampling or limiting collection size."""
import json
import tempfile
from pathlib import Path

TEXT_SUFFIXES = {'.txt', '.md', '.pdf', '.docx'}

def combine(payload):
    from ..capabilities import dataset_import, verify_dataset
    from .readers import visit_dataset_rows, load_dataset
    datasets = payload.get('datasets', [])
    if not datasets:
        raise ValueError('请选择至少一个文件。')
    root = Path(payload['uploadDir'])
    root.mkdir(parents=True, exist_ok=True)
    count = 0
    with tempfile.TemporaryDirectory(dir=root) as temporary:
        combined = Path(temporary) / '文档合集.jsonl'
        with combined.open('w', encoding='utf-8') as output:
            for dataset in datasets:
                source = verify_dataset(dataset)
                column = dataset.get('textColumn')
                if source.suffix.lower() in TEXT_SUFFIXES or dataset.get('inputKind') == 'text':
                    column = 'text'
                if not column:
                    columns = load_dataset(source, profile_limit=10).columns
                    candidates = [c for c in columns if c.lower() in {'text', 'content', 'body', '正文', '内容', '文本'}]
                    column = candidates[0] if len(candidates) == 1 else columns[0] if len(columns) == 1 else None
                if not column:
                    raise ValueError(f"请为 {dataset['fileName']} 选择正文列后再合并。")
                before = count
                def write_row(row):
                    nonlocal count
                    if column not in row:
                        raise ValueError(f"{dataset['fileName']} 缺少正文列 {column}，请重新选择。")
                    text = row.get(column)
                    if text is None or not str(text).strip():
                        return
                    output.write(json.dumps({**row, 'text': str(text), 'source_file': dataset['fileName'], 'source_dataset': dataset['datasetRef']}, ensure_ascii=False) + '\n')
                    count += 1
                visit_dataset_rows(source, write_row)
                if count == before:
                    raise ValueError(f"{dataset['fileName']} 未读取到正文。请检查所选正文列；扫描文档请先完成文字识别。")
        result = dataset_import({'filePath': str(combined), 'uploadDir': str(root)})
    return {**result, 'inputKind': 'text', 'documentCount': len(datasets), 'recordCount': count}
