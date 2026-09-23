"""Byte-bound row lineage for CSV cleaning and model preprocessing; no labels."""
import hashlib
import json
from pathlib import Path


def text_hash(text):
    return hashlib.sha256(str(text).encode('utf-8')).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record_cleaning(source, cleaned, original_texts, cleaned_texts):
    if len(original_texts) != len(cleaned_texts):
        raise ValueError('Cleaning lineage requires one output per input row')
    record = {'schema': 'theta.csv-cleaning-lineage',
              'sourceFileSha256': file_hash(source), 'cleanedFileSha256': file_hash(cleaned),
              'rows': [{'sourceRow': i, 'sourceTextSha256': text_hash(raw),
                        'modelTextSha256': text_hash(text)}
                       for i, (raw, text) in enumerate(zip(original_texts, cleaned_texts))]}
    Path(str(cleaned) + '.lineage.json').write_text(json.dumps(record), encoding='utf-8')


def export_lineage(data_path, destination, texts, source_rows):
    """Export only fresh, verified cleaning lineage for the selected model rows.

    Absence means legacy/uncleaned input: no stronger alignment claim is made.
    Malformed or stale lineage fails preprocessing rather than fabricating a map.
    """
    sidecar = Path(str(data_path) + '.lineage.json')
    target = Path(destination) / 'row_provenance.json'
    if not sidecar.is_file():
        # A forced rebuild must not retain a previous cleaning run's lineage.
        if target.is_file():
            target.unlink()
        return
    record = json.loads(sidecar.read_text(encoding='utf-8'))
    if record.get('schema') != 'theta.csv-cleaning-lineage' or record.get('cleanedFileSha256') != file_hash(data_path):
        raise ValueError('Cleaning input changed or lineage schema is invalid')
    indices = [int(i) for i in source_rows]
    if len(indices) != len(texts) or len(set(indices)) != len(indices):
        raise ValueError('Model row indices must be unique and match exported texts')
    rows = record['rows']
    selected = []
    for index, text in zip(indices, texts):
        if index < 0 or index >= len(rows) or rows[index]['sourceRow'] != index or rows[index]['modelTextSha256'] != text_hash(text):
            raise ValueError('Model texts do not match cleaning row lineage')
        selected.append(rows[index])
    target.write_text(json.dumps({'schema': 'theta.model-row-lineage',
        'sourceFileSha256': record['sourceFileSha256'], 'cleanedFileSha256': record['cleanedFileSha256'],
        'rows': selected}), encoding='utf-8')


def verify_lineage(record, source_file_hash, original_texts, model_texts, indices):
    """Check source identity AND each selected original/cleaned row's text hash."""
    if record.get('schema') != 'theta.model-row-lineage' or record.get('sourceFileSha256') != source_file_hash:
        return False
    rows = record.get('rows', [])
    if len(rows) != len(model_texts) or len(rows) != len(indices):
        return False
    try:
        return all(type(row['sourceRow']) is int and row['sourceRow'] == int(index)
                   and 0 <= int(index) < len(original_texts)
                   and row['sourceTextSha256'] == text_hash(original_texts[int(index)])
                   and row['modelTextSha256'] == text_hash(text)
                   for row, text, index in zip(rows, model_texts, indices))
    except (KeyError, TypeError, ValueError):
        return False
