"""Persist numeric training signals without copying corpus text or command arguments."""
import json
import math
from pathlib import Path
import re
import time

NUMBER = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
METRICS = {'loss': 'loss', 'train': 'train_loss', 'train_loss': 'train_loss',
           'val': 'val_loss', 'val_loss': 'val_loss', 'recon': 'recon_loss',
           'recon_loss': 'recon_loss', 'kl': 'kl_loss', 'kl_loss': 'kl_loss',
           'ppl': 'perplexity', 'perplexity': 'perplexity', 'nll': 'recon_loss',
           'ce': 'ce_loss', 'contr': 'contrastive_loss'}


def parse_progress(line):
    line = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', line).strip()
    if line.startswith('THETA_DEVICE '):
        try:
            payload = json.loads(line[len('THETA_DEVICE '):])
            if payload.get('status') not in {'selected', 'cpu_model', 'unavailable', 'fallback'}:
                return None
            device = payload.get('device', '')
            if device != 'cpu' and not re.fullmatch(r'cuda:[0-9]+', device):
                return None
            return {'kind': 'device', 'device': device, 'status': payload['status']}
        except (ValueError, TypeError, AttributeError):
            return None
    if line.startswith('THETA_EMBEDDING '):
        try:
            payload = json.loads(line[len('THETA_EMBEDDING '):])
            if payload.get('source') not in {'local', 'cloud'} or payload.get('scope') not in {'documents', 'vocabulary'}:
                return None
            event = {key: payload[key] for key in ('source', 'scope')}
            for key in ('current', 'total', 'completedBatches', 'chunks', 'chunkTotal', 'totalBatches'):
                if key in payload:
                    if type(payload[key]) is not int or payload[key] < 0:
                        return None
                    event[key] = payload[key]
            if not 0 <= event['current'] <= event['total'] or event['total'] == 0:
                return None
            return {'kind': 'embedding', **event}
        except (ValueError, KeyError, TypeError, AttributeError):
            return None
    if line.startswith('COMMAND:'):
        return {'kind': 'command'}
    if re.search(r'Running Visualizations|Generating visualizations|Additional Visualizations|Covariate Visualizations', line):
        return {'kind': 'visualizing'}
    match = re.fullmatch(r'iteration:\s*(\d+)\s+of max_iter:\s*(\d+)', line)
    if match and 0 < int(match[1]) <= int(match[2]):
        return {'kind': 'iteration', 'current': int(match[1]), 'total': int(match[2])}
    match = re.search(r'Early stopping at epoch\s+(\d+)', line, re.I)
    if match:
        return {'kind': 'early_stop', 'current': int(match[1])}
    epoch = re.search(r'(?:(Stage[12])\s+)?Epoch\s+(\d+)(?:\s*/\s*(\d+))?', line, re.I)
    batch = re.search(r'\|\s*(\d+)\s*/\s*(\d+)\s*\[', line)
    activity = next((label for label in ['Generating embeddings', 'Embedding vocabulary', 'Cleaning text', 'Tokenizing', 'BOW', 'Batches'] if line.startswith(label)), None)
    if not epoch and not (batch and activity):
        return None
    event = {'kind': 'epoch' if epoch else 'batch'}
    if epoch:
        current, total = int(epoch[2]), int(epoch[3]) if epoch[3] else None
        if current < 1 or (total is not None and not 0 < current <= total):
            return None
        event.update(current=current, total=total, stage=epoch[1].lower() if epoch[1] else None)
    if batch and 0 <= int(batch[1]) <= int(batch[2]) and int(batch[2]) > 0:
        event['batch'] = {'current': int(batch[1]), 'total': int(batch[2])}
        event['kind'] = 'batch'
    elif not epoch or epoch[3] is None:
        return None
    if activity:
        event['activity'] = activity
    metrics = {}
    for key, value in re.findall(r'\b(train_loss|val_loss|recon_loss|kl_loss|loss|train|val|recon|kl|ppl|perplexity|nll|ce|contr)\s*[:=]\s*(' + NUMBER + r')(?![\w.])', line, re.I):
        number = float(value)
        if math.isfinite(number):
            metrics[METRICS[key.lower()]] = number
    if metrics:
        event['metrics'] = metrics
    return event


class ProgressRecorder:
    def __init__(self, file):
        self.file = Path(file)
        self.sequence = 0
        self.previous = None
        self.last_batch_at = 0

    def __call__(self, line):
        for part in line.replace('\r', '\n').splitlines():
            event = parse_progress(part)
            if not event or event == self.previous:
                continue
            now = time.time()
            incremental = event['kind'] in {'batch', 'embedding'}
            count = event['batch'] if event['kind'] == 'batch' else event
            if incremental and now - self.last_batch_at < 1 and count['current'] < count['total']:
                continue
            if incremental:
                self.last_batch_at = now
            self.previous = event
            self.sequence += 1
            self.file.parent.mkdir(parents=True, exist_ok=True)
            with self.file.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps({**event, 'id': self.sequence, 'at': now}, allow_nan=False) + '\n')
