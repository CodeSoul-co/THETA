"""Read bounded progress signals from a job's own log; never estimate overall completion."""
import os
import json
from pathlib import Path
import re
import time
from .progress_events import parse_progress


def observe(home, job, updated, now=None):
    now = time.time() if now is None else now
    phase = job.get('phase', 'unknown')
    active = job['status'] in {'queued', 'running'}
    heartbeat_age = max(0, now - updated) if active else None
    result = {'schemaVersion': 'theta.job-observation.v1', 'observedAt': now,
              'percentKind': 'stage_marker', 'heartbeatAgeSeconds': heartbeat_age,
              'health': ('waiting' if job['status'] == 'queued' else 'responding') if active and heartbeat_age <= 30 else 'unresponsive' if active else 'finished',
              'elapsedSeconds': None, 'phaseElapsedSeconds': None,
              'lastLogAgeSeconds': None, 'iteration': None, 'activity': None,
              'events': [], 'detail': None,
              'limitation': '阶段百分比不是实际完成比例；心跳只证明worker仍响应，不证明算法推进或模型质量。'}
    end = job.get('finishedAt', now)
    if job.get('startedAt') is not None:
        result['elapsedSeconds'] = max(0, end - job['startedAt'])
    if job.get('phaseStartedAt') is not None:
        result['phaseElapsedSeconds'] = max(0, end - job['phaseStartedAt'])
    if not re.fullmatch(r'job-[0-9a-f]{64}', job['id']):
        return result
    root = Path(home).resolve() / 'compute' / job['id']
    try:
        paths = list((root / 'jobs').glob('task-*-attempt-*/worker.log'))
        paths = [p for p in paths if p.resolve() == p and p.is_file()]
        if not paths:
            return result
        file = max(paths, key=lambda p: p.stat().st_mtime)
        with os.fdopen(os.open(file, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)), 'rb') as handle:
            size = handle.seek(0, os.SEEK_END)
            handle.seek(max(0, size - 32768))
            text = handle.read(32768).decode('utf-8', errors='replace')
            result['lastLogAgeSeconds'] = max(0, now - os.fstat(handle.fileno()).st_mtime)
    except OSError:
        return result
    # Only expose recognized signals, never raw dataset text, paths or credentials.
    events = []
    journal = root / 'progress.jsonl'
    if journal.is_file() and journal.resolve() == journal:
        try:
            with os.fdopen(os.open(journal, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)), 'rb') as handle:
                size = handle.seek(0, os.SEEK_END)
                handle.seek(max(0, size - 262144))
                records = handle.read(262144).split(b'\n')
            # The final record can still be in flight. Keep only complete JSON lines.
            for record in records[:-1]:
                try:
                    event = json.loads(record)
                    if isinstance(event, dict) and event.get('kind') in {'epoch', 'iteration', 'batch', 'early_stop', 'command', 'visualizing', 'embedding', 'device'}:
                        events.append(event)
                except (ValueError, UnicodeDecodeError):
                    pass
        except OSError:
            pass
    if not events:
        # Compatibility for jobs launched before structured recording was introduced.
        for index, line in enumerate(text.replace('\r', '\n').splitlines()):
            event = parse_progress(line)
            if event:
                events.append({**event, 'id': f'legacy-{index}', 'at': None})
    result['events'] = events[-200:]
    for event in events:
        kind = event['kind']
        if kind == 'device':
            result['computeDevice'] = event['device']
            continue
        if kind in {'command', 'visualizing'}:
            result.update(iteration=None, detail=None, activity='visualizing' if kind == 'visualizing' and phase == 'training' else None)
        elif phase == 'training' or active and (kind == 'embedding' or kind == 'batch' and event.get('activity')):
            result['detail'] = event
            result['activity'] = 'embedding' if kind == 'embedding' else 'fitting'
            if kind == 'iteration':
                result['iteration'] = {'current': event['current'], 'total': event['total'], 'source': 'worker_log', 'meaning': 'reported_iteration_not_completed_count'}
    return result
