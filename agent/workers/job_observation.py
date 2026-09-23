"""Read bounded progress signals from a job's own log; never estimate overall completion."""
import os
from pathlib import Path
import re
import time


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
    for line in text.replace('\r', '\n').splitlines():
        if line.startswith('COMMAND:'):
            result.update(iteration=None, activity=None)
        match = re.fullmatch(r'\s*iteration:\s*(\d+)\s+of max_iter:\s*(\d+)\s*', line)
        if phase == 'training' and match and 0 < int(match[1]) <= int(match[2]):
            result['iteration'] = {'current': int(match[1]), 'total': int(match[2]), 'source': 'worker_log', 'meaning': 'reported_iteration_not_completed_count'}
            result['activity'] = 'fitting'
        if phase == 'training' and re.search(r'Running Visualizations|Generating visualizations|Additional Visualizations|Covariate Visualizations', line):
            result.update(activity='visualizing', iteration=None)
    return result
