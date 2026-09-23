"""Bounded, read-only failure evidence from a job's own logs, never arbitrary paths."""
import os
from pathlib import Path
import re


def redact(text):
    for key, value in os.environ.items():
        if len(value) >= 4 and any(word in key.upper() for word in ('KEY', 'SECRET', 'TOKEN', 'PASSWORD')):
            text = text.replace(value, '[REDACTED]')
    text = re.sub(r'(?i)(Bearer\s+|(?:api[_-]?key|token|password|secret)\s*[=:]\s*)[^\s,;]+', r'\1[REDACTED]', text)
    text = re.sub(r'https?://\S+', '[endpoint]', text)
    text = re.sub(r'(?:/[^\s:]+)+', '[path]', text)
    return re.sub(r'[\x00-\x08\x0b-\x1f\x7f]', '', text)[:1200]


def failure_diagnostics(home, job_id):
    if not re.fullmatch(r'job-[0-9a-f]{64}', job_id):
        return {'available': False, 'reason': '无效任务标识'}
    root = Path(home).resolve() / 'compute' / job_id
    if root.resolve() != root:
        return {'available': False, 'reason': '日志目录越过任务边界'}
    paths = sorted((root / 'jobs').glob('task-*-attempt-*/worker.log'))[-4:]
    if not paths:
        paths = [root / 'host.log']
    evidence = []
    stage = 'unknown'
    for file in paths:
        try:
            if file.is_symlink() or file.resolve() != file:
                continue
            file.resolve().relative_to(root)
            descriptor = os.open(file, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
            with os.fdopen(descriptor, 'rb') as handle:
                size = handle.seek(0, os.SEEK_END)
                handle.seek(max(0, size - 32_768))
                text = handle.read(32_768).decode('utf-8', errors='replace')
        except (OSError, ValueError):
            continue
        commands = re.findall(r'^COMMAND: (.+)$', text, re.MULTILINE)
        if commands:
            stage = 'preparing' if 'prepare_data.py' in commands[-1] else 'training' if 'run_pipeline.py' in commands[-1] else 'unknown'
        frames = []
        for line in text.splitlines():
            if line.startswith('Traceback (most recent call last):'):
                frames = []
            frame = re.match(r'^\s+File "(.+)", line (\d+), in ([\w<>]+)', line)
            if frame:
                frames = [*frames, {'file': redact(Path(frame[1]).name), 'line': int(frame[2]), 'function': redact(frame[3])}][-6:]

            argument = re.match(r'^\S+\.py: error: (.+)$', line)
            exception = re.match(r'^(?:[\w.]+(?:Error|Exception)|KeyboardInterrupt): (.+)$', line)
            if argument or exception:
                evidence.append({'source': file.name, 'category': 'invalid_parameter' if argument else 'runtime_error',
                                 'message': redact(line), 'frames': list(frames) if exception else []})
    if not evidence:
        return {'available': False, 'stage': stage, 'reason': '日志中没有可提取的错误摘要；不能仅凭退出码判断失败原因'}
    exporting = any(frame['function'] in {'export_baseline_metadata', 'write_topic_result_manifest', 'save_results'} for frame in evidence[-1]['frames'])
    return {'available': True, 'stage': 'exporting_results' if exporting else stage, 'processStage': stage,
            'category': 'result_export_error' if exporting else evidence[-1]['category'],
            'location': evidence[-1]['frames'][-1] if evidence[-1]['frames'] else None,
            'instruction': '调用栈定位于结果导出；进程阶段 training 不等于拟合失败。' if exporting else '只根据已有错误与调用栈解释，不推测未知根因。',
            'message': evidence[-1]['message'], 'evidence': evidence[-3:],
            'bounded': True, 'untrustedEvidence': True}
