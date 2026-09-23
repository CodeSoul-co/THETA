"""Host-owned runtime routing. Model requests never supply executables or images."""
from __future__ import annotations

import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

PROFILES = {
    'classic': ('lda', 'btm', 'hdp', 'dtm', 'stm'),
    'neural': ('etm', 'nvdm', 'gsm', 'prodlda', 'ctm', 'bertopic', 'theta'),
    'reports': (),
    'statistics': (),
}


def model_profile(model: str) -> str:
    for profile, models in PROFILES.items():
        if model.lower() in models:
            return profile
    raise ValueError(f'未登记模型运行环境：{model}')


def selected(profile: str) -> dict:
    if profile not in PROFILES:
        raise ValueError('未登记的 worker 环境')
    key = 'THETA_WORKER_' + profile.upper() + '_PYTHON'
    configured = os.environ.get(key, '').strip()
    if profile == 'statistics' and not configured:
        # Statistics is opt-in and isolated; never silently use the topic environment.
        configured = str(Path(__file__).resolve().parents[1] / '.local' / 'runtimes' / 'statistics' / 'bin' / 'python')
    # Do not resolve symlinks: different venvs often link to the same base Python.
    executable = configured or os.environ.get('THETA_WORKER_CONTROL_PYTHON') or sys.executable
    executable = os.path.abspath(os.path.expanduser(executable)) if os.path.dirname(executable) else shutil.which(executable) or executable
    return {'profile': profile, 'revision': os.environ.get('THETA_WORKER_' + profile.upper() + '_REVISION', 'v1'),
            'python': executable, 'mode': 'configured' if configured else 'shared',
            'available': Path(executable).is_file() and os.access(executable, os.X_OK),
            'configurationVariable': key, 'models': list(PROFILES[profile])}


def catalog(_: dict) -> list:
    return [selected(profile) for profile in PROFILES]


def identity(profile: str) -> dict:
    runtime = selected(profile)
    if os.path.abspath(sys.executable) != runtime['python']:
        raise ValueError(f"必须在 {profile} 环境执行：{runtime['python']}")
    packages = sorted((d.metadata.get('Name', '').lower(), d.version) for d in metadata.distributions())
    config = Path(sys.prefix) / 'pyvenv.cfg'
    inherits = config.is_file() and any(line.strip().lower() == 'include-system-site-packages = true' for line in config.read_text().splitlines())
    version = sys.version.split()[0]
    digest = hashlib.sha256(json.dumps([version, sys.prefix, packages], sort_keys=True).encode()).hexdigest()
    return {**runtime, 'pythonVersion': version, 'prefix': sys.prefix, 'dependencyFingerprint': digest,
            'isolatedVenv': sys.prefix != sys.base_prefix and not inherits and bool(sys.flags.no_user_site),
            'inheritsSystemPackages': inherits, 'sandbox': False}


def inspect(payload: dict) -> dict:
    return identity(payload['profile'])


def route(operation: str, payload: dict) -> None:
    """Replace the transport process, preserving PID, stdin JSON and cancellation.

    No proxy subprocess remains between Node and the selected worker. Detached
    training then inherits this interpreter through the existing submit path.
    """
    profile = None
    if operation in {'statistics.preview', 'statistics.execute'}:
        profile = 'statistics'
    if operation in {'runtime.check', 'compute.preview', 'compute.submit'}:
        profile = model_profile((payload.get('plan') or payload).get('modelId', ''))
    elif operation == 'figure.adjust' or operation == 'compute.results' and payload.get('view') != 'artifacts':
        profile = 'reports'
    elif operation == 'runtime.environment':
        profile = payload.get('profile')
    if profile is None:
        return
    runtime = selected(profile)
    if not runtime['available']:
        raise ValueError(f"{profile} 环境 Python 不可执行：{runtime['python']}；请修正 {runtime['configurationVariable']}，不会回退到其他环境")
    if (os.path.abspath(sys.executable) == runtime['python'] and sys.flags.no_user_site
            and not os.environ.get('PYTHONPATH') and not os.environ.get('PYTHONHOME')):
        return
    if os.environ.get('THETA_WORKER_ROUTED_PYTHON') == runtime['python']:
        raise ValueError('配置的 Python 未进入所选环境，请使用该环境的 bin/python 绝对路径')
    environment = dict(os.environ)
    environment.setdefault('THETA_WORKER_CONTROL_PYTHON', sys.executable)
    environment['THETA_WORKER_ROUTED_PYTHON'] = runtime['python']
    environment['PYTHONNOUSERSITE'] = '1'
    if profile == 'statistics':
        environment.update({'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'})
    environment.pop('PYTHONPATH', None)
    environment.pop('PYTHONHOME', None)
    with tempfile.TemporaryFile() as request:
        request.write(json.dumps(payload, ensure_ascii=False).encode())
        request.seek(0)
        os.dup2(request.fileno(), 0, inheritable=True)
        os.execve(runtime['python'], [runtime['python'], '-m', 'workers', operation], environment)
