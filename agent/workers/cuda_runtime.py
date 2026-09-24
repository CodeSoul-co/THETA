"""Verified, optional Windows CUDA component. The bundled CPU environment stays intact."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile

WHEEL_URL = 'https://download.pytorch.org/whl/cu128/torch-2.11.0%2Bcu128-cp312-cp312-win_amd64.whl'
WHEEL_SHA256 = '7c78215c3af4f62e63f2b2e360f1722fc719b0853c7ac22666483d9810613a4c'
WHEEL_SIZE = 2753189216
RUNTIME_NAME = 'torch-2.11.0-cu128'


def compatible_driver():
    if sys.platform != 'win32':
        return False
    code = '''import ctypes
cuda = ctypes.WinDLL('nvcuda.dll')
count, version = ctypes.c_int(), ctypes.c_int()
assert cuda.cuInit(0) == 0
assert cuda.cuDeviceGetCount(ctypes.byref(count)) == 0 and count.value > 0
assert cuda.cuDriverGetVersion(ctypes.byref(version)) == 0 and version.value >= 12080
print('THETA_DRIVER_READY')
'''
    try:
        result = subprocess.run([sys.executable, '-I', '-c', code], capture_output=True, text=True, timeout=10)
        return result.returncode == 0 and 'THETA_DRIVER_READY' in result.stdout.splitlines()
    except (OSError, subprocess.TimeoutExpired):
        return False


@contextmanager
def install_lock(root):
    # OS locks release on process exit; a crashed download cannot leave a stale lock.
    with (root / '.install.lock').open('a+b') as handle:
        if handle.tell() == 0:
            handle.write(b'0'); handle.flush()
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def ready(target):
    try:
        return (target / '.verified').read_text() == WHEEL_SHA256 and (target / 'torch/__init__.py').is_file()
    except OSError:
        return False


def download(destination, notify, check):
    offset = destination.stat().st_size if destination.exists() else 0
    if offset > WHEEL_SIZE:
        destination.unlink(); offset = 0
    if offset == WHEEL_SIZE:
        return
    request = urllib.request.Request(WHEEL_URL, headers={'Range': f'bytes={offset}-'} if offset else {})
    with urllib.request.urlopen(request, timeout=15) as response:
        if response.status == 206:
            expected = f'bytes {offset}-'
            if not response.headers.get('Content-Range', '').startswith(expected):
                raise OSError('Invalid CUDA download range')
        elif response.status == 200:
            offset = 0
        else:
            raise OSError('CUDA download failed')
        last = 0
        with destination.open('ab' if offset else 'wb') as handle:
            while True:
                check()
                chunk = response.read(4 * 1024 * 1024)
                if not chunk:
                    break
                offset += len(chunk)
                if offset > WHEEL_SIZE:
                    raise OSError('CUDA download exceeds expected size')
                handle.write(chunk)
                now = time.monotonic()
                if now - last >= 1 or offset == WHEEL_SIZE:
                    notify('downloading', offset, WHEEL_SIZE); last = now
    if offset != WHEEL_SIZE:
        raise OSError('Incomplete CUDA download')


def prepare_runtime(home, on_output, is_cancelled, deadline):
    if not compatible_driver():
        return None
    from worker.errors import JobCancelled
    def notify(status, current=None, total=None):
        event = {'device': 'cpu', 'status': status}
        if current is not None:
            event.update(current=current, total=total)
        on_output('THETA_DEVICE ' + json.dumps(event))
    # Both desktop modes share one component; source/CLI installations keep their own home.
    root = Path(os.environ.get('THETA_DESKTOP_HOME', home)).resolve() / 'runtime' / 'cuda'
    target = root / RUNTIME_NAME
    if ready(target):
        return target
    download_deadline = min(deadline, time.monotonic() + min(1200, max(1, (deadline-time.monotonic()) / 2)))
    def check():
        if is_cancelled():
            raise JobCancelled('用户已取消 GPU 组件准备')
        if time.monotonic() >= download_deadline:
            raise TimeoutError('CUDA setup exceeded its time budget')
    staging = None
    partial = root / 'torch.whl.part'
    try:
        root.mkdir(parents=True, exist_ok=True)
        with install_lock(root):
            if ready(target):
                return target
            check()
            # Wheel plus unpacked CUDA libraries; existing partial downloads count as available.
            required = 9 * 1024**3 - (partial.stat().st_size if partial.exists() else 0)
            if shutil.disk_usage(root).free < required:
                notify('disk_space'); return None
            notify('downloading', partial.stat().st_size if partial.exists() else 0, WHEEL_SIZE)
            download(partial, notify, check)
            digest = hashlib.sha256()
            with partial.open('rb') as handle:
                while chunk := handle.read(4 * 1024 * 1024):
                    check(); digest.update(chunk)
            if digest.hexdigest() != WHEEL_SHA256:
                partial.unlink(); raise OSError('CUDA component checksum mismatch')
            notify('installing')
            staging = Path(tempfile.mkdtemp(prefix='install-', dir=root))
            with zipfile.ZipFile(partial) as archive:
                for item in archive.infolist():
                    check()
                    (staging / item.filename).resolve().relative_to(staging)
                    destination = staging / item.filename
                    if item.is_dir():
                        destination.mkdir(parents=True, exist_ok=True)
                        continue
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(item) as source, destination.open('wb') as output:
                        while chunk := source.read(4 * 1024 * 1024):
                            check(); output.write(chunk)
            if not (staging / 'torch/__init__.py').is_file():
                raise OSError('CUDA component package missing torch')
            (staging / '.verified').write_text(WHEEL_SHA256)
            if target.exists():
                shutil.rmtree(target)
            staging.rename(target); staging = None
            partial.unlink(missing_ok=True)
            return target
    except JobCancelled:
        raise
    except (OSError, ValueError, zipfile.BadZipFile):
        notify('setup_failed')
        return None
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
