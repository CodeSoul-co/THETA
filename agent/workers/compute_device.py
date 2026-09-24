"""Select local acceleration in an isolated probe; never initialize CUDA in the host."""
import json
import os
import subprocess
import sys


def requested_device(plan):
    return plan.get('device') or ('auto' if sys.platform == 'win32' else 'cpu')


def resolve_device(plan):
    requested = requested_device(plan)
    if requested != 'auto':
        return requested, 'selected'
    if plan.get('modelId') in {'lda', 'btm', 'hdp', 'stm'}:
        return 'cpu', 'cpu_model'
    # Probe real allocation and kernel execution, not just driver enumeration.
    # A broken driver cannot poison the separate training process's CUDA context.
    probe = '''import torch
assert torch.cuda.is_available()
x = torch.ones((32, 32), device='cuda:0')
assert (x @ x).sum().item() == 32768
torch.cuda.synchronize()
print('THETA_CUDA_READY')
'''
    try:
        result = subprocess.run([sys.executable, '-I', '-c', probe],
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0'}, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and 'THETA_CUDA_READY' in result.stdout.splitlines():
            return 'cuda:0', 'selected'
    except (OSError, subprocess.TimeoutExpired):
        pass
    return 'cpu', 'unavailable'


def device_event(device, status):
    return 'THETA_DEVICE ' + json.dumps({'device': device, 'status': status})
