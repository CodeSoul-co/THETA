"""GPU selection helpers shared by model entry points."""

import os
from typing import List, Optional


def _validate_gpu_id(gpu_id: int) -> None:
    if gpu_id < 0:
        raise ValueError(f"GPU ID must be non-negative, got {gpu_id}")


def configure_cuda_visible_devices(
    gpu_id: Optional[int],
    default_gpu: int = 0,
) -> str:
    if gpu_id is not None:
        _validate_gpu_id(gpu_id)
        visible_devices = str(gpu_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
        return visible_devices

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return os.environ["CUDA_VISIBLE_DEVICES"]

    _validate_gpu_id(default_gpu)
    visible_devices = str(default_gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    return visible_devices


def append_gpu_argument(command: List[str], gpu_id: Optional[int]) -> List[str]:
    if gpu_id is not None:
        _validate_gpu_id(gpu_id)
        command.extend(["--gpu", str(gpu_id)])
    return command
