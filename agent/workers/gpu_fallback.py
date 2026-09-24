"""Recognize failed CUDA processes without retrying data errors, cancellation or timeout."""
import re
from worker.errors import ProcessExecutionError
from worker.process import ProcessRunner


class GPUExecutionError(ProcessExecutionError):
    def __init__(self, log_path):
        super().__init__('GPU 计算失败，可以使用 CPU 重新执行')
        self.log_path = log_path


# Only exception/error lines qualify. Do not mistake ordinary model output for failure.
GPU_ERROR = re.compile(
    r'(?:^|\s)(?:[\w.]*Error:|\[ERROR\]|\[Error\]).*'
    r'(?:CUDA|cuDNN|cuBLAS|CUDNN_STATUS|CUBLAS_STATUS|HIP error|not compiled with CUDA|NVIDIA driver)', re.I)


class GPUAwareProcessRunner(ProcessRunner):
    def run(self, **kwargs):
        gpu_error = False
        output = self.on_output
        def inspect(line):
            nonlocal gpu_error
            gpu_error = gpu_error or bool(GPU_ERROR.search(line))
            if output:
                output(line)
        self.on_output = inspect
        try:
            return super().run(**kwargs)
        except ProcessExecutionError:
            if gpu_error and kwargs['env'].get('THETA_COMPUTE_DEVICE', '').startswith('cuda:'):
                raise GPUExecutionError(kwargs['log_path']) from None
            raise
        finally:
            self.on_output = output
