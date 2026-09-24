"""CI-only live check of the optional, publisher-verified Windows component."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from unittest.mock import patch

root=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(root/'agent'),str(root/'trainning')]
from workers.cuda_runtime import prepare_runtime
from workers.compute_device import resolve_device

assert sys.platform=='win32'
home=root/'desktop/.smoke-home'
with patch('workers.cuda_runtime.compatible_driver',return_value=True):
    runtime=prepare_runtime(str(home),lambda line:print(line,flush=True),lambda:False,time.monotonic()+3600)
assert runtime, 'Optional CUDA component failed to download, verify or install'
code='import sys; sys.path.insert(0,sys.argv[1]); import torch; assert torch.version.cuda=="12.8"; assert (torch.ones((2,2)) @ torch.ones((2,2))).sum().item()==8; print("Verified optional CUDA package and CPU operation",torch.__version__)'
subprocess.run([sys.executable,'-I','-c',code,str(runtime)],check=True,timeout=90)
# Hosted Windows runners have no NVIDIA GPU. The optional package must still let
# automatic selection fall back, without replacing the base interpreter's torch.
assert resolve_device({'device':'auto','modelId':'prodlda'},runtime)==('cpu','unavailable')
subprocess.run([sys.executable,'-I','-c','import torch; assert torch.version.cuda is None; print("Bundled CPU environment remains intact")'],check=True,timeout=60)
shutil.rmtree(runtime.parent)
print('Optional CUDA component verified and CI download cleaned.')
