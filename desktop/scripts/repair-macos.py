"""Make all bundled native libraries relocatable; never depend on Homebrew."""
import os
from pathlib import Path
import sys
from delocate.tools import get_install_names, set_install_name
from delocate.delocating import delocate_path

root = Path(sys.argv[1]).resolve()
# PyTorch already ships a redistributable OpenMP runtime. Numba's macOS wheel
# lacks an rpath; LightGBM otherwise resolves Homebrew on the build machine.
omp = next(root.glob('lib/python*/site-packages/torch/lib/libomp.dylib'))
for file in root.rglob('*'):
    if file.is_symlink() or file.suffix not in {'.so', '.dylib'}:
        continue
    for dependency in get_install_names(str(file)):
        if dependency.endswith('/libomp.dylib') and file != omp:
            relative = os.path.relpath(omp, file.parent)
            replacement = '@loader_path/' + relative
            if dependency != replacement:
                set_install_name(str(file), dependency, replacement)
delocate_path(str(root), str(root / '.dylibs'), lib_filt_func='dylibs-only', sanitize_rpaths=True)
print('Native libraries are self-contained; no Homebrew dependency remains.')
