'''One JSON request / response transport for locally hosted capability workers.'''
import json
import os
import sys
import tempfile
from pathlib import Path


def configure_numba_cache() -> None:
    '''Give numba a writable cache directory before any worker imports it.

    UMAP's kernels use cache=True. Without NUMBA_CACHE_DIR the locator falls back
    to the read-only site-packages tree, document-projection figures abort with
    "cannot cache function 'rdist'", and reports are delivered as incomplete.
    Every routed or child worker inherits this setting.
    '''
    if os.environ.get('NUMBA_CACHE_DIR'):
        return
    target = Path(tempfile.gettempdir()) / 'theta-numba-cache'
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    os.environ['NUMBA_CACHE_DIR'] = str(target)


configure_numba_cache()

from . import capabilities, local_compute
from . import execution_policy, runtime_environments
from . import remote_report, figure_tools
from .dataset import catalog, business
from .dataset.preprocess import preprocess

def statistics_call(name, payload):
    # Discovery remains usable even if the optional numerical environment is absent.
    if name in {'catalog', 'inspect'}:
        from .statistics import registry
        return getattr(registry, name)(payload)
    from .statistics import engine
    return getattr(engine, name)(payload)

HANDLERS = {
    "statistics.methods": lambda p: statistics_call('catalog', p),
    "statistics.inspect": lambda p: statistics_call('inspect', p),
    "statistics.preview": lambda p: statistics_call('preview', p),
    "statistics.execute": lambda p: statistics_call('execute', p),
    "models.list": lambda _: [{"modelId": model, "description": description} for model, description in capabilities.MODELS.items()],
    "models.inspect": capabilities.model_inspect,
    "runtime.environments": runtime_environments.catalog,
    "runtime.environment": runtime_environments.inspect,
    "runtime.check": capabilities.runtime_check,
    "runtime.config": execution_policy.configuration_summary,
    "compute.preview": execution_policy.preview,
    "dataset.import": capabilities.dataset_import,
    "dataset.discover": catalog.discover,
    "dataset.use": catalog.use,
    "dataset.understand": business.understand,
    "dataset.profile": capabilities.dataset_profile,
    "dataset.preprocess": preprocess,
    "dataset.preview": capabilities.dataset_preview,
    "plan.validate": capabilities.validate_plan,
    "compute.submit": local_compute.submit,
    "compute.status": local_compute.status,
    "compute.cancel": local_compute.cancel,
    "compute.results": local_compute.results,
    "compute.remote_report": remote_report.generate,
    "figure.adjust": figure_tools.adjust,
}

if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "compute.run":
        local_compute.run(sys.argv[2], sys.argv[3])
    else:
        try:
            operation = sys.argv[1]
            raw = sys.stdin.read(2 * 1024 * 1024 + 1)
            if len(raw) > 2 * 1024 * 1024:
                raise ValueError("Worker 请求超出预算")
            payload = json.loads(raw)
            runtime_environments.route(operation, payload)
            data = HANDLERS[operation](payload)
            print(json.dumps({"ok": True, "data": data}, ensure_ascii=False, allow_nan=False))
        except Exception as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
            sys.exit(1)
