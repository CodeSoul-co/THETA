from __future__ import annotations

from pathlib import Path

from worker.config import RedisConfig, StorageConfig, WorkerConfig
from worker.protocol import ExecutionSpec


def execution_payload(model: str = "lda", gpu_count: int = 0) -> dict:
    return {
        "schema_version": 2,
        "type": "job.ready",
        "event_id": "ready-42",
        "task_id": 42,
        "attempt": 1,
        "task_version": 1,
        "user_id": "00000000-0000-0000-0000-000000000001",
        "priority": 10,
        "dataset": {
            "ref": "lda-smoke",
            "project_id": "10000000-0000-0000-0000-000000000001",
            "object_key": "datasets/dev/train.csv",
            "filename": "data.csv",
            "format": "csv",
            "sha256": "",
        },
        "model": {"id": 3, "name": model, "framework": "test"},
        "runtime": {
            "id": 4,
            "key": f"{model}-{'gpu' if gpu_count else 'cpu'}",
            "version": "v1",
            "executor": "theta_pipeline",
            "image": "theta-worker:test",
        },
        "params": {"num_topics": 20, "epochs": 2, "skip_viz": True},
        "resources": {
            "accelerator": "cuda" if gpu_count else "none",
            "cpu_cores": 2,
            "memory_mb": 2048,
            "gpu_count": gpu_count,
            "gpu_memory_mb": 4096 if gpu_count else 0,
            "timeout_seconds": 300,
        },
        "output_prefix": "training-results/42/attempt-1/",
        "created_at": "2026-08-21T00:00:00Z",
    }


def execution_spec(model: str = "lda", gpu_count: int = 0) -> ExecutionSpec:
    return ExecutionSpec.from_dict(execution_payload(model, gpu_count))


def worker_config(root: Path, resource_class: str = "cpu") -> WorkerConfig:
    redis = RedisConfig(
        url="",
        host="127.0.0.1",
        port=6379,
        username="",
        password="",
        database=0,
        job_stream=f"theta:jobs:{resource_class}",
        consumer_group=f"theta-workers-{resource_class}",
        consumer_name="test-worker",
        event_stream="theta:jobs:events",
        dead_letter_stream="theta:jobs:dead-letter",
        cancel_key_prefix="theta:job-cancel:",
        lock_key_prefix="theta:job-lock:",
        done_key_prefix="theta:job-done:",
        block_ms=100,
        batch_size=1,
        claim_idle_ms=1000,
        stream_max_length=1000,
        lock_ttl_seconds=60,
        done_ttl_seconds=3600,
    )
    storage = StorageConfig(
        driver="filesystem",
        endpoint="",
        bucket="test",
        region="us-east-1",
        access_key="",
        secret_key="",
        session_token="",
        use_ssl=False,
        filesystem_root=root / "objects",
    )
    return WorkerConfig(
        worker_id="test-worker",
        resource_class=resource_class,
        supported_runtime_ids=frozenset(),
        supported_models=frozenset(),
        project_root=root / "theta",
        job_root=root / "jobs",
        python_executable="python",
        gpu_id=0 if resource_class == "gpu" else None,
        keep_job_dir=True,
        heartbeat_interval_seconds=1,
        shutdown_grace_seconds=1,
        redis=redis,
        storage=storage,
    )
