from __future__ import annotations

import os
import socket
import tempfile
from dataclasses import dataclass
from pathlib import Path


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _positive_int(name: str, default: int) -> int:
    raw = _env(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _bool(name: str, default: bool = False) -> bool:
    raw = _env(name, "true" if default else "false").lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _csv(name: str) -> frozenset[str]:
    return frozenset(item.strip() for item in _env(name).split(",") if item.strip())


def _redis_host_port() -> tuple[str, int]:
    address = _env("REDIS_ADDR")
    if address and ":" in address:
        host, port = address.rsplit(":", 1)
        try:
            parsed_port = int(port)
        except ValueError as exc:
            raise ValueError("REDIS_ADDR port must be an integer") from exc
        if not host or parsed_port <= 0:
            raise ValueError("REDIS_ADDR must contain a host and positive port")
        return host, parsed_port
    return _env("REDIS_HOST", "127.0.0.1"), _positive_int("REDIS_PORT", 6379)


@dataclass(frozen=True)
class RedisConfig:
    url: str
    host: str
    port: int
    username: str
    password: str
    database: int
    job_stream: str
    consumer_group: str
    consumer_name: str
    event_stream: str
    dead_letter_stream: str
    cancel_key_prefix: str
    lock_key_prefix: str
    done_key_prefix: str
    block_ms: int
    batch_size: int
    claim_idle_ms: int
    stream_max_length: int
    lock_ttl_seconds: int
    done_ttl_seconds: int


@dataclass(frozen=True)
class StorageConfig:
    driver: str
    endpoint: str
    bucket: str
    region: str
    access_key: str
    secret_key: str
    session_token: str
    use_ssl: bool
    filesystem_root: Path


@dataclass(frozen=True)
class WorkerConfig:
    worker_id: str
    resource_class: str
    supported_runtime_ids: frozenset[str]
    supported_models: frozenset[str]
    project_root: Path
    job_root: Path
    python_executable: str
    gpu_id: int | None
    keep_job_dir: bool
    heartbeat_interval_seconds: int
    shutdown_grace_seconds: int
    redis: RedisConfig
    storage: StorageConfig

    @classmethod
    def load(cls) -> "WorkerConfig":
        resource_class = _env("WORKER_RESOURCE_CLASS", "cpu").lower()
        if resource_class not in {"cpu", "gpu"}:
            raise ValueError("WORKER_RESOURCE_CLASS must be cpu or gpu")

        worker_id = _env("WORKER_ID") or f"{socket.gethostname()}-{resource_class}"
        cpu_stream = _env("REDIS_STREAM_CPU_JOBS", "theta:jobs:cpu")
        gpu_stream = _env("REDIS_STREAM_GPU_JOBS", "theta:jobs:gpu")
        job_stream = gpu_stream if resource_class == "gpu" else cpu_stream
        group_default = f"theta-workers-{resource_class}"

        module_path = Path(__file__).resolve()
        default_project_root = module_path.parents[2]
        project_root = Path(_env("THETA_PROJECT_ROOT", str(default_project_root))).resolve()
        default_job_root = Path(tempfile.gettempdir()) / "theta-worker" / "jobs"
        job_root = Path(_env("WORKER_JOB_ROOT", str(default_job_root))).resolve()

        redis_database = int(_env("REDIS_DB", "0"))
        if redis_database < 0:
            raise ValueError("REDIS_DB must be non-negative")
        redis_host, redis_port = _redis_host_port()
        redis_config = RedisConfig(
            url=_env("REDIS_URL"),
            host=redis_host,
            port=redis_port,
            username=_env("REDIS_USERNAME"),
            password=_env("REDIS_PASSWORD"),
            database=redis_database,
            job_stream=job_stream,
            consumer_group=_env("REDIS_WORKER_CONSUMER_GROUP") or group_default,
            consumer_name=_env("REDIS_WORKER_CONSUMER_NAME") or worker_id,
            event_stream=_env("REDIS_STREAM_JOB_EVENTS", "theta:jobs:events"),
            dead_letter_stream=_env("REDIS_STREAM_DEAD_LETTER", "theta:jobs:dead-letter"),
            cancel_key_prefix=_env("REDIS_CANCEL_KEY_PREFIX", "theta:job-cancel:"),
            lock_key_prefix=_env("REDIS_JOB_LOCK_KEY_PREFIX", "theta:job-lock:"),
            done_key_prefix=_env("REDIS_JOB_DONE_KEY_PREFIX", "theta:job-done:"),
            block_ms=_positive_int("REDIS_WORKER_BLOCK_MS", 5000),
            batch_size=_positive_int("REDIS_WORKER_BATCH_SIZE", 1),
            claim_idle_ms=_positive_int("REDIS_WORKER_CLAIM_IDLE_MS", 60000),
            stream_max_length=_positive_int("REDIS_STREAM_MAX_LENGTH", 100000),
            lock_ttl_seconds=_positive_int("REDIS_JOB_LOCK_TTL_SECONDS", 60),
            done_ttl_seconds=_positive_int("REDIS_JOB_DONE_TTL_SECONDS", 604800),
        )

        storage_driver = _env("OBJECT_STORAGE_DRIVER", "filesystem").lower()
        if storage_driver not in {"filesystem", "s3"}:
            raise ValueError("OBJECT_STORAGE_DRIVER must be filesystem or s3")
        storage_config = StorageConfig(
            driver=storage_driver,
            endpoint=_env("OBJECT_STORAGE_ENDPOINT"),
            bucket=_env("OBJECT_STORAGE_BUCKET", "training-artifacts"),
            region=_env("OBJECT_STORAGE_REGION", "us-east-1"),
            access_key=_env("OBJECT_STORAGE_ACCESS_KEY"),
            secret_key=_env("OBJECT_STORAGE_SECRET_KEY"),
            session_token=_env("OBJECT_STORAGE_SESSION_TOKEN"),
            use_ssl=_bool("OBJECT_STORAGE_USE_SSL", False),
            filesystem_root=Path(
                _env("OBJECT_STORAGE_FILESYSTEM_ROOT", str(project_root / "object-storage"))
            ).resolve(),
        )

        gpu_raw = _env("WORKER_GPU_ID")
        gpu_id = int(gpu_raw) if gpu_raw else (0 if resource_class == "gpu" else None)
        if gpu_id is not None and gpu_id < 0:
            raise ValueError("WORKER_GPU_ID must be non-negative")

        return cls(
            worker_id=worker_id,
            resource_class=resource_class,
            supported_runtime_ids=_csv("WORKER_SUPPORTED_RUNTIME_IDS"),
            supported_models=_csv("WORKER_SUPPORTED_MODELS"),
            project_root=project_root,
            job_root=job_root,
            python_executable=_env("WORKER_PYTHON_EXECUTABLE", "python"),
            gpu_id=gpu_id,
            keep_job_dir=_bool("WORKER_KEEP_JOB_DIR", False),
            heartbeat_interval_seconds=_positive_int("WORKER_HEARTBEAT_INTERVAL_SECONDS", 15),
            shutdown_grace_seconds=_positive_int("WORKER_SHUTDOWN_GRACE_SECONDS", 15),
            redis=redis_config,
            storage=storage_config,
        )
