from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from .errors import PermanentJobError


SCHEMA_VERSION = 2


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PermanentJobError(f"{field} must be an object")
    return value


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PermanentJobError(f"{field} must be a positive integer")
    return value


def _required_string(value: Any, field: str, max_length: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PermanentJobError(f"{field} is required")
    normalized = value.strip()
    if len(normalized) > max_length:
        raise PermanentJobError(f"{field} must not exceed {max_length} characters")
    return normalized


def validate_object_key(value: str, field: str, allow_trailing_slash: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PermanentJobError(f"{field} is required")
    normalized = value.strip().replace("\\", "/").lstrip("/")
    components = normalized.split("/")
    for index, part in enumerate(components):
        if allow_trailing_slash and index == len(components) - 1 and part == "":
            continue
        if part in {"", ".", ".."}:
            raise PermanentJobError(f"{field} contains an unsafe path component")
    if not allow_trailing_slash and normalized.endswith("/"):
        raise PermanentJobError(f"{field} must identify an object")
    return normalized


@dataclass(frozen=True)
class DatasetSpec:
    ref: str
    project_id: str
    object_key: str
    filename: str
    format: str
    sha256: str


@dataclass(frozen=True)
class ModelSpec:
    id: int
    name: str
    framework: str


@dataclass(frozen=True)
class RuntimeSpec:
    id: int
    key: str
    version: str
    executor: str
    image: str


@dataclass(frozen=True)
class ResourceRequest:
    accelerator: str
    cpu_cores: int
    memory_mb: int
    gpu_count: int
    gpu_memory_mb: int
    timeout_seconds: int

    @property
    def queue_class(self) -> str:
        return "gpu" if self.gpu_count > 0 else "cpu"


@dataclass(frozen=True)
class ExecutionSpec:
    event_id: str
    task_id: int
    attempt: int
    task_version: int
    user_id: str
    priority: int
    dataset: DatasetSpec
    model: ModelSpec
    runtime: RuntimeSpec
    params: dict[str, Any]
    resources: ResourceRequest
    output_prefix: str
    created_at: str

    @property
    def identity(self) -> str:
        return f"{self.task_id}:{self.attempt}"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExecutionSpec":
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise PermanentJobError("unsupported schema_version")
        if payload.get("type") != "job.ready":
            raise PermanentJobError("message type must be job.ready")

        dataset_raw = _mapping(payload.get("dataset"), "dataset")
        model_raw = _mapping(payload.get("model"), "model")
        runtime_raw = _mapping(payload.get("runtime"), "runtime")
        resources_raw = _mapping(payload.get("resources"), "resources")
        params_raw = _mapping(payload.get("params", {}), "params")

        filename = dataset_raw.get("filename")
        if not isinstance(filename, str) or not filename or filename in {".", ".."}:
            raise PermanentJobError("dataset.filename is required")
        if "/" in filename or "\\" in filename:
            raise PermanentJobError("dataset.filename must not contain a path")

        model_name = model_raw.get("name")
        if not isinstance(model_name, str) or not model_name.strip():
            raise PermanentJobError("model.name is required")
        runtime_key = runtime_raw.get("key")
        executor = runtime_raw.get("executor")
        if not isinstance(runtime_key, str) or not runtime_key:
            raise PermanentJobError("runtime.key is required")
        if executor != "theta_pipeline":
            raise PermanentJobError(f"unsupported executor {executor!r}")

        event_id = payload.get("event_id")
        if not isinstance(event_id, str) or not event_id.strip():
            raise PermanentJobError("event_id is required")

        output_prefix = validate_object_key(
            str(payload.get("output_prefix", "")), "output_prefix", allow_trailing_slash=True
        )
        if not output_prefix.endswith("/"):
            output_prefix += "/"

        try:
            priority = int(payload.get("priority", 0))
            cpu_cores = int(resources_raw.get("cpu_cores", 1))
            memory_mb = int(resources_raw.get("memory_mb", 1))
            gpu_count = int(resources_raw.get("gpu_count", 0))
            gpu_memory_mb = int(resources_raw.get("gpu_memory_mb", 0))
        except (TypeError, ValueError) as exc:
            raise PermanentJobError("priority and resource values must be integers") from exc
        if not 0 <= priority <= 255:
            raise PermanentJobError("priority must be between 0 and 255")
        if cpu_cores <= 0 or memory_mb <= 0:
            raise PermanentJobError("resources.cpu_cores and resources.memory_mb must be positive")
        if gpu_count < 0:
            raise PermanentJobError("resources.gpu_count must be non-negative")
        if gpu_memory_mb < 0:
            raise PermanentJobError("resources.gpu_memory_mb must be non-negative")

        return cls(
            event_id=event_id.strip(),
            task_id=_positive_int(payload.get("task_id"), "task_id"),
            attempt=_positive_int(payload.get("attempt"), "attempt"),
            task_version=_positive_int(payload.get("task_version"), "task_version"),
            user_id=_required_string(payload.get("user_id"), "user_id", 36),
            priority=priority,
            dataset=DatasetSpec(
                ref=_required_string(dataset_raw.get("ref"), "dataset.ref", 191),
                project_id=_required_string(dataset_raw.get("project_id"), "dataset.project_id", 36),
                object_key=validate_object_key(str(dataset_raw.get("object_key", "")), "dataset.object_key"),
                filename=filename,
                format=str(dataset_raw.get("format", "")),
                sha256=str(dataset_raw.get("sha256", "")),
            ),
            model=ModelSpec(
                id=_positive_int(model_raw.get("id"), "model.id"),
                name=model_name.strip().lower(),
                framework=str(model_raw.get("framework", "")),
            ),
            runtime=RuntimeSpec(
                id=_positive_int(runtime_raw.get("id"), "runtime.id"),
                key=runtime_key,
                version=str(runtime_raw.get("version", "")),
                executor=executor,
                image=str(runtime_raw.get("image", "")),
            ),
            params=dict(params_raw),
            resources=ResourceRequest(
                accelerator=str(resources_raw.get("accelerator", "none")),
                cpu_cores=cpu_cores,
                memory_mb=memory_mb,
                gpu_count=gpu_count,
                gpu_memory_mb=gpu_memory_mb,
                timeout_seconds=_positive_int(resources_raw.get("timeout_seconds"), "resources.timeout_seconds"),
            ),
            output_prefix=output_prefix,
            created_at=str(payload.get("created_at", "")),
        )


def worker_event(
    event_type: str,
    spec: ExecutionSpec,
    worker_id: str,
    event_id: str,
    **fields: Any,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "event_id": event_id,
        "type": event_type,
        "task_id": spec.task_id,
        "attempt": spec.attempt,
        "worker_id": worker_id,
        "occurred_at": utc_now(),
    }
    event.update({name: value for name, value in fields.items() if value is not None})
    return event
