from __future__ import annotations

import logging
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from typing import Any

from .artifacts import ArtifactPublisher
from .config import WorkerConfig
from .errors import (
    JobCancelled,
    LeaseLost,
    PermanentJobError,
    RetryableJobError,
    WorkerStopping,
)
from .pipeline import JobPaths, ThetaPipeline
from .protocol import ExecutionSpec, worker_event
from .redis_queue import JobLease, QueueMessage, RedisQueue


LOGGER = logging.getLogger(__name__)
MAX_ERROR_MESSAGE = 4000


@dataclass(frozen=True)
class TerminalOutcome:
    event_type: str
    fields: dict[str, Any]


class LeaseKeeper:
    def __init__(self, queue: RedisQueue, lease: JobLease, ttl_seconds: int):
        self.queue = queue
        self.lease = lease
        self.interval = max(1.0, ttl_seconds / 3)
        self.stop_event = threading.Event()
        self.lost_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=self.interval + 1)

    def is_lost(self) -> bool:
        return self.lost_event.is_set()

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval):
            try:
                if not self.queue.renew_lease(self.lease):
                    self.lost_event.set()
                    return
            except Exception as exc:
                # A brief Redis outage does not prove the lease is gone. Keep trying;
                # a later zero result means another Worker owns it.
                LOGGER.warning("could not renew lease %s: %s", self.lease.key, exc)


class WorkerService:
    def __init__(
        self,
        config: WorkerConfig,
        queue: RedisQueue,
        pipeline: ThetaPipeline,
        artifacts: ArtifactPublisher,
    ):
        self.config = config
        self.queue = queue
        self.pipeline = pipeline
        self.artifacts = artifacts
        self.stop_event = threading.Event()

    def stop(self) -> None:
        LOGGER.info("worker shutdown requested")
        self.stop_event.set()

    def run(self) -> None:
        self.queue.ping()
        self.queue.ensure_group()
        self.pipeline.validate_installation()
        LOGGER.info(
            "worker started: id=%s resource=%s stream=%s group=%s",
            self.config.worker_id,
            self.config.resource_class,
            self.config.redis.job_stream,
            self.config.redis.consumer_group,
        )

        reclaimed = self.queue.reclaim_pending()
        for message in reclaimed:
            if self.stop_event.is_set():
                break
            self._process_safely(message)

        last_reclaim = time.monotonic()
        while not self.stop_event.is_set():
            try:
                messages = self.queue.read()
                for message in messages:
                    if self.stop_event.is_set():
                        break
                    self._process_safely(message)
                if time.monotonic() - last_reclaim >= max(
                    30, self.config.redis.claim_idle_ms / 1000
                ):
                    for message in self.queue.reclaim_pending():
                        if self.stop_event.is_set():
                            break
                        self._process_safely(message)
                    last_reclaim = time.monotonic()
            except RetryableJobError as exc:
                LOGGER.warning("Redis temporarily unavailable: %s", exc)
                self.stop_event.wait(2)
            except Exception:
                LOGGER.exception("unexpected worker loop failure")
                self.stop_event.wait(2)
        LOGGER.info("worker stopped")

    def _process_safely(self, message: QueueMessage) -> None:
        try:
            self.process_message(message)
        except WorkerStopping as exc:
            LOGGER.info("job message %s left pending during shutdown: %s", message.message_id, exc)
        except LeaseLost as exc:
            LOGGER.warning("job message %s left pending after lease loss: %s", message.message_id, exc)
        except Exception:
            # Do not ACK on an infrastructure failure. The pending job can be reclaimed.
            LOGGER.exception("job message %s was left pending", message.message_id)

    def process_message(self, message: QueueMessage) -> None:
        try:
            payload = self.queue.decode_payload(message)
            spec = ExecutionSpec.from_dict(payload)
        except Exception as exc:
            LOGGER.error("dead-lettering invalid job %s: %s", message.message_id, exc)
            self.queue.dead_letter(message, str(exc))
            self.queue.acknowledge(message)
            return

        if self.queue.is_done(spec):
            LOGGER.info("job %s is already terminal; acknowledging duplicate", spec.identity)
            self.queue.acknowledge(message)
            return

        lease = self.queue.acquire_lease(spec)
        if lease is None:
            LOGGER.info("job %s is leased by another worker", spec.identity)
            return

        paths: JobPaths | None = None
        terminal_committed = False
        keeper = LeaseKeeper(self.queue, lease, self.config.redis.lock_ttl_seconds)
        keeper.start()
        try:
            outcome = self._run_job(spec, lease, keeper)
            # The pipeline creates a deterministic path even when it later fails.
            possible_root = (
                self.config.job_root.resolve()
                / f"task-{spec.task_id}-attempt-{spec.attempt}"
            )
            if possible_root.exists():
                paths = JobPaths(
                    root=possible_root,
                    data_root=possible_root / "data",
                    workspace=possible_root / "workspace",
                    result_root=possible_root / "result",
                    source_file=possible_root / "data" / f"dataset_{spec.task_id}" / spec.dataset.filename,
                    dataset_file=possible_root / "data" / f"dataset_{spec.task_id}" / "data.csv",
                    log_file=possible_root / "worker.log",
                )
            event = worker_event(
                outcome.event_type,
                spec,
                self.config.worker_id,
                self._terminal_event_id(spec, outcome.event_type),
                **outcome.fields,
            )
            self.queue.publish_event(event)
            self.queue.mark_done(spec, outcome.event_type)
            self.queue.acknowledge(message)
            terminal_committed = True
            LOGGER.info("job %s ended with %s", spec.identity, outcome.event_type)
        finally:
            keeper.stop()
            self.queue.release_lease(lease)
            if terminal_committed and paths is not None and not self.config.keep_job_dir:
                paths.cleanup()

    def _run_job(
        self, spec: ExecutionSpec, lease: JobLease, keeper: LeaseKeeper
    ) -> TerminalOutcome:
        try:
            self._validate_capability(spec)
            if self.queue.is_cancelled(spec):
                raise JobCancelled("task was cancelled before execution")
            self.queue.publish_event(
                worker_event(
                    "job.started",
                    spec,
                    self.config.worker_id,
                    uuid.uuid4().hex,
                    phase="downloading",
                    progress=1,
                    message="worker accepted task",
                )
            )

            def cancelled() -> bool:
                return (
                    self.stop_event.is_set()
                    or keeper.is_lost()
                    or self.queue.is_cancelled(spec)
                )

            def progress(phase: str, percentage: int, message: str) -> None:
                if keeper.is_lost():
                    raise LeaseLost("job lease was lost")
                if not self.queue.renew_lease(lease):
                    raise LeaseLost("job lease was lost")
                self.queue.publish_event(
                    worker_event(
                        "job.progress",
                        spec,
                        self.config.worker_id,
                        uuid.uuid4().hex,
                        phase=phase,
                        progress=percentage,
                        message=message,
                    )
                )

            pipeline_result = self.pipeline.execute(spec, cancelled, progress)
            progress("uploading", 92, "uploading result bundle")
            bundle = self.artifacts.publish(spec, pipeline_result)
            if keeper.is_lost():
                raise LeaseLost("job lease was lost while publishing results")
            if self.stop_event.is_set():
                raise WorkerStopping("worker stopped while publishing results")
            if self.queue.is_cancelled(spec):
                raise JobCancelled("task was cancelled while publishing results")
            return TerminalOutcome(
                "job.completed",
                {
                    "phase": "completed",
                    "progress": 100,
                    "message": "训练已完成",
                    "model_weights_path": bundle.model_weights_path,
                    "metrics": bundle.metrics,
                    "output_files": bundle.output_files,
                },
            )
        except JobCancelled as exc:
            if keeper.is_lost():
                raise LeaseLost("job lease was lost") from exc
            if self.stop_event.is_set() and not self.queue.is_cancelled(spec):
                raise WorkerStopping("worker stopped during task execution") from exc
            return TerminalOutcome(
                "job.cancelled",
                {
                    "phase": "cancelled",
                    "message": str(exc),
                },
            )
        except (LeaseLost, WorkerStopping):
            raise
        except PermanentJobError as exc:
            return self._failure("INVALID_JOB", str(exc), False)
        except RetryableJobError as exc:
            return self._failure("EXECUTION_FAILED", str(exc), True)
        except Exception as exc:
            LOGGER.error("unhandled task exception:\n%s", traceback.format_exc())
            return self._failure("WORKER_ERROR", str(exc), True)

    def _validate_capability(self, spec: ExecutionSpec) -> None:
        if spec.resources.queue_class != self.config.resource_class:
            raise PermanentJobError(
                f"task requires {spec.resources.queue_class}, worker is {self.config.resource_class}"
            )
        if (
            self.config.supported_runtime_ids
            and spec.runtime.key not in self.config.supported_runtime_ids
            and str(spec.runtime.id) not in self.config.supported_runtime_ids
        ):
            raise PermanentJobError(f"runtime is not supported by this worker: {spec.runtime.key}")
        if self.config.supported_models and spec.model.name not in self.config.supported_models:
            raise PermanentJobError(f"model is not supported by this worker: {spec.model.name}")

    @staticmethod
    def _failure(code: str, message: str, retryable: bool) -> TerminalOutcome:
        return TerminalOutcome(
            "job.failed",
            {
                "phase": "failed",
                "message": "training failed",
                "error_code": code,
                "error_message": message[:MAX_ERROR_MESSAGE] or code,
                "retryable": retryable,
            },
        )

    @staticmethod
    def _terminal_event_id(spec: ExecutionSpec, event_type: str) -> str:
        suffix = event_type.removeprefix("job.")
        return f"task-{spec.task_id}-attempt-{spec.attempt}-{suffix}"
