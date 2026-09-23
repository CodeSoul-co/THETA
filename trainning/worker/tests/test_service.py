from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from worker.artifacts import ArtifactBundle
from worker.errors import LeaseLost
from worker.pipeline import JobPaths, PipelineResult
from worker.redis_queue import JobLease, QueueMessage
from worker.service import WorkerService
from worker.tests.helpers import execution_payload, worker_config


class FakeQueue:
    def __init__(self, payload: dict, cancelled: bool = False, renews: bool = True):
        self.payload = payload
        self.cancelled = cancelled
        self.renews = renews
        self.events: list[dict] = []
        self.acked = False
        self.done = ""

    def decode_payload(self, _message: QueueMessage) -> dict:
        return self.payload

    def is_done(self, _spec: object) -> bool:
        return False

    def acquire_lease(self, _spec: object) -> JobLease:
        return JobLease("lock", "token")

    def renew_lease(self, _lease: JobLease) -> bool:
        return self.renews

    def release_lease(self, _lease: JobLease) -> None:
        pass

    def is_cancelled(self, _spec: object) -> bool:
        return self.cancelled

    def publish_event(self, event: dict) -> str:
        self.events.append(event)
        return "1-0"

    def mark_done(self, _spec: object, terminal_type: str) -> None:
        self.done = terminal_type

    def acknowledge(self, _message: QueueMessage) -> None:
        self.acked = True


class FakePipeline:
    def __init__(self, root: Path):
        self.root = root

    def execute(self, spec: object, _cancelled: object, progress: object) -> PipelineResult:
        paths = JobPaths.create(self.root / "jobs", spec)
        result = paths.result_root / "model"
        result.mkdir(parents=True)
        (result / "weights.npy").write_bytes(b"weights")
        progress("training", 50, "training")
        return PipelineResult(result, paths, 1.0)


class FakeArtifacts:
    def publish(self, spec: object, _result: PipelineResult) -> ArtifactBundle:
        return ArtifactBundle(
            model_weights_path=spec.output_prefix + "model.tar.gz",
            metrics={"npmi": 0.5},
            output_files={"manifest": spec.output_prefix + "manifest.json"},
        )


class WorkerServiceTests(unittest.TestCase):
    def test_successful_job_publishes_lifecycle_and_acknowledges(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = worker_config(root)
            queue = FakeQueue(execution_payload())
            service = WorkerService(config, queue, FakePipeline(root), FakeArtifacts())
            service.process_message(QueueMessage("1-0", {"payload": "unused"}))
            event_types = [event["type"] for event in queue.events]
            self.assertEqual(event_types[0], "job.started")
            self.assertIn("job.progress", event_types)
            self.assertEqual(event_types[-1], "job.completed")
            self.assertEqual(queue.done, "job.completed")
            self.assertTrue(queue.acked)

    def test_cancelled_queued_job_does_not_start_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = worker_config(root)
            queue = FakeQueue(execution_payload(), cancelled=True)
            service = WorkerService(config, queue, FakePipeline(root), FakeArtifacts())
            service.process_message(QueueMessage("1-0", {"payload": "unused"}))
            self.assertEqual([event["type"] for event in queue.events], ["job.cancelled"])
            self.assertTrue(queue.acked)

    def test_lost_lease_leaves_message_pending_without_terminal_event(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = worker_config(root)
            queue = FakeQueue(execution_payload(), renews=False)
            service = WorkerService(config, queue, FakePipeline(root), FakeArtifacts())
            with self.assertRaises(LeaseLost):
                service.process_message(QueueMessage("1-0", {"payload": "unused"}))
            self.assertEqual([event["type"] for event in queue.events], ["job.started"])
            self.assertFalse(queue.acked)
            self.assertEqual(queue.done, "")


if __name__ == "__main__":
    unittest.main()
