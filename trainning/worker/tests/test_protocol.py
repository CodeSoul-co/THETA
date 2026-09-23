from __future__ import annotations

import unittest

from worker.errors import PermanentJobError
from worker.protocol import ExecutionSpec, validate_object_key, worker_event
from worker.tests.helpers import execution_payload, execution_spec


class ProtocolTests(unittest.TestCase):
    def test_parses_execution_spec(self) -> None:
        spec = execution_spec()
        self.assertEqual(spec.identity, "42:1")
        self.assertEqual(spec.resources.queue_class, "cpu")
        self.assertEqual(spec.output_prefix, "training-results/42/attempt-1/")

    def test_rejects_unsafe_object_keys(self) -> None:
        for key in ("../secret", "a//b", "a/./b", "a/../b"):
            with self.subTest(key=key), self.assertRaises(PermanentJobError):
                validate_object_key(key, "key")

    def test_allows_one_trailing_slash_only(self) -> None:
        self.assertEqual(validate_object_key("a/b/", "key", True), "a/b/")
        with self.assertRaises(PermanentJobError):
            validate_object_key("a//", "key", True)

    def test_rejects_missing_event_and_invalid_resources(self) -> None:
        payload = execution_payload()
        payload["event_id"] = ""
        with self.assertRaises(PermanentJobError):
            ExecutionSpec.from_dict(payload)
        payload = execution_payload()
        payload["resources"]["memory_mb"] = 0
        with self.assertRaises(PermanentJobError):
            ExecutionSpec.from_dict(payload)

    def test_worker_event_contains_contract_fields(self) -> None:
        event = worker_event("job.started", execution_spec(), "worker-1", "event-1")
        self.assertEqual(event["schema_version"], 2)
        self.assertEqual(event["task_id"], 42)
        self.assertTrue(event["occurred_at"].endswith("Z"))


if __name__ == "__main__":
    unittest.main()
