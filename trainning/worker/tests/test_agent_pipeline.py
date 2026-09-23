from __future__ import annotations

import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from worker.agent_pipeline import create_agent_pipeline
from worker.errors import PermanentJobError
from worker.storage import FilesystemObjectStorage
from worker.tests.helpers import execution_spec, worker_config


REPOSITORY = Path(__file__).resolve().parents[3]


class DistributedAgentPipelineTests(unittest.TestCase):
    def test_factory_loads_the_same_parameter_adapter_as_the_agent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = replace(worker_config(root), project_root=REPOSITORY, python_executable=sys.executable)
            pipeline = create_agent_pipeline(config, FilesystemObjectStorage(root / "objects"))
            self.assertEqual(type(pipeline).__name__, "DistributedAgentPipeline")
            self.assertTrue(hasattr(pipeline.process_runner, "plan"))

    def test_process_runner_injects_a_worker_owned_plan_without_local_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = replace(worker_config(root), project_root=REPOSITORY, python_executable=sys.executable)
            pipeline = create_agent_pipeline(config, FilesystemObjectStorage(root / "objects"))
            pipeline.process_runner.plan = {"modelId": "lda", "textColumn": "text", "device": "cpu", "params": {"num_topics": 3}}
            log = root / "job" / "worker.log"
            result = pipeline.process_runner.run(
                command=[sys.executable, str(REPOSITORY / "src/models/artifact_utils.py")],
                cwd=REPOSITORY / "src/models",
                env=dict(os.environ),
                log_path=log,
                timeout_seconds=10,
                is_cancelled=lambda: False,
                heartbeat=lambda: None,
                shutdown_grace_seconds=1,
            )
            self.assertEqual(result.return_code, 0)
            plan = log.parent / "agent-execution-plan.json"
            self.assertTrue(plan.is_file())
            self.assertEqual(plan.stat().st_mode & 0o777, 0o600)

    def test_remote_task_cannot_inject_a_host_model_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = replace(worker_config(root), project_root=REPOSITORY, python_executable=sys.executable)
            pipeline = create_agent_pipeline(config, FilesystemObjectStorage(root / "objects"))
            spec = execution_spec("theta")
            spec = replace(spec, params={**spec.params, "embedding.model_path": "/tmp/untrusted"})
            with self.assertRaisesRegex(PermanentJobError, "host-managed"):
                pipeline.execute(spec, lambda: False, lambda *_: None)


if __name__ == "__main__":
    unittest.main()
