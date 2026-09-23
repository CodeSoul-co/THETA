from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

from worker.process import ProcessRunner


class ProcessRunnerTests(unittest.TestCase):
    def test_captures_combined_process_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "worker.log"
            heartbeats: list[bool] = []
            result = ProcessRunner(poll_seconds=0.02).run(
                [sys.executable, "-c", "print('worker child ok')"],
                cwd=root,
                env=os.environ,
                log_path=log,
                timeout_seconds=5,
                is_cancelled=lambda: False,
                heartbeat=lambda: heartbeats.append(True),
                shutdown_grace_seconds=1,
            )
            self.assertEqual(result.return_code, 0)
            self.assertIn("worker child ok", log.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
