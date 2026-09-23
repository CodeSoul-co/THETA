from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from worker.config import WorkerConfig


class ConfigTests(unittest.TestCase):
    def test_blank_consumer_values_use_resource_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {
                "THETA_PROJECT_ROOT": directory,
                "WORKER_RESOURCE_CLASS": "gpu",
                "WORKER_ID": "",
                "REDIS_WORKER_CONSUMER_GROUP": "",
                "REDIS_WORKER_CONSUMER_NAME": "",
            },
            clear=True,
        ):
            config = WorkerConfig.load()
        self.assertEqual(config.redis.job_stream, "theta:jobs:gpu")
        self.assertEqual(config.redis.consumer_group, "theta-workers-gpu")
        self.assertTrue(config.worker_id.endswith("-gpu"))
        self.assertEqual(config.redis.consumer_name, config.worker_id)


if __name__ == "__main__":
    unittest.main()
