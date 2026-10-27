"""Windows routing must return a worker response instead of crashing Python."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(os.name == 'nt', 'Windows interpreter routing only')
class WindowsRuntimeRoutingTests(unittest.TestCase):
    def test_reports_route_with_ambient_pythonpath_returns_json(self):
        agent = Path(__file__).resolve().parents[2]
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith('THETA_WORKER_')}
        environment.update({
            'PYTHONPATH': str(agent),
            'PYTHONNOUSERSITE': '1',
            'PYTHONUTF8': '1',
            'THETA_WORKER_CONTROL_PYTHON': sys.executable,
            'THETA_WORKER_REPORTS_PYTHON': sys.executable,
        })
        with tempfile.TemporaryDirectory() as home:
            result = subprocess.run(
                [sys.executable, '-m', 'workers', 'compute.results'],
                input=json.dumps({'home': home, 'jobId': 'missing', 'view': 'report'}),
                cwd=agent, env=environment, capture_output=True, text=True,
                timeout=30,
            )
        self.assertEqual(result.returncode, 1, result.stderr)
        response = json.loads(result.stdout)
        self.assertFalse(response['ok'])
        self.assertIn('任务不存在', response['error'])


if __name__ == '__main__':
    unittest.main()
