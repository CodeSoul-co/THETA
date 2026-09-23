import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from workers.diagnostics import failure_diagnostics
from workers.local_compute import database, status


class DiagnosticsTests(unittest.TestCase):
    def test_existing_failed_job_returns_real_argument_error_without_reexecution(self):
        with tempfile.TemporaryDirectory() as home:
            job = 'job-' + 'a' * 64
            log = Path(home) / 'compute' / job / 'jobs/task-1-attempt-1/worker.log'
            log.parent.mkdir(parents=True)
            log.write_text('unrelated output\n' * 4000 +
                           "COMMAND: 'python' 'prepare_data.py' '--language' 'zh'\n"
                           "prepare_data.py: error: argument --language: invalid choice: 'zh' (choose from 'english', 'chinese')\n")
            value = {'id': job, 'status': 'failed', 'phase': 'failed', 'percent': 20,
                     'error': 'training command exited with code 2; see worker.log'}
            with database(home) as db:
                db.execute('INSERT INTO jobs(id,request,state,value,updated) VALUES (?,?,?,?,?)',
                           (job, '{}', 'failed', json.dumps(value), 1))
            with patch('workers.local_compute.subprocess.Popen') as spawn:
                result = status({'home': home, 'jobId': job})
                spawn.assert_not_called()
            self.assertEqual(result['diagnostics']['stage'], 'preparing')
            self.assertEqual(result['diagnostics']['category'], 'invalid_parameter')
            self.assertIn("invalid choice: 'zh'", result['diagnostics']['message'])
            self.assertLess(len(json.dumps(result['diagnostics'])), 2000)
            with database(home) as db:
                self.assertEqual(json.loads(db.execute('SELECT value FROM jobs WHERE id=?', (job,)).fetchone()[0]), value)

    def test_secrets_are_redacted_and_symlinked_logs_cannot_escape_job(self):
        with tempfile.TemporaryDirectory() as home:
            job = 'job-' + 'b' * 64
            log = Path(home) / 'compute' / job / 'host.log'
            log.parent.mkdir(parents=True)
            log.write_text('ValueError: fixture-secret Bearer hidden-token https://api.fixture/?key=hidden\n')
            with patch.dict(os.environ, {'FIXTURE_API_KEY': 'fixture-secret'}):
                result = json.dumps(failure_diagnostics(home, job))
                for secret in ['fixture-secret', 'hidden-token', 'key=hidden']:
                    self.assertNotIn(secret, result)
            log.unlink()
            outside = Path(home) / 'outside.log'
            outside.write_text('ValueError: private-other-job\n')
            log.symlink_to(outside)
            self.assertFalse(failure_diagnostics(home, job)['available'])
            self.assertFalse(failure_diagnostics(home, '../outside')['available'])

    def test_export_traceback_distinguishes_fitting_from_result_export(self):
        with tempfile.TemporaryDirectory() as home:
            job = 'job-' + 'd' * 64
            log = Path(home) / 'compute' / job / 'jobs/task-1-attempt-1/worker.log'
            log.parent.mkdir(parents=True)
            log.write_text("COMMAND: 'python' 'run_pipeline.py'\n"
                'Traceback (most recent call last):\n'
                '  File "/private/source/run_pipeline.py", line 720, in run_baseline\n'
                '    export_baseline_metadata()\n'
                '  File "/private/source/artifact_utils.py", line 283, in export_baseline_metadata\n'
                "KeyError: ''\n")
            result = failure_diagnostics(home, job)
            self.assertEqual(result['processStage'], 'training')
            self.assertEqual(result['stage'], 'exporting_results')
            self.assertEqual(result['category'], 'result_export_error')
            self.assertEqual(result['location'], {'file': 'artifact_utils.py', 'line': 283, 'function': 'export_baseline_metadata'})
            self.assertNotIn('/private/source', json.dumps(result))

    def test_missing_logs_do_not_invent_a_cause(self):
        with tempfile.TemporaryDirectory() as home:
            result = failure_diagnostics(home, 'job-' + 'c' * 64)
            self.assertFalse(result['available'])
            self.assertNotIn('message', result)
