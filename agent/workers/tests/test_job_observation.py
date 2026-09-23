import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from workers.job_observation import observe
from workers.local_compute import database, update, status


class ObservationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.home = Path(self.temp.name)
        self.job = dict(id='job-' + 'a' * 64, status='running', phase='training', percent=60,
                        startedAt=800, phaseStartedAt=900)
        self.log = self.home / 'compute' / self.job['id'] / 'jobs/task-1-attempt-1/worker.log'
        self.log.parent.mkdir(parents=True)

    def tearDown(self):
        self.temp.cleanup()

    def log_text(self, text, modified=999):
        self.log.write_text(text)
        os.utime(self.log, (modified, modified))

    def test_real_iteration_progress_changes_with_constant_stage_percent(self):
        self.log_text('COMMAND: trainer\niteration: 19 of max_iter: 50\n')
        first = observe(self.home, self.job, 999, now=1000)
        self.log_text('iteration: 20 of max_iter: 50\n')
        second = observe(self.home, self.job, 1002, now=1003)
        self.assertEqual(first['iteration']['current'], 19)
        self.assertEqual(second['iteration']['current'], 20)
        self.assertEqual(second['percentKind'], 'stage_marker')
        self.assertEqual(second['elapsedSeconds'], 203)
        self.assertEqual(second['phaseElapsedSeconds'], 103)
        self.assertEqual(second['iteration']['meaning'], 'reported_iteration_not_completed_count')

    def test_quiet_logs_and_missing_heartbeat_are_distinct(self):
        self.log_text('private text is not progress\n', modified=800)
        live = observe(self.home, self.job, 999, now=1000)
        self.assertEqual(live['health'], 'responding')
        self.assertEqual(live['lastLogAgeSeconds'], 200)
        self.assertIsNone(live['iteration'])
        dead = observe(self.home, self.job, 960, now=1000)
        self.assertEqual(dead['health'], 'unresponsive')
        self.assertNotIn('private text', json.dumps(live))

    def test_visualization_and_next_command_clear_previous_iteration(self):
        for suffix in ['Running Visualizations', 'COMMAND: next']:
            self.log_text('iteration: 50 of max_iter: 50\n' + suffix)
            value = observe(self.home, self.job, 999, now=1000)
            self.assertIsNone(value['iteration'])
        self.job['phase'] = 'preparing_data'
        self.log_text('iteration: 19 of max_iter: 50\n')
        self.assertIsNone(observe(self.home, self.job, 999, now=1000)['iteration'])

    def test_legacy_job_and_symlink_do_not_invent_progress(self):
        del self.job['startedAt']; del self.job['phaseStartedAt']
        outside = self.home / 'outside.log'; outside.write_text('iteration: 19 of max_iter: 50\n')
        self.log.symlink_to(outside)
        value = observe(self.home, self.job, 999, now=1000)
        self.assertIsNone(value['lastLogAgeSeconds'])
        self.assertIsNone(value['elapsedSeconds'])
        self.assertIsNone(value['iteration'])

    def test_phase_clock_is_not_reset_by_heartbeat_and_finished_time_freezes(self):
        with database(self.home) as db:
            db.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?)', (self.job['id'], '{}', 'running', json.dumps(self.job), 0, 999))
        with patch('workers.local_compute.time.time', return_value=1000):
            update(self.home, self.job['id'], phase='training', percent=60)
            self.assertEqual(status({'home': self.home, 'jobId': self.job['id']})['telemetry']['phaseElapsedSeconds'], 100)
        with patch('workers.local_compute.time.time', return_value=1020):
            update(self.home, self.job['id'], phase='uploading')
        with patch('workers.local_compute.time.time', return_value=1040):
            update(self.home, self.job['id'], status='completed', phase='completed')
        with patch('workers.local_compute.time.time', return_value=1100):
            result = status({'home': self.home, 'jobId': self.job['id']})
        self.assertEqual(result['telemetry']['elapsedSeconds'], 240)
        self.assertEqual([item['phase'] for item in result['phaseHistory']], ['uploading', 'completed'])
        self.assertEqual(result['phaseHistory'][0]['at'], 1020)
        self.assertEqual(result['telemetry']['health'], 'finished')
