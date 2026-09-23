from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
import venv
from unittest.mock import patch

from workers import runtime_environments as runtimes
from workers.capabilities import MODELS, dataset_import
from workers.execution_policy import assert_authorized, execution_policy


class RuntimeEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.python = cls.root / 'clean environment' / 'bin/python'
        venv.EnvBuilder(with_pip=False).create(cls.python.parent.parent)
        cls.agent = Path(__file__).resolve().parents[2]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def environment(self, **changes):
        base = {k: v for k, v in os.environ.items() if not k.startswith('THETA_WORKER_')}
        return {**base, 'THETA_WORKER_CONTROL_PYTHON': sys.executable,
                'THETA_WORKER_CLASSIC_PYTHON': str(self.python),
                'THETA_WORKER_NEURAL_PYTHON': str(self.python),
                'THETA_WORKER_STATISTICS_PYTHON': str(self.python), 'THETA_WORKER_REPORTS_PYTHON': str(self.python), **changes}

    def call(self, operation, payload, **environment):
        result = subprocess.run([sys.executable, '-m', 'workers', operation], input=json.dumps(payload),
                                env=self.environment(**environment), cwd=self.agent,
                                capture_output=True, text=True, timeout=20)
        return json.loads(result.stdout)

    def test_all_models_registered_and_real_clean_venv_used(self):
        self.assertEqual(set(MODELS), {model for models in runtimes.PROFILES.values() for model in models})
        for profile in runtimes.PROFILES:
            response = self.call('runtime.environment', {'profile': profile})
            self.assertTrue(response['ok'], response)
            info = response['data']
            self.assertEqual(info['python'], str(self.python))
            self.assertEqual(info['profile'], profile)
            self.assertTrue(info['isolatedVenv'])
            self.assertFalse(info['inheritsSystemPackages'])
        ready = self.call('runtime.check', {'modelId': 'lda'})['data']
        self.assertFalse(ready['ready'])
        self.assertIn('numpy', ready['missingDependencies'])
        self.assertEqual(ready['runtime']['python'], str(self.python))
        # Neither ambient PYTHONPATH nor model input may inject another environment.
        bad = self.root / 'ambient'; bad.mkdir(exist_ok=True)
        (bad / 'numpy.py').write_text('raise RuntimeError("must not import")')
        ready = self.call('runtime.check', {'modelId': 'lda', 'python': sys.executable}, PYTHONPATH=str(bad))['data']
        self.assertIn('numpy', ready['missingDependencies'])

    def test_preview_and_readiness_share_runtime_without_training(self):
        source = self.root / 'texts.csv'
        source.write_text('text\nexample one\nexample two\n')
        dataset = dataset_import({'filePath': str(source), 'uploadDir': str(self.root / 'uploads')})
        plan = {'modelId': 'lda', 'textColumn': 'text', 'params': {'num_topics': 2}, 'timeoutSeconds': 60}
        response = self.call('compute.preview', {'plan': plan, 'dataset': dataset})
        self.assertTrue(response['ok'], response)
        self.assertEqual(response['data']['execution']['runtime'], response['data']['readiness']['runtime'])
        self.assertFalse(response['data']['readiness']['ready'])
        # A synthetic approval reaches the same selected runtime, then stops at
        # missing dependencies. No training is started by this regression check.
        with tempfile.TemporaryDirectory() as home:
            request = {'jobId': 'job-' + 'a' * 64, 'runId': 'fixture', 'plan': plan,
                       'dataset': dataset, 'execution': response['data']['execution']}
            approval = {'status': 'approved', 'hash': 'fixture', 'action': 'compute.submit',
                        'target': 'local', 'payload': request, 'expiresAt': time.time() * 1000 + 60000}
            with closing(sqlite3.connect(Path(home) / 'research.sqlite')) as db, db:
                db.execute('CREATE TABLE records(kind TEXT, id TEXT, value TEXT)')
                db.execute('INSERT INTO records VALUES (?,?,?)', ('effect-approval', 'fixture', json.dumps(approval)))
            submitted = self.call('compute.submit', {**request, 'home': home,
                                  'authorization': {'id': 'fixture', 'hash': 'fixture'}})
            self.assertFalse(submitted['ok'])
            self.assertIn('训练依赖或本地模型资产未就绪', submitted['error'])
            with closing(sqlite3.connect(Path(home) / 'compute.sqlite')) as db:
                self.assertEqual(db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0], 0)

    def test_broken_training_or_report_environment_does_not_break_control(self):
        missing = str(self.root / 'missing-python')
        error = self.call('runtime.check', {'modelId': 'lda'}, THETA_WORKER_CLASSIC_PYTHON=missing)
        self.assertFalse(error['ok'])
        self.assertIn('不会回退', error['error'])
        error = self.call('compute.results', {'view': 'report'}, THETA_WORKER_REPORTS_PYTHON=missing)
        self.assertIn('reports', error['error'])
        error = self.call('figure.adjust', {}, THETA_WORKER_REPORTS_PYTHON=missing)
        self.assertIn('reports', error['error'])
        error = self.call('compute.results', {'view': 'artifacts', 'home': str(self.root), 'jobId': 'missing'}, THETA_WORKER_REPORTS_PYTHON=missing)
        self.assertNotIn('Python', error['error'])
        for operation in ['compute.status', 'compute.cancel']:
            error = self.call(operation, {'home': str(self.root), 'jobId': 'missing'}, THETA_WORKER_CLASSIC_PYTHON=missing)
            self.assertIn('任务不存在', error['error'])
        self.assertTrue(self.call('models.list', {}, THETA_WORKER_CLASSIC_PYTHON=missing)['ok'])

    def test_environment_revision_change_invalidates_approved_execution(self):
        with patch.dict(os.environ, {'THETA_WORKER_CONTROL_PYTHON': sys.executable}, clear=True), tempfile.TemporaryDirectory() as home:
            plan = {'modelId': 'lda', 'params': {}, 'timeoutSeconds': 60}
            payload = {'plan': plan, 'execution': execution_policy(plan)}
            approval = {'status': 'approved', 'hash': 'fixture', 'action': 'compute.submit',
                        'target': 'local', 'payload': payload, 'expiresAt': time.time() * 1000 + 60000}
            with closing(sqlite3.connect(Path(home) / 'research.sqlite')) as db, db:
                db.execute('CREATE TABLE records(kind TEXT, id TEXT, value TEXT)')
                db.execute('INSERT INTO records VALUES (?,?,?)', ('effect-approval', 'fixture', json.dumps(approval)))
            request = {**payload, 'home': home, 'authorization': {'id': 'fixture', 'hash': 'fixture'}}
            assert_authorized(request)
            with patch.dict(os.environ, {'THETA_WORKER_CLASSIC_REVISION': 'v2'}):
                with self.assertRaisesRegex(ValueError, '运行环境'):
                    assert_authorized(request)
