import hashlib
import json
import os
from pathlib import Path
import sqlite3
from contextlib import closing
import subprocess
import sys
import tempfile
import unittest
import urllib.request
from unittest.mock import patch

from workers.execution_policy import execution_policy, training_environment, configuration_summary
from workers.engine_entry import guarded_urlopen, NoRedirect
from workers.local_compute import submit, database


class ExecutionPolicyTests(unittest.TestCase):
    def test_entry_wrapper_preserves_script_arguments_and_blocks_unapproved_io(self):
        with tempfile.TemporaryDirectory() as home:
            root = Path(home)
            models = root / 'src/models'
            models.mkdir(parents=True)
            (models / 'fixture_helper.py').write_text('VALUE = "fixture"\n')
            script = models / 'fixture_entry.py'
            script.write_text(
                'import sys, urllib.request\nfrom fixture_helper import VALUE\n'
                'assert sys.argv[1:] == ["--fixture", "value"]\n'
                'try:\n    urllib.request.urlopen("https://never-send.invalid")\n'
                'except PermissionError:\n    print(VALUE + ":blocked")\n'
                'else:\n    raise AssertionError("unapproved request escaped")\n')
            with database(home) as db:
                db.execute('INSERT INTO jobs(id,request,state,value,updated) VALUES (?,?,?,?,?)',
                           ('job-fixture', json.dumps({'execution': {'embedding': {'mode': 'local'}}}), 'running', '{}', 0))
            wrapper = Path(__file__).resolve().parents[1] / 'engine_entry.py'
            result = subprocess.run([sys.executable, str(wrapper), str(script), '--fixture', 'value'],
                                    env={**os.environ, 'THETA_PROJECT_ROOT': home,
                                         'THETA_COMPUTE_DATABASE': str(root / 'compute.sqlite'),
                                         'THETA_COMPUTE_JOB_ID': 'job-fixture'},
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), 'fixture:blocked')

    def cloud(self):
        return {'modelId': 'theta', 'params': {'embedding_provider': 'cloud', 'mode': 'zero_shot'},
                'timeoutSeconds': 300, 'externalRequestLimit': 2}

    def settings(self):
        return patch.dict(os.environ, {'EMBEDDING_API_BASE': 'https://embedding.fixture/v1/embeddings',
            'EMBEDDING_MODEL': 'fixture-model', 'EMBEDDING_API_KEY_ENV': 'TEST_EMBEDDING_KEY',
            'TEST_EMBEDDING_KEY': 'fixture-key', 'EMBEDDING_API_KEY': ''}, clear=True)

    def test_configuration_never_enables_cloud_implicitly_and_credentials_are_scoped(self):
        with self.settings(), patch.dict(os.environ, {'EMBEDDING_PROVIDER': 'cloud', 'OPENAI_API_KEY': 'unrelated', 'DEEPSEEK_API_KEY': 'dialogue'}):
            local = execution_policy({'modelId': 'theta', 'params': {}, 'timeoutSeconds': 30})
            self.assertEqual(local['embedding']['mode'], 'local')
            environment = training_environment(local)
            self.assertEqual(environment['EMBEDDING_PROVIDER'], 'local')
            self.assertNotIn('OPENAI_API_KEY', environment)
            self.assertNotIn('DEEPSEEK_API_KEY', environment)
            cloud = execution_policy(self.cloud())
            self.assertEqual(cloud['maxExternalRequests'], 2)
            self.assertNotIn('fixture-key', json.dumps(cloud))
            public = configuration_summary({})
            self.assertFalse(public['agentInferenceRequiresConfirmation'])
            self.assertNotIn('credentialFingerprint', json.dumps(public))
            environment = training_environment(cloud)
            self.assertEqual(environment['EMBEDDING_API_KEY'], 'fixture-key')
            self.assertNotIn('DEEPSEEK_API_KEY', environment)
            self.assertEqual(environment['EMBEDDING_MAX_RETRIES'], '2')
            self.assertEqual(environment['EMBEDDING_API_BASE'], 'https://embedding.fixture/v1')
            with self.assertRaises(ValueError):
                execution_policy({**self.cloud(), 'params': {'embedding_provider': 'cloud', 'mode': 'supervised'}})

    def test_explicit_cloud_options_are_bound_and_local_assets_device_reach_environment(self):
        with self.settings():
            plan = self.cloud()
            plan['params'].update({'embedding_model': 'explicit-model', 'embedding_api_base': 'https://selected.fixture/v1',
                                   'embedding_dimensions': 128, 'embedding.normalize': False})
            policy = execution_policy(plan)
            self.assertEqual(policy['embedding']['model'], 'explicit-model')
            self.assertEqual(policy['embedding']['endpoint'], 'https://selected.fixture/v1/embeddings')
            self.assertEqual(policy['embedding']['dimensions'], 128)
            self.assertEqual(training_environment(policy)['EMBEDDING_NORMALIZE'], 'false')
            self.assertEqual(training_environment(policy)['EMBEDDING_DIMENSIONS'], '128')
        with tempfile.TemporaryDirectory() as home:
            Path(home, 'config.json').write_text('{}')
            policy = execution_policy({'modelId': 'ctm', 'device': 'cuda:2', 'params': {'embedding.model_path': home}, 'timeoutSeconds': 30})
            environment = training_environment(policy)
            self.assertEqual(environment['SBERT_MODEL_PATH'], str(Path(home).resolve()))
            self.assertEqual(environment['CUDA_VISIBLE_DEVICES'], '2')

    def test_direct_worker_submission_without_consent_never_spawns(self):
        with tempfile.TemporaryDirectory() as home, patch('workers.local_compute.subprocess.Popen') as spawn:
            payload = {'home': home, 'jobId': 'job-' + 'a' * 64, 'plan': {}, 'dataset': {}}
            with self.assertRaisesRegex(ValueError, '确认'):
                submit(payload)
            spawn.assert_not_called()

    def test_actual_http_boundary_rejects_changed_target_model_credential_and_budget_overrun(self):
        with self.settings(), tempfile.TemporaryDirectory() as home:
            policy = execution_policy(self.cloud())
            db_file = str(Path(home) / 'compute.sqlite')
            with database(home) as db:
                db.execute('INSERT INTO jobs(id,request,state,value,updated) VALUES (?,?,?,?,?)', ('job-fixture', json.dumps({'execution': policy}), 'running', '{}', 0))
            calls = []
            send = guarded_urlopen(lambda *args, **kwargs: calls.append(args[0].full_url), db_file, 'job-fixture')
            def request(endpoint='https://embedding.fixture/v1/embeddings', model='fixture-model', key='fixture-key'):
                return urllib.request.Request(endpoint, data=json.dumps({'model': model, 'input': ['fixture']}).encode(), headers={'Authorization': 'Bearer ' + key}, method='POST')
            for value in [request(endpoint='https://other.fixture'), request(model='changed'), request(key='changed')]:
                with self.assertRaises(PermissionError):
                    send(value)
            self.assertEqual(calls, [])
            send(request()); send(request())
            with self.assertRaisesRegex(PermissionError, '额度'):
                send(request())
            self.assertEqual(len(calls), 2)
            with closing(sqlite3.connect(db_file)) as db, db:
                self.assertIn('额度', db.execute('SELECT reason FROM external_denials WHERE job_id=?', ('job-fixture',)).fetchone()[0])
            # A new process/transport sees the same consumed budget.
            again = guarded_urlopen(lambda *_: self.fail('must not send'), db_file, 'job-fixture')
            with self.assertRaises(PermissionError):
                again(request())
            with self.assertRaises(PermissionError):
                NoRedirect().redirect_request(None, None, 302, '', {}, 'https://other.fixture')

    def test_failed_requests_still_consume_budget(self):
        with self.settings(), tempfile.TemporaryDirectory() as home:
            policy = execution_policy(self.cloud()); policy['maxExternalRequests'] = 1
            with database(home) as db:
                db.execute('INSERT INTO jobs(id,request,state,value,updated) VALUES (?,?,?,?,?)', ('job-fixture', json.dumps({'execution': policy}), 'running', '{}', 0))
            def fail(*_args, **_kwargs):
                raise TimeoutError('uncertain outcome')
            send = guarded_urlopen(fail, str(Path(home) / 'compute.sqlite'), 'job-fixture')
            request = urllib.request.Request(policy['embedding']['endpoint'], data=b'{"model":"fixture-model","input":["fixture"]}', headers={'Authorization': 'Bearer fixture-key'}, method='POST')
            with self.assertRaises(TimeoutError):
                send(request)
            with self.assertRaises(PermissionError):
                send(request)


if __name__ == '__main__':
    unittest.main()
