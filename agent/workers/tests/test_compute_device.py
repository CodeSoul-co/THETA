import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'trainning'))
from workers.compute_device import requested_device, resolve_device, device_event
from workers.gpu_fallback import GPUAwareProcessRunner, GPUExecutionError
from workers.progress_events import parse_progress
from worker.errors import JobCancelled, ProcessExecutionError, RetryableJobError


class ComputeDeviceTests(unittest.TestCase):
    def test_windows_default_and_explicit_cpu(self):
        with patch('workers.compute_device.sys.platform', 'win32'):
            self.assertEqual(requested_device({}), 'auto')
            self.assertEqual(requested_device({'device': 'cpu'}), 'cpu')
        with patch('workers.compute_device.sys.platform', 'darwin'):
            self.assertEqual(requested_device({}), 'cpu')

    def test_actual_kernel_probe_and_failure_fallback(self):
        with patch('workers.compute_device.subprocess.run') as probe:
            probe.return_value = SimpleNamespace(returncode=0, stdout='THETA_CUDA_READY\n')
            self.assertEqual(resolve_device({'device': 'auto', 'modelId': 'prodlda'}), ('cuda:0', 'selected'))
            self.assertIn('synchronize()', probe.call_args.args[0][-1])
            self.assertEqual(probe.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '0')
            probe.return_value = SimpleNamespace(returncode=1, stdout='')
            self.assertEqual(resolve_device({'device': 'auto', 'modelId': 'theta'}), ('cpu', 'unavailable'))
            probe.side_effect = subprocess.TimeoutExpired('probe', 30)
            self.assertEqual(resolve_device({'device': 'auto', 'modelId': 'ctm'}), ('cpu', 'unavailable'))

    def test_cpu_and_classical_models_do_not_probe(self):
        with patch('workers.compute_device.subprocess.run') as probe:
            self.assertEqual(resolve_device({'device': 'cpu', 'modelId': 'theta'}), ('cpu', 'selected'))
            self.assertEqual(resolve_device({'device': 'auto', 'modelId': 'lda'}), ('cpu', 'cpu_model'))
            probe.assert_not_called()

    def call(self, code, device='cuda:0', cancel=False, timeout=5):
        with tempfile.TemporaryDirectory() as folder:
            return GPUAwareProcessRunner(poll_seconds=.01).run(
                command=[sys.executable, '-c', code], cwd=Path(folder),
                env={**os.environ, 'THETA_COMPUTE_DEVICE': device}, log_path=Path(folder)/'worker.log',
                timeout_seconds=timeout, is_cancelled=lambda: cancel, heartbeat=lambda: None, shutdown_grace_seconds=1)

    def test_cuda_oom_is_retryable_but_data_errors_are_not(self):
        with self.assertRaises(GPUExecutionError):
            self.call("raise RuntimeError('CUDA out of memory')")
        for code, device in [("raise ValueError('invalid column')", 'cuda:0'), ("raise RuntimeError('CUDA out of memory')", 'cpu')]:
            with self.assertRaises(ProcessExecutionError) as caught:
                self.call(code, device)
            self.assertNotIsInstance(caught.exception, GPUExecutionError)
        self.call("print('CUDA available')")

    def test_cancel_and_timeout_never_retry_even_after_gpu_error_output(self):
        code = "import time; print('RuntimeError: CUDA error', flush=True); time.sleep(5)"
        with self.assertRaises(JobCancelled): self.call(code, cancel=True)
        with self.assertRaises(RetryableJobError) as caught: self.call(code, timeout=.05)
        self.assertNotIsInstance(caught.exception, GPUExecutionError)

    def test_public_progress_omits_raw_errors(self):
        self.assertEqual(parse_progress(device_event('cpu', 'fallback')), {'kind': 'device', 'device': 'cpu', 'status': 'fallback'})
        self.assertIsNone(parse_progress('THETA_DEVICE {"device":"secret","status":"fallback"}'))
        value=parse_progress('THETA_DEVICE {"device":"cpu","status":"fallback","error":"private corpus"}')
        self.assertNotIn('error', value)


class LocalFallbackIntegrationTests(unittest.TestCase):
    def test_failed_gpu_attempt_restarts_cleanly_and_retains_approval(self):
        from workers import local_compute
        from worker.pipeline import JobPaths, PipelineResult
        calls = []
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            models = root / 'src/models'; models.mkdir(parents=True)
            script = models / 'fixture_compute.py'
            script.write_text("import os\nfrom pathlib import Path\nr=Path(os.environ['RESULT_DIR'])\n"
                "if os.environ['THETA_COMPUTE_DEVICE'].startswith('cuda:'):\n"
                " (r/'partial').write_text('invalid')\n raise RuntimeError('CUDA out of memory')\n"
                "assert os.environ['CUDA_VISIBLE_DEVICES']==''\nassert not (r/'partial').exists()\n"
                "(r/'complete').write_text('CPU result')\n")
            class Pipeline:
                def __init__(self, config, storage, runner): self.config=config; self.runner=runner
                def execute(self, spec, is_cancelled, progress):
                    calls.append((self.config.resource_class, self.config.gpu_id, spec.resources.timeout_seconds))
                    paths=JobPaths.create(self.config.job_root, spec)
                    self.runner.run(command=[sys.executable, str(script)], cwd=models,
                        env={'WORKSPACE_DIR':str(paths.workspace), 'RESULT_DIR':str(paths.result_root)},
                        log_path=paths.log_file, timeout_seconds=spec.resources.timeout_seconds,
                        is_cancelled=is_cancelled, heartbeat=lambda:None, shutdown_grace_seconds=1)
                    return PipelineResult(paths.result_root, paths, .1)
            home=str(root/'home'); job='job-'+'b'*64
            payload={'home':home, 'jobId':job, 'runId':'run-test',
                'dataset':{'datasetRef':'fixture', 'sha256':'original'},
                'plan':{'modelId':'prodlda','device':'auto','params':{},'timeoutSeconds':30},
                'execution':{'embedding':{'mode':'local'},'device':'auto','runtime':{'profile':'topic','revision':'test'}}}
            with local_compute.database(home) as db:
                db.execute('INSERT INTO jobs(id,request,state,value,updated) VALUES (?,?,?,?,?)',
                    (job,json.dumps(payload),'queued',json.dumps({'id':job,'status':'queued'}),0))
            def normalize(payload,path): path.parent.mkdir(parents=True);path.write_text('text\nfixture\n')
            with patch.object(local_compute,'assert_authorized'), patch.object(local_compute,'normalize_dataset',side_effect=normalize), \
                 patch.object(local_compute,'engine_root',return_value=root), patch.object(local_compute,'resolve_device',return_value=('cuda:0','selected')), \
                 patch('workers.pipeline_adapter.AgentThetaPipeline',Pipeline):
                local_compute.run(home,job)
            state=local_compute.status({'home':home,'jobId':job})
            self.assertEqual(state['status'],'completed',state.get('error'))
            self.assertEqual(state['computeDevice'],'cpu'); self.assertTrue(state['gpuFallback'])
            self.assertEqual([(c[0],c[1]) for c in calls],[('gpu',0),('cpu',None)])
            self.assertLess(calls[1][2],calls[0][2])
            jobroot=Path(home)/'compute'/job
            self.assertIn('CUDA out of memory',(jobroot/'gpu-attempt.log').read_text())
            self.assertEqual(len(list((jobroot/'jobs').glob('*/workspace'))),1)
            self.assertEqual(state['plan'],payload['plan'])
            self.assertTrue(any(e.get('status')=='fallback' for e in state['telemetry']['events']))
            with local_compute.database(home) as db:
                saved=json.loads(db.execute('SELECT request FROM jobs WHERE id=?',(job,)).fetchone()[0])
            self.assertEqual(saved,payload)
