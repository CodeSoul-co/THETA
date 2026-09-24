import hashlib
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import zipfile

from workers import cuda_runtime
from workers.progress_events import ProgressRecorder
from workers.job_observation import observe
from worker.errors import JobCancelled


class CUDARuntimeTests(unittest.TestCase):
    def wheel(self, invalid=False):
        content=io.BytesIO()
        with zipfile.ZipFile(content,'w') as archive:
            archive.writestr('../escape.py' if invalid else 'torch/__init__.py','VERSION="fixture"')
        return content.getvalue()

    def test_verified_install_reuses_cache_without_redownload(self):
        data=self.wheel(); events=[]; downloads=[]
        def download(path,notify,check): check();downloads.append(path);path.write_bytes(data)
        with tempfile.TemporaryDirectory() as home, patch.dict('os.environ',{},clear=True), \
             patch.object(cuda_runtime,'compatible_driver',return_value=True), \
             patch.object(cuda_runtime,'WHEEL_SHA256',hashlib.sha256(data).hexdigest()), \
             patch.object(cuda_runtime,'download',side_effect=download):
            path=cuda_runtime.prepare_runtime(home,events.append,lambda:False,time.monotonic()+30)
            self.assertTrue(cuda_runtime.ready(path))
            self.assertFalse((path.parent/'torch.whl.part').exists())
            self.assertEqual(cuda_runtime.prepare_runtime(home,events.append,lambda:False,time.monotonic()+30),path)
            self.assertEqual(len(downloads),1)
            self.assertFalse(list(path.parent.glob('install-*')))

    def test_checksum_failure_and_invalid_archive_fall_back_without_installing(self):
        for bad_hash in [True,False]:
            data=self.wheel(invalid=not bad_hash);events=[]
            with tempfile.TemporaryDirectory() as home, patch.dict('os.environ',{},clear=True), \
                 patch.object(cuda_runtime,'compatible_driver',return_value=True), \
                 patch.object(cuda_runtime,'WHEEL_SHA256','bad' if bad_hash else hashlib.sha256(data).hexdigest()), \
                 patch.object(cuda_runtime,'download',side_effect=lambda p,n,c:p.write_bytes(data)):
                self.assertIsNone(cuda_runtime.prepare_runtime(home,events.append,lambda:False,time.monotonic()+30))
                self.assertIn('setup_failed',events[-1])
                self.assertFalse(list(Path(home).rglob('.verified')))
                self.assertFalse(list(Path(home).rglob('escape.py')))

    def test_cancel_is_propagated_and_no_driver_never_downloads(self):
        with tempfile.TemporaryDirectory() as home, patch.dict('os.environ',{},clear=True), \
             patch.object(cuda_runtime,'compatible_driver',return_value=False), patch.object(cuda_runtime,'download') as download:
            self.assertIsNone(cuda_runtime.prepare_runtime(home,lambda e:None,lambda:False,time.monotonic()+30))
            download.assert_not_called()
        with tempfile.TemporaryDirectory() as home, patch.dict('os.environ',{},clear=True), \
             patch.object(cuda_runtime,'compatible_driver',return_value=True):
            with self.assertRaises(JobCancelled):
                cuda_runtime.prepare_runtime(home,lambda e:None,lambda:True,time.monotonic()+30)

    def test_insufficient_disk_skips_download(self):
        with tempfile.TemporaryDirectory() as home, patch.dict('os.environ',{},clear=True), \
             patch.object(cuda_runtime,'compatible_driver',return_value=True), patch.object(cuda_runtime,'download') as download, \
             patch.object(cuda_runtime.shutil,'disk_usage',return_value=type('Disk',(),{'free':100})()):
            events=[]
            self.assertIsNone(cuda_runtime.prepare_runtime(home,events.append,lambda:False,time.monotonic()+30))
            self.assertIn('disk_space',events[-1]);download.assert_not_called()

    def test_resume_and_server_ignoring_range(self):
        for status in [206,200]:
            with tempfile.TemporaryDirectory() as home:
                path=Path(home)/'part';path.write_bytes(b'abc')
                response=io.BytesIO(b'def' if status==206 else b'abcdef')
                response.status=status;response.headers={'Content-Range':'bytes 3-5/6'}
                with patch.object(cuda_runtime,'WHEEL_SIZE',6), patch.object(cuda_runtime.urllib.request,'urlopen',return_value=response) as open_url:
                    cuda_runtime.download(path,lambda *args:None,lambda:None)
                self.assertEqual(path.read_bytes(),b'abcdef')
                self.assertEqual(open_url.call_args.args[0].get_header('Range'),'bytes=3-')

    def test_progress_is_visible_before_first_training_process(self):
        with tempfile.TemporaryDirectory() as home:
            job={'id':'job-'+'d'*64,'phase':'preparing','status':'running'}
            file=Path(home)/'compute'/job['id']/'progress.jsonl'
            ProgressRecorder(file)('THETA_DEVICE '+json.dumps({'device':'cpu','status':'downloading','current':10,'total':100}))
            state=observe(home,job,time.time())
            self.assertEqual(state['detail']['status'],'downloading')
            self.assertEqual(state['detail']['current'],10)
