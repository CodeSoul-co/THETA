import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from workers.progress_events import ProgressRecorder, parse_progress
from workers.job_observation import observe


class TrainingProgressTests(unittest.TestCase):
    def test_epoch_metrics_stage_and_carriage_return_batches(self):
        value = parse_progress('2026-09-24 INFO Stage2 Epoch 3/50: train_loss=1.2, val_loss=1e-3, kl=0.02')
        self.assertEqual((value['current'], value['total'], value['stage']), (3, 50, 'stage2'))
        self.assertEqual(value['metrics'], {'train_loss': 1.2, 'val_loss': .001, 'kl_loss': .02})
        value = parse_progress('\x1b[32mEpoch 4/50: 25%|██ | 2/8 [00:01<00:03, 2it/s, loss=0.5]\x1b[0m')
        self.assertEqual(value['batch'], {'current': 2, 'total': 8})
        self.assertEqual(value['metrics']['loss'], .5)
        self.assertEqual(parse_progress('Stage1 Epoch 1: 50%|##| 1/2 [00:01]')['total'], None)
        self.assertEqual(parse_progress('Epoch  5/50 | Train: 2 | Val: 1 | NLL: .8 | KL: .2 | PPL: 5')['metrics']['perplexity'], 5)
        self.assertIsNone(parse_progress('Epoch 51/50'))
        self.assertIsNone(parse_progress('corpus text, key=secret, /private/model/path'))

    def test_journal_restores_finished_history_and_resets_current_step(self):
        with tempfile.TemporaryDirectory() as folder:
            home = Path(folder).resolve()
            job = {'id': 'job-' + 'a' * 64, 'phase': 'training', 'status': 'running'}
            root = home / 'compute' / job['id']
            log = root / 'jobs/task-1-attempt-1/worker.log'
            log.parent.mkdir(parents=True)
            log.write_text('unrecognized output')
            recorder = ProgressRecorder(root / 'progress.jsonl')
            recorder('COMMAND: private-key private-path')
            recorder('Epoch 1/3 - Loss: 5.2, Recon: 5.0, KL: 0.2')
            recorder('Epoch 2/3 - Loss: 4.2')
            running = observe(home, job, 1000, now=1001)
            self.assertEqual(running['detail']['current'], 2)
            recorder('Running Visualizations')
            self.assertIsNone(observe(home, job, 1000, now=1001)['detail'])
            job.update(status='completed', phase='completed')
            restored = observe(home, job, 1000, now=1001)
            self.assertEqual([e['current'] for e in restored['events'] if e['kind'] == 'epoch'], [1, 2])
            self.assertNotIn('private-', (root / 'progress.jsonl').read_text())
            with (root / 'progress.jsonl').open('a') as handle:
                handle.write('{"incomplete":')
            self.assertEqual(observe(home, job, 1000, now=1001)['events'], restored['events'])

    def test_records_child_output_before_process_finishes(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'trainning'))
        from worker.process import ProcessRunner
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            recorder = ProgressRecorder(root / 'progress.jsonl')
            during_run = []
            def heartbeat():
                if recorder.file.exists():
                    during_run.append(recorder.file.read_text())
            code = "import time; print('Epoch 1/2 - Loss: 2', flush=True); time.sleep(1.3); print('Epoch 2/2 - Loss: 1', flush=True)"
            ProcessRunner(poll_seconds=.02, on_output=recorder).run(
                [sys.executable, '-u', '-c', code], root, os.environ, root / 'worker.log', 10, lambda: False, heartbeat, 1)
            self.assertTrue(any('"current": 1' in text and '"current": 2' not in text for text in during_run))
            self.assertEqual(json.loads(recorder.file.read_text().splitlines()[-1])['current'], 2)

    def test_batch_noise_is_bounded_but_epoch_summaries_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder = ProgressRecorder(Path(folder) / 'progress.jsonl')
            with patch('workers.progress_events.time.time', return_value=100):
                for batch in range(101):
                    recorder(f'Epoch 1/2: 10%|#| {batch}/100 [00:01, loss=2]')
                recorder('Epoch 1/2 - Loss: 1')
            events = [json.loads(line) for line in recorder.file.read_text().splitlines()]
            self.assertEqual(len(events), 3)
            self.assertEqual(events[-1]['kind'], 'epoch')

    def test_embedding_progress_is_validated_and_visible_during_preprocessing(self):
        line = 'THETA_EMBEDDING ' + json.dumps(dict(source='cloud', scope='documents', current=2, total=5, completedBatches=1, chunks=3, chunkTotal=6, secret='excluded'))
        event = parse_progress(line)
        self.assertEqual(event['kind'], 'embedding')
        self.assertNotIn('secret', event)
        self.assertIsNone(parse_progress(line.replace('"current": 2', '"current": 9')))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve() / 'compute' / ('job-' + 'b' * 64)
            log = root / 'jobs/task-1-attempt-1/worker.log'
            log.parent.mkdir(parents=True)
            log.write_text(line)
            state = observe(folder, dict(id=root.name, status='running', phase='preprocessing'), 1, now=2)
            self.assertEqual(state['detail']['current'], 2)
            self.assertEqual(state['activity'], 'embedding')

    def test_failed_process_exposes_specific_cause_without_credentials(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'trainning'))
        from worker.process import ProcessRunner
        from worker.errors import ProcessExecutionError
        with tempfile.TemporaryDirectory() as folder:
            for source, expected in [
                ("raise ValueError('BOW vocabulary is empty after tokenization')", '没有可用词语'),
                ("raise RuntimeError('api_key=sk-private-example bad configuration')", '[已隐藏]'),
            ]:
                with self.assertRaises(ProcessExecutionError) as raised:
                    ProcessRunner(poll_seconds=.01).run(command=[sys.executable, '-c', source], cwd=Path(folder), env=os.environ,
                        log_path=Path(folder) / 'worker.log', timeout_seconds=10, is_cancelled=lambda: False, heartbeat=lambda: None, shutdown_grace_seconds=1)
                self.assertIn(expected, str(raised.exception))
                self.assertNotIn('sk-private-example', str(raised.exception))
