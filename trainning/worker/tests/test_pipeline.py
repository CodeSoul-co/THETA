from __future__ import annotations

import tempfile
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

from worker.errors import PermanentJobError, RetryableJobError
from worker.pipeline import JobPaths, ThetaPipeline, filesystem_path
from worker.storage import FilesystemObjectStorage
from worker.tests.helpers import execution_spec, worker_config


class PipelineCommandTests(unittest.TestCase):
    def test_model_export_and_reload_under_long_unicode_job_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = filesystem_path(Path(directory))
            try:
                root = base / ('研究项目' + 'a' * 90) / ('b' * 80)
                config = worker_config(root)
                paths = JobPaths.create(config.job_root, execution_spec('lda'))
                pipeline = ThetaPipeline(config, FilesystemObjectStorage(root / 'objects'))
                environment = pipeline._child_environment(paths)
                result = paths.result_root / 'user_local' / 'dataset_240909878805725' / 'lda' / 'job_240909878805725_a1' / 'delivery'
                self.assertGreater(len(str(result / 'trained_model.joblib')), 320)
                engine = Path(__file__).resolve().parents[3] / 'src/models'
                code = """
import os
from pathlib import Path
import numpy as np
from model_delivery import save_trained_model, load_trained_model
folder = Path(os.environ['RESULT_DIR']) / 'user_local/dataset_240909878805725/lda/job_240909878805725_a1/delivery'
save_trained_model({'weights': np.eye(2)}, folder, 'lda', ['苹果', '代码'])
restored = load_trained_model(folder, trusted=True)
np.testing.assert_array_equal(restored['weights'], np.eye(2))
"""
                completed = subprocess.run([sys.executable, '-c', code], cwd=engine, env=environment,
                                           capture_output=True, text=True, timeout=60)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertTrue((result / 'trained_model.joblib').is_file())
                self.assertTrue((result / 'model_delivery.json').is_file())
            finally:
                shutil.rmtree(base)

    def test_baseline_commands_are_isolated_and_whitelisted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = worker_config(root)
            pipeline = ThetaPipeline(config, FilesystemObjectStorage(root / "objects"))
            spec = execution_spec("lda")
            paths = JobPaths.create(config.job_root, spec)
            prepare = pipeline.build_prepare_command(spec, paths)
            train = pipeline.build_train_command(spec, paths, None)
            self.assertIn("--output_dir", prepare)
            self.assertIn("--skip-sbert", prepare)
            self.assertIn("--bow-only", prepare)
            self.assertIn("--workspace_dir", train)
            self.assertEqual(train[train.index("--models") + 1], "lda")
            self.assertIn("--skip-viz", train)

    def test_theta_command_uses_exact_preprocessing_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = worker_config(root, "gpu")
            pipeline = ThetaPipeline(config, FilesystemObjectStorage(root / "objects"))
            spec = execution_spec("theta", gpu_count=1)
            paths = JobPaths.create(config.job_root, spec)
            data_exp = paths.result_root / "dataset_2" / "0.6B" / "theta" / "exp_test"
            command = pipeline.build_train_command(spec, paths, data_exp)
            self.assertEqual(command[command.index("--data_exp") + 1], "exp_test")
            self.assertEqual(command[command.index("--gpu") + 1], "0")

    def test_dtm_uses_time_aware_preprocessor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = worker_config(root, "gpu")
            pipeline = ThetaPipeline(config, FilesystemObjectStorage(root / "objects"))
            spec = execution_spec("dtm", gpu_count=1)
            paths = JobPaths.create(config.job_root, spec)
            command = pipeline.build_prepare_command(spec, paths)
            self.assertEqual(command[command.index("--model") + 1], "dtm")
            self.assertIn("--with-time", command)

    def test_unknown_parameter_never_becomes_cli_argument(self) -> None:
        spec = execution_spec()
        params = dict(spec.params)
        params["evil"] = "value"
        with self.assertRaises(PermanentJobError):
            ThetaPipeline._parameter_arguments(params, set(params))

    def test_only_model_matrix_pair_counts_as_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config.json').write_text('{}')
            (root / 'metrics.json').write_text('{}')
            for name in ['lda', 'theta']:
                with self.subTest(model=name), self.assertRaises(RetryableJobError):
                    ThetaPipeline._validate_result(execution_spec(name), root)
            (root / 'theta_k3.npy').write_bytes(b'fixture')
            with self.assertRaises(RetryableJobError):
                ThetaPipeline._validate_result(execution_spec('lda'), root)
            (root / 'beta_k3.npy').write_bytes(b'fixture')
            ThetaPipeline._validate_result(execution_spec('lda'), root)

    def test_bertopic_pair_without_finished_export_is_not_completed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'theta_k3.npy').write_bytes(b'fixture')
            (root / 'beta_k3.npy').write_bytes(b'fixture')
            with self.assertRaisesRegex(RetryableJobError, 'export incomplete'):
                ThetaPipeline._validate_result(execution_spec('bertopic'), root)
            (root / 'document_topics.npy').write_bytes(b'fixture')
            with self.assertRaisesRegex(RetryableJobError, 'export incomplete'):
                ThetaPipeline._validate_result(execution_spec('bertopic'), root)
            (root / 'result_manifest.json').write_text('{}')
            ThetaPipeline._validate_result(execution_spec('bertopic'), root)

    def test_child_process_uses_utf8_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = worker_config(root)
            pipeline = ThetaPipeline(config, FilesystemObjectStorage(root / "objects"))
            paths = JobPaths.create(config.job_root, execution_spec("lda"))
            environment = pipeline._child_environment(paths)
            self.assertEqual(environment["PYTHONUTF8"], "1")
            self.assertEqual(environment["PYTHONIOENCODING"], "utf-8")


if __name__ == "__main__":
    unittest.main()
