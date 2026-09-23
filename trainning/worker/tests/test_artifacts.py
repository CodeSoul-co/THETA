from __future__ import annotations

import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from worker.artifacts import ArtifactPublisher
from worker.pipeline import JobPaths, PipelineResult
from worker.storage import FilesystemObjectStorage
from worker.tests.helpers import execution_spec


class ArtifactTests(unittest.TestCase):
    def test_publishes_archive_manifest_log_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = execution_spec()
            paths = JobPaths.create(root / "jobs", spec)
            result_dir = paths.result_root / "model"
            result_dir.mkdir(parents=True)
            (result_dir / "weights.npy").write_bytes(b"weights")
            (result_dir / "visualizations").mkdir()
            (result_dir / "visualizations" / "topics.svg").write_text("<svg/>", encoding="utf-8")
            (result_dir / "metrics.json").write_text(
                json.dumps({"npmi": 0.42}), encoding="utf-8"
            )
            paths.dataset_file.write_text("text\n原始文本\n", encoding="utf-8")
            (paths.workspace / "texts.json").write_text('["原始文本"]', encoding="utf-8")
            (paths.workspace / "source_rows.npy").write_bytes(b"fixture-index")
            (paths.workspace / "ignored.bin").write_bytes(b"not published")
            paths.log_file.write_text("training output", encoding="utf-8")
            object_root = root / "objects"
            bundle = ArtifactPublisher(FilesystemObjectStorage(object_root)).publish(
                spec, PipelineResult(result_dir, paths, 1.25)
            )

            archive = object_root / Path(*bundle.model_weights_path.split("/"))
            manifest = object_root / "training-results" / "42" / "attempt-1" / "manifest.json"
            self.assertTrue(archive.is_file())
            self.assertTrue(manifest.is_file())
            self.assertEqual(bundle.metrics["npmi"], 0.42)
            self.assertIn("visualizations/topics.svg", bundle.output_files["files"])
            with tarfile.open(archive, "r:gz") as value:
                self.assertIn("result/weights.npy", value.getnames())
                self.assertIn("result/visualizations/topics.svg", value.getnames())
                self.assertIn("input/data.csv", value.getnames())
                self.assertIn("workspace/texts.json", value.getnames())
                self.assertIn("workspace/source_rows.npy", value.getnames())
                self.assertNotIn("workspace/ignored.bin", value.getnames())
                self.assertIn("worker.log", value.getnames())
                self.assertIn("使用方法.md", value.getnames())
                self.assertIn("代码/src/models/model_delivery.py", value.getnames())
                self.assertFalse(any('/.env' in name or '__pycache__' in name for name in value.getnames()))
            self.assertEqual(bundle.output_files["integrity"]["archive_sha256"], json.loads(manifest.read_text())["archive"]["sha256"])
            self.assertEqual(len(bundle.output_files["integrity"]["normalized_input_sha256"]), 64)

    def test_reads_native_model_specific_metrics_filename(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metrics_k2.json").write_text(json.dumps({"NPMI": 0.21}), encoding="utf-8")
            self.assertEqual(ArtifactPublisher._read_metrics(root), {"NPMI": 0.21})

            (root / "metrics_k2.json").unlink()
            (root / "metrics_zero_shot.json").write_text(json.dumps({"TD": 0.74}), encoding="utf-8")
            self.assertEqual(ArtifactPublisher._read_metrics(root), {"TD": 0.74})


if __name__ == "__main__":
    unittest.main()
