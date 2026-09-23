import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from workers.remote_report import generate


class RemoteReportTests(unittest.TestCase):
    def test_verified_bundle_proves_source_row_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            job_id = "job-verified"
            bundle = home / "remote-results" / job_id / "bundle-test"
            result = bundle / "result"
            workspace = bundle / "workspace"
            normalized = bundle / "input" / "data.csv"
            result.mkdir(parents=True)
            workspace.mkdir(parents=True)
            normalized.parent.mkdir(parents=True)
            original = home / "source.csv"
            content = "text,channel\n公交改善,A\n医疗服务,B\n"
            original.write_text(content, encoding="utf-8")
            normalized.write_text(content, encoding="utf-8")
            np.save(result / "theta.npy", np.array([[0.8, 0.2], [0.1, 0.9]]))
            np.save(result / "beta.npy", np.array([[0.6, 0.4], [0.2, 0.8]]))
            (result / "metrics.json").write_text(json.dumps({"NPMI": 0.2}), encoding="utf-8")
            (workspace / "texts.json").write_text(json.dumps(["公交改善", "医疗服务"]), encoding="utf-8")
            np.save(workspace / "source_rows.npy", np.array([0, 1], dtype=np.int64))
            dataset_hash = hashlib.sha256(original.read_bytes()).hexdigest()
            report = generate({
                "home": str(home), "jobId": job_id, "bundleRoot": str(bundle),
                "normalizedInputSha256": hashlib.sha256(normalized.read_bytes()).hexdigest(),
                "dataset": {"datasetRef": "dataset-test", "sha256": dataset_hash,
                            "fileName": "source.csv", "managedPath": str(original), "sizeBytes": original.stat().st_size},
                "plan": {"modelId": "lda", "textColumn": "text", "covariates": ["channel"],
                         "params": {"num_topics": 2}, "rationale": "test", "timeoutSeconds": 30},
            })
            self.assertEqual(report["schemaVersion"], "theta.result-report.v2")
            self.assertTrue(report["sourceData"]["matrixRowsAligned"])
            self.assertEqual(report["sourceData"]["rows"][0]["sourceRow"], 1)
            self.assertEqual(report["sourceData"]["rows"][0]["topicWeights"], [0.8, 0.2])
            self.assertTrue(Path(report["reportPath"]).is_file())

    def test_rejects_changed_normalized_input(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            bundle = home / "remote-results" / "job-bad" / "bundle-test"
            for relative in ["result", "workspace", "input"]:
                (bundle / relative).mkdir(parents=True)
            (bundle / "input" / "data.csv").write_text("text\nchanged\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Normalized input hash"):
                generate({"home": str(home), "jobId": "job-bad", "bundleRoot": str(bundle),
                          "normalizedInputSha256": "0" * 64, "dataset": {}, "plan": {}})


if __name__ == "__main__":
    unittest.main()
