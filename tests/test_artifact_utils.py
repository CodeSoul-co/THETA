import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODELS_DIR = Path(__file__).resolve().parents[1] / "src" / "models"
sys.path.insert(0, str(MODELS_DIR))

from artifact_utils import (
    BOW_ARTIFACT_FILENAMES,
    BOW_MANIFEST_NAME,
    adapter_signature,
    commit_bow_artifacts,
    has_valid_bow_manifest,
    manifest_signature,
    validate_bow_manifest,
)


def _artifact_tree():
    return ast.parse(
        (MODELS_DIR / "artifact_utils.py").read_text(encoding="utf-8")
    )


def _function(tree, name):
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


class BowManifestTests(unittest.TestCase):
    def _commit(self, directory):
        temp_paths = {}
        contents = {
            "matrix": b"matrix-bytes",
            "vocab_embeddings": b"embedding-bytes",
            "vocab": b"alpha\nbeta\n",
        }
        for role, content in contents.items():
            temp_path = Path(directory) / f".{role}.tmp"
            temp_path.write_bytes(content)
            temp_paths[role] = temp_path

        metadata = {
            "matrix": {
                "shape": [2, 2],
                "format": "scipy_csr_npz",
            },
            "vocab_embeddings": {
                "shape": [2, 4],
                "dtype": "float32",
            },
            "vocab": {
                "count": 2,
                "encoding": "utf-8",
            },
        }
        return commit_bow_artifacts(directory, temp_paths, metadata)

    def test_commit_publishes_valid_manifest_and_signature(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._commit(directory)
            validated, signature = validate_bow_manifest(directory)

            self.assertEqual(manifest, validated)
            self.assertEqual(manifest["signature"], signature)
            self.assertEqual("theta.bow-artifacts", manifest["schema"])
            self.assertEqual(1, manifest["version"])
            self.assertEqual(
                BOW_ARTIFACT_FILENAMES,
                {
                    role: entry["filename"]
                    for role, entry in manifest["artifacts"].items()
                },
            )
            self.assertTrue(has_valid_bow_manifest(directory))

    def test_checksum_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            self._commit(directory)
            matrix_path = (
                Path(directory) / BOW_ARTIFACT_FILENAMES["matrix"]
            )
            matrix_path.write_bytes(b"tampered")

            with self.assertRaisesRegex(ValueError, "checksum"):
                validate_bow_manifest(directory)
            self.assertFalse(has_valid_bow_manifest(directory))

    def test_incomplete_canonical_set_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            self._commit(directory)
            (
                Path(directory)
                / BOW_ARTIFACT_FILENAMES["vocab_embeddings"]
            ).unlink()

            with self.assertRaisesRegex(ValueError, "missing"):
                validate_bow_manifest(directory)
            self.assertFalse(has_valid_bow_manifest(directory))

    def test_partial_canonical_files_without_manifest_are_not_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            (
                Path(directory) / BOW_ARTIFACT_FILENAMES["matrix"]
            ).write_bytes(b"partial")

            self.assertFalse(has_valid_bow_manifest(directory))

    def test_commit_removes_stale_legacy_dense_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            legacy_path = Path(directory) / "bow_matrix.npy"
            legacy_path.write_bytes(b"stale")

            self._commit(directory)

            self.assertFalse(legacy_path.exists())

    def test_manifest_metadata_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            self._commit(directory)
            manifest_path = Path(directory) / BOW_MANIFEST_NAME
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"]["vocab"]["count"] = 99
            manifest_path.write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "signature"):
                validate_bow_manifest(directory)

    def test_cross_component_vocab_dimensions_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            temp_paths = {}
            for role in BOW_ARTIFACT_FILENAMES:
                temp_path = Path(directory) / f".{role}.tmp"
                temp_path.write_bytes(role.encode("ascii"))
                temp_paths[role] = temp_path

            metadata = {
                "matrix": {
                    "shape": [2, 3],
                    "format": "scipy_csr_npz",
                },
                "vocab_embeddings": {
                    "shape": [2, 4],
                    "dtype": "float32",
                },
                "vocab": {
                    "count": 2,
                    "encoding": "utf-8",
                },
            }

            with self.assertRaisesRegex(ValueError, "vocab dimensions"):
                commit_bow_artifacts(directory, temp_paths, metadata)

    def test_loader_rejects_signed_cross_component_dimension_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            self._commit(directory)
            manifest_path = Path(directory) / BOW_MANIFEST_NAME
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"]["matrix"]["shape"] = [2, 3]
            manifest["signature"] = manifest_signature(manifest)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "vocab dimensions"):
                validate_bow_manifest(directory)

    def test_manifest_is_invalidated_before_components_and_published_last(self):
        function = _function(_artifact_tree(), "commit_bow_artifacts")
        calls = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "os"
            and node.func.attr in {"remove", "replace"}
        ]
        manifest_remove = next(
            node
            for node in calls
            if node.func.attr == "remove"
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "manifest_path"
        )
        component_replaces = [
            node
            for node in calls
            if node.func.attr == "replace"
            and not (
                isinstance(node.args[1], ast.Name)
                and node.args[1].id == "manifest_path"
            )
        ]
        manifest_replace = next(
            node
            for node in calls
            if node.func.attr == "replace"
            and isinstance(node.args[1], ast.Name)
            and node.args[1].id == "manifest_path"
        )

        self.assertTrue(component_replaces)
        self.assertLess(
            manifest_remove.lineno,
            min(node.lineno for node in component_replaces),
        )
        self.assertGreater(
            manifest_replace.lineno,
            max(node.lineno for node in component_replaces),
        )


class AdapterSignatureTests(unittest.TestCase):
    def test_adapter_signature_is_deterministic_and_detects_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter_dir = Path(directory)
            (adapter_dir / "adapter_config.json").write_text(
                '{"r": 8}',
                encoding="utf-8",
            )
            weights = adapter_dir / "adapter_model.safetensors"
            weights.write_bytes(b"weights")

            first = adapter_signature(adapter_dir)
            second = adapter_signature(adapter_dir)
            self.assertEqual(first, second)

            weights.write_bytes(b"different")
            self.assertNotEqual(first, adapter_signature(adapter_dir))

    def test_adapter_signature_requires_config_and_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter_dir = Path(directory)
            (adapter_dir / "adapter_config.json").write_text(
                "{}",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "weights"):
                adapter_signature(adapter_dir)
