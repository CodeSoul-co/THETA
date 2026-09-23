from __future__ import annotations

import json
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .pipeline import PipelineResult
from .protocol import ExecutionSpec
from .storage import ObjectStorage, sha256_file


SELECTED_SUFFIXES = {".json", ".csv", ".txt", ".png", ".jpg", ".jpeg", ".svg", ".pdf", ".html"}
MAX_SELECTED_FILE_BYTES = 50 * 1024 * 1024
WORKSPACE_EVIDENCE = {
    "bow_matrix.npy", "bow_matrix.npz", "config.json", "covariates.npy",
    "covariate_names.json", "covariate_encoding.json", "texts.json",
    "source_rows.npy", "row_provenance.json", "time_indices.npy",
    "time_slices.json", "vocab.json", "vocab.txt",
}


@dataclass(frozen=True)
class ArtifactBundle:
    model_weights_path: str
    metrics: dict[str, Any]
    output_files: dict[str, Any]


class ArtifactPublisher:
    def __init__(self, storage: ObjectStorage):
        self.storage = storage

    def publish(self, spec: ExecutionSpec, result: PipelineResult) -> ArtifactBundle:
        artifact_dir = result.paths.root / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        archive_path = artifact_dir / "model.tar.gz"
        manifest_path = artifact_dir / "manifest.json"

        files = self._describe_files(result.result_dir)
        if not result.paths.dataset_file.is_file():
            raise RuntimeError("normalized training input is missing; result bundle cannot prove row provenance")
        self._create_archive(result, archive_path)
        metrics = self._read_metrics(result.result_dir)

        archive_key = spec.output_prefix + archive_path.name
        log_key = spec.output_prefix + "worker.log"
        manifest_key = spec.output_prefix + manifest_path.name
        selected: dict[str, str] = {}
        for item in files:
            relative = str(item["path"])
            source = result.result_dir / Path(*relative.split("/"))
            if source.suffix.lower() not in SELECTED_SUFFIXES:
                continue
            if source.stat().st_size > MAX_SELECTED_FILE_BYTES:
                continue
            key = spec.output_prefix + "files/" + relative
            self.storage.upload(source, key)
            selected[relative] = key

        manifest = {
            "schema_version": 1,
            "task_id": spec.task_id,
            "attempt": spec.attempt,
            "model": spec.model.name,
            "runtime": spec.runtime.key,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "elapsed_seconds": round(result.elapsed_seconds, 3),
            "archive": {
                "object_key": archive_key,
                "size_bytes": archive_path.stat().st_size,
                "sha256": sha256_file(archive_path),
            },
            "normalized_input": {
                "path": "input/data.csv",
                "size_bytes": result.paths.dataset_file.stat().st_size,
                "sha256": sha256_file(result.paths.dataset_file),
            },
            "workspace_files": self._describe_workspace(result.paths.workspace),
            "log_object_key": log_key if result.paths.log_file.is_file() else None,
            "selected_files": selected,
            "result_files": files,
            "metrics": metrics,
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

        # Upload the archive first and the manifest last. The manifest is therefore the
        # commit marker for a completely published result bundle.
        self.storage.upload(archive_path, archive_key)
        if result.paths.log_file.is_file():
            self.storage.upload(result.paths.log_file, log_key)
        self.storage.upload(manifest_path, manifest_key)

        integrity = {
            "archive_sha256": manifest["archive"]["sha256"],
            "archive_size_bytes": manifest["archive"]["size_bytes"],
            "manifest_sha256": sha256_file(manifest_path),
            "normalized_input_sha256": manifest["normalized_input"]["sha256"],
            "normalized_input_size_bytes": manifest["normalized_input"]["size_bytes"],
        }

        return ArtifactBundle(
            model_weights_path=archive_key,
            metrics=metrics,
            output_files={
                "archive": archive_key,
                "manifest": manifest_key,
                "log": log_key if result.paths.log_file.is_file() else None,
                "files": selected,
                "integrity": integrity,
            },
        )

    @staticmethod
    def _create_archive(result: PipelineResult, destination: Path) -> None:
        with tarfile.open(destination, "w:gz") as archive:
            archive.add(result.result_dir, arcname="result", recursive=True)
            archive.add(result.paths.dataset_file, arcname="input/data.csv", recursive=False)
            for item in ArtifactPublisher._workspace_paths(result.paths.workspace):
                archive.add(item, arcname=f"workspace/{item.name}", recursive=False)
            if result.paths.log_file.is_file() and not result.paths.log_file.is_symlink():
                archive.add(result.paths.log_file, arcname="worker.log", recursive=False)
            # Ship the same fixed loader and plotting implementation as the web
            # delivery, without credentials, caches, or embedding base weights.
            code_root = Path(__file__).resolve().parents[2] / "src" / "models"
            for file in sorted(code_root.rglob("*")):
                relative = file.relative_to(code_root)
                if any(part.startswith('.') or part == '__pycache__' for part in relative.parts):
                    continue
                if file.is_symlink() or any(parent.is_symlink() for parent in file.parents if parent != code_root.parent):
                    continue
                if file.is_file() and (file.suffix == '.py' or relative.parts[0] == 'resources' and file.suffix == '.txt'):
                    archive.add(file, arcname=f"代码/src/models/{relative.as_posix()}", recursive=False)
            usage = code_root / 'MODEL_USAGE.zh-CN.md'
            if usage.is_file():
                archive.add(usage, arcname='使用方法.md', recursive=False)

    @staticmethod
    def _workspace_paths(workspace: Path) -> list[Path]:
        if not workspace.is_dir() or workspace.is_symlink():
            return []
        return [
            item for item in sorted(workspace.iterdir())
            if item.name in WORKSPACE_EVIDENCE and item.is_file() and not item.is_symlink()
        ]

    @staticmethod
    def _describe_workspace(workspace: Path) -> list[dict[str, Any]]:
        return [
            {"path": f"workspace/{item.name}", "size_bytes": item.stat().st_size, "sha256": sha256_file(item)}
            for item in ArtifactPublisher._workspace_paths(workspace)
        ]

    @staticmethod
    def _describe_files(result_dir: Path) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for path in sorted(result_dir.rglob("*")):
            if path.is_symlink():
                raise RuntimeError(f"result bundle cannot contain symbolic links: {path.relative_to(result_dir)}")
            if not path.is_file() and not path.is_dir():
                raise RuntimeError(f"result bundle contains a non-regular entry: {path.relative_to(result_dir)}")
            if not path.is_file():
                continue
            entries.append(
                {
                    "path": path.relative_to(result_dir).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        return entries

    @staticmethod
    def _read_metrics(result_dir: Path) -> dict[str, Any]:
        candidates = [*sorted(result_dir.rglob("metrics.json")), *sorted(result_dir.rglob("metrics_*.json"))]
        for path in candidates:
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(value, dict):
                return value
        return {}
