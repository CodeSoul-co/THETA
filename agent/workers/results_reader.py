"""Bounded result evidence reader. Never reads dataset rows or arbitrary paths."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

EVIDENCE_NAME = re.compile(r"^(?:metrics|topic_words|info|training_history|topic_evolution|covariate_effects|covariate_info)(?:_[\w.-]+)?\.json$")


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(root.rglob("*")):
        if file.is_symlink():
            raise ValueError("Result tree contains a symbolic link")
        if file.is_file():
            digest.update(file.relative_to(root).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(file_hash(file).encode("ascii"))
            digest.update(b"\n")
    return digest.hexdigest()


def file_hash(file: Path) -> str:
    digest = hashlib.sha256()
    with file.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_result_evidence(payload: dict) -> dict:
    run_id = payload.get("trainingRunId")
    artifacts = payload.get("artifacts")
    if not isinstance(run_id, str) or not run_id or not isinstance(artifacts, list):
        raise ValueError("Verified trainingRunId and artifacts are required")
    evidence = []
    skipped = []
    presentations = {'matrices': [], 'tables': [], 'figures': [], 'analysisSkipped': []}
    for artifact in artifacts:
        if not str(artifact.get("kind", "")).startswith("results"):
            continue
        source = Path(artifact["path"])
        if source.is_symlink():
            raise ValueError("Result root is a symbolic link")
        root = source.resolve(strict=True)
        if not any(part.startswith(run_id + "__") or part == run_id for part in root.parts):
            raise ValueError("Result path is not bound to this training run")
        expected = artifact.get("sha256")
        if tree_hash(root) != expected:
            raise ValueError("Result artifact changed after verification")
        for file in sorted(root.rglob("*.json")):
            if not EVIDENCE_NAME.fullmatch(file.name):
                continue
            if file.stat().st_size > 1024 * 1024 or len(evidence) >= 12:
                skipped.append(file.relative_to(root).as_posix())
                continue
            raw = file.read_bytes()
            value = json.loads(raw)
            evidence.append({
                "artifactId": artifact["artifactId"],
                "relativePath": file.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "kind": "topics" if file.name.startswith("topic_words") else "metrics" if file.name.startswith("metrics") else "context",
                "content": bounded(value),
            })
        from .result_analysis import presentation_evidence
        presentation = presentation_evidence(root, file_hash, [item for item in evidence if item['artifactId'] == artifact['artifactId']], payload.get('modelId'))
        for key, values in presentation.items():
            presentations[key].extend(values)
        if tree_hash(root) != expected:
            raise ValueError("Result artifact changed during reading")
    return {"trainingRunId": run_id, "evidence": evidence, "skipped": skipped, **presentations,
            "limitations": ["Evidence is bounded to 12 files, 20 entries per collection and 300 characters per string.",
                            "Figures are inventoried and interpreted from verified underlying data, not inspected as image pixels; never invent chart geometry.",
                            "Topic labels and substantive interpretations are hypotheses, not causal evidence."]}


def bounded(value, depth: int = 0):
    if depth > 6:
        return "[depth limit]"
    if isinstance(value, dict):
        result = {str(key)[:100]: bounded(item, depth + 1) for key, item in list(value.items())[:20]}
        if len(value) > 20:
            result["_omitted_entries"] = len(value) - 20
        return result
    if isinstance(value, list):
        result = [bounded(item, depth + 1) for item in value[:20]]
        if len(value) > 20:
            result.append({"omitted_entries": len(value) - 20})
        return result
    if isinstance(value, str):
        return value[:300]
    if isinstance(value, float):
        import math
        return value if math.isfinite(value) else None
    return value
