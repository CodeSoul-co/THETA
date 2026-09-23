#!/usr/bin/env python3
"""Read-only preflight inspection for THETA workflows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


SENSITIVE_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
COMMON_TEXT_COLUMNS = ("text", "content", "正文", "abstract", "body", "document")
COMMON_TIME_COLUMNS = ("timestamp", "time", "date", "year", "年份")
REQUIRED_REPO_PATHS = (
    "README.md",
    ".env.example",
    "scripts",
    "src/models/run_pipeline.py",
    "src/models/prepare_data.py",
    "src/models/main.py",
)
SCRIPT_PATHS = (
    "scripts/env_setup.sh",
    "scripts/quick_start.sh",
    "scripts/clean_data.sh",
    "scripts/train_theta.sh",
    "scripts/train_baseline.sh",
    "scripts/visualize.sh",
    "scripts/sweep_topics.sh",
    "scripts/scrape.sh",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only THETA repository, environment, and dataset preflight."
    )
    parser.add_argument("--repo", default=None, help="THETA repository root. Defaults to auto-detect.")
    parser.add_argument("--dataset", default=None, help="Dataset file or directory to inspect.")
    parser.add_argument("--text-column", default=None, help="Expected text column name.")
    parser.add_argument(
        "--mode",
        choices=("zero_shot", "supervised", "unsupervised"),
        default=None,
        help="THETA mode used to validate embedding expectations.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of Markdown.")
    return parser.parse_args()


def find_repo_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()

    candidates = [Path.cwd().resolve()]
    script_path = Path(__file__).resolve()
    candidates.extend(script_path.parents)

    for start in candidates:
        for path in (start, *start.parents):
            if (path / ".env.example").exists() and (path / "src/models/run_pipeline.py").exists():
                return path
    return Path.cwd().resolve()


def run_command(command: list[str], cwd: Path | None = None) -> dict[str, Any]:
    if not shutil.which(command[0]):
        return {"available": False, "command": command, "output": None}
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            text=True,
            capture_output=True,
            timeout=8,
            check=False,
        )
    except Exception as exc:  # pragma: no cover - defensive inspection path
        return {"available": True, "command": command, "error": str(exc)}
    output = (completed.stdout or completed.stderr or "").strip()
    return {
        "available": True,
        "command": command,
        "returncode": completed.returncode,
        "output": output[:4000],
    }


def parse_env_file(path: Path) -> dict[str, dict[str, Any]]:
    keys: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return keys
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        keys[key] = {
            "present": bool(value),
            "sensitive": any(marker in key.upper() for marker in SENSITIVE_MARKERS),
        }
    return keys


def summarize_env(repo: Path) -> dict[str, Any]:
    env_example = parse_env_file(repo / ".env.example")
    env_actual = parse_env_file(repo / ".env")
    keys: dict[str, Any] = {}
    for key in sorted(set(env_example) | set(env_actual)):
        keys[key] = {
            "in_example": key in env_example,
            "in_env": key in env_actual,
            "configured": env_actual.get(key, {}).get("present", False),
            "sensitive": env_example.get(key, env_actual.get(key, {})).get("sensitive", False),
        }
    return {
        "env_example_exists": (repo / ".env.example").exists(),
        "env_exists": (repo / ".env").exists(),
        "keys": keys,
    }


def resolve_dataset(dataset: str | None, repo: Path) -> dict[str, Any]:
    if not dataset:
        return {"provided": False, "path": None, "exists": False}

    path = Path(dataset).expanduser()
    if not path.is_absolute():
        path = repo / path
    path = path.resolve()

    result: dict[str, Any] = {
        "provided": True,
        "path": str(path),
        "exists": path.exists(),
        "is_dir": path.is_dir(),
        "selected_file": None,
    }
    if path.is_dir():
        files = sorted(
            [p for p in path.rglob("*") if p.suffix.lower() in {".csv", ".tsv", ".jsonl", ".xlsx"}]
        )
        result["candidate_files"] = [str(p) for p in files[:20]]
        if files:
            result["selected_file"] = str(files[0])
    elif path.exists():
        result["selected_file"] = str(path)
    return result


def inspect_csv(path: Path, text_column: str | None) -> dict[str, Any]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    total_rows = 0
    empty_text = 0
    total_text_len = 0
    duplicates = 0
    seen_hashes: set[str] = set()

    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        columns = reader.fieldnames or []
        detected_text = text_column
        if not detected_text:
            detected_text = next((col for col in COMMON_TEXT_COLUMNS if col in columns), None)
        if detected_text and detected_text not in columns:
            detected_text = None

        time_columns = [col for col in columns if col in COMMON_TIME_COLUMNS]
        covariate_columns = [col for col in columns if col.startswith("cov_")]
        if not covariate_columns:
            covariate_columns = [
                col
                for col in columns
                if col not in set(COMMON_TEXT_COLUMNS) | set(COMMON_TIME_COLUMNS)
                and col.lower() not in {"id", "uuid", "index"}
            ][:20]

        for row in reader:
            total_rows += 1
            if detected_text:
                text = (row.get(detected_text) or "").strip()
                if not text:
                    empty_text += 1
                total_text_len += len(text)
                digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
                if text and digest in seen_hashes:
                    duplicates += 1
                elif text:
                    seen_hashes.add(digest)

    return {
        "path": str(path),
        "format": path.suffix.lower().lstrip("."),
        "columns": columns,
        "row_count": total_rows,
        "requested_text_column": text_column,
        "detected_text_column": detected_text,
        "text_column_exists": bool(detected_text),
        "empty_text_count": empty_text if detected_text else None,
        "duplicate_text_count": duplicates if detected_text else None,
        "average_text_length": round(total_text_len / total_rows, 2) if detected_text and total_rows else None,
        "time_columns": time_columns,
        "covariate_candidates": covariate_columns,
    }


def inspect_jsonl(path: Path, text_column: str | None) -> dict[str, Any]:
    total_rows = 0
    keys: set[str] = set()
    empty_text = 0
    total_text_len = 0
    duplicates = 0
    seen_hashes: set[str] = set()
    detected_text = text_column

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            total_rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                keys.update(str(key) for key in row.keys())
                if not detected_text:
                    detected_text = next((col for col in COMMON_TEXT_COLUMNS if col in row), None)
                if detected_text:
                    text = str(row.get(detected_text) or "").strip()
                    if not text:
                        empty_text += 1
                    total_text_len += len(text)
                    digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
                    if text and digest in seen_hashes:
                        duplicates += 1
                    elif text:
                        seen_hashes.add(digest)

    columns = sorted(keys)
    time_columns = [col for col in columns if col in COMMON_TIME_COLUMNS]
    covariate_columns = [col for col in columns if col.startswith("cov_")]
    return {
        "path": str(path),
        "format": "jsonl",
        "columns": columns,
        "row_count": total_rows,
        "requested_text_column": text_column,
        "detected_text_column": detected_text if detected_text in keys else None,
        "text_column_exists": bool(detected_text and detected_text in keys),
        "empty_text_count": empty_text if detected_text else None,
        "duplicate_text_count": duplicates if detected_text else None,
        "average_text_length": round(total_text_len / total_rows, 2) if detected_text and total_rows else None,
        "time_columns": time_columns,
        "covariate_candidates": covariate_columns,
    }


def inspect_dataset(dataset: dict[str, Any], text_column: str | None) -> dict[str, Any] | None:
    selected = dataset.get("selected_file")
    if not selected:
        return None
    path = Path(selected)
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        return inspect_csv(path, text_column)
    if suffix == ".jsonl":
        return inspect_jsonl(path, text_column)
    return {
        "path": str(path),
        "format": suffix.lstrip("."),
        "parsed": False,
        "note": "File exists, but this read-only helper only parses CSV, TSV, and JSONL without optional dependencies.",
    }


def inspect_repo(repo: Path) -> dict[str, Any]:
    return {
        "root": str(repo),
        "required_paths": {path: (repo / path).exists() for path in REQUIRED_REPO_PATHS},
        "scripts": {path: (repo / path).exists() for path in SCRIPT_PATHS},
    }


def inspect_runtime(repo: Path) -> dict[str, Any]:
    conda_info = run_command(["conda", "env", "list"])
    git_status = run_command(["git", "status", "--short", "--branch"], cwd=repo)
    nvidia = run_command(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "conda_available": conda_info.get("available", False),
        "conda_envs": conda_info.get("output"),
        "git_status": git_status.get("output"),
        "nvidia_smi_available": nvidia.get("available", False),
        "nvidia_smi": nvidia.get("output"),
    }


def embedding_guidance(mode: str | None, env_info: dict[str, Any]) -> dict[str, Any]:
    keys = env_info.get("keys", {})
    configured = {key: value for key, value in keys.items() if value.get("configured")}
    provider_configured = "EMBEDDING_PROVIDER" in configured
    cloud_keys = [key for key in configured if key.endswith("API_KEY") or "API_KEY" in key]
    local_keys = [key for key in configured if key.startswith("QWEN_MODEL") or key == "SBERT_MODEL_PATH"]

    if mode == "zero_shot":
        rule = "Cloud embedding is allowed for zero_shot after explicit confirmation."
    elif mode in {"supervised", "unsupervised"}:
        rule = "Local embeddings are required for supervised/unsupervised finetune-capable workflows."
    else:
        rule = "Choose mode before selecting embedding provider; only zero_shot may use cloud embedding."

    return {
        "mode": mode,
        "rule": rule,
        "embedding_provider_configured": provider_configured,
        "configured_cloud_secret_keys": cloud_keys,
        "configured_local_model_keys": local_keys,
        "note": "Secret values are intentionally omitted.",
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    repo = find_repo_root(args.repo)
    dataset = resolve_dataset(args.dataset, repo)
    env_info = summarize_env(repo)
    dataset_info = inspect_dataset(dataset, args.text_column)
    return {
        "repo": inspect_repo(repo),
        "runtime": inspect_runtime(repo),
        "env": env_info,
        "embedding": embedding_guidance(args.mode, env_info),
        "dataset": dataset,
        "dataset_profile": dataset_info,
        "safety": {
            "read_only": True,
            "writes_files": False,
            "installs_dependencies": False,
            "calls_external_apis": False,
            "starts_training": False,
            "prints_secret_values": False,
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# THETA Preflight Report")
    lines.append("")
    lines.append("## Repository")
    lines.append(f"- Root: `{report['repo']['root']}`")
    for path, exists in report["repo"]["required_paths"].items():
        lines.append(f"- {'OK' if exists else 'MISSING'} `{path}`")
    lines.append("")
    lines.append("## Runtime")
    runtime = report["runtime"]
    lines.append(f"- Python: `{runtime['python']}`")
    lines.append(f"- Executable: `{runtime['python_executable']}`")
    lines.append(f"- Conda available: `{runtime['conda_available']}`")
    lines.append(f"- NVIDIA GPU detected: `{runtime['nvidia_smi_available']}`")
    if runtime.get("git_status"):
        lines.append("- Git status:")
        lines.append("```text")
        lines.append(runtime["git_status"])
        lines.append("```")
    lines.append("")
    lines.append("## Environment")
    env = report["env"]
    lines.append(f"- `.env.example`: `{env['env_example_exists']}`")
    lines.append(f"- `.env`: `{env['env_exists']}`")
    configured = [key for key, value in env["keys"].items() if value["configured"]]
    lines.append(f"- Configured keys: `{len(configured)}` (values omitted)")
    for key in configured:
        suffix = " (secret value omitted)" if env["keys"][key]["sensitive"] else ""
        lines.append(f"  - `{key}`{suffix}")
    lines.append("")
    lines.append("## Embedding")
    embedding = report["embedding"]
    lines.append(f"- Mode: `{embedding['mode']}`")
    lines.append(f"- Rule: {embedding['rule']}")
    lines.append(f"- Cloud secret keys configured: `{len(embedding['configured_cloud_secret_keys'])}`")
    lines.append(f"- Local model keys configured: `{len(embedding['configured_local_model_keys'])}`")
    lines.append("")
    lines.append("## Dataset")
    dataset = report["dataset"]
    if not dataset["provided"]:
        lines.append("- No dataset path provided.")
    else:
        lines.append(f"- Path: `{dataset['path']}`")
        lines.append(f"- Exists: `{dataset['exists']}`")
        if dataset.get("selected_file"):
            lines.append(f"- Selected file: `{dataset['selected_file']}`")
    profile = report.get("dataset_profile")
    if profile:
        lines.append(f"- Format: `{profile.get('format')}`")
        if profile.get("parsed", True) is False:
            lines.append(f"- Note: {profile.get('note')}")
        else:
            lines.append(f"- Rows: `{profile.get('row_count')}`")
            lines.append(f"- Columns: `{', '.join(profile.get('columns', []))}`")
            lines.append(f"- Text column: `{profile.get('detected_text_column')}`")
            lines.append(f"- Empty text: `{profile.get('empty_text_count')}`")
            lines.append(f"- Duplicate text: `{profile.get('duplicate_text_count')}`")
            lines.append(f"- Average text length: `{profile.get('average_text_length')}`")
            lines.append(f"- Time columns: `{', '.join(profile.get('time_columns', []))}`")
            lines.append(f"- Covariate candidates: `{', '.join(profile.get('covariate_candidates', []))}`")
    lines.append("")
    lines.append("## Safety")
    for key, value in report["safety"].items():
        lines.append(f"- {key}: `{value}`")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    report = build_report(args)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
