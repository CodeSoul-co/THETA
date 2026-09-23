"""Build a bounded Agent report from a verified distributed Worker bundle."""
from __future__ import annotations

import html
import json
from pathlib import Path
from uuid import uuid4

from .capabilities import verify_dataset
from .dataset.business import excerpt
from .dataset.readers import load_dataset
from .results_reader import file_hash, read_result_evidence, tree_hash
from .runtime_environments import identity


def _inside(root: Path, value: str, *, directory: bool = False) -> Path:
    root = root.resolve(strict=True)
    path = Path(value)
    if path.is_symlink():
        raise ValueError("Remote result path cannot be a symbolic link")
    path = path.resolve(strict=True)
    path.relative_to(root)
    if directory and not path.is_dir():
        raise ValueError("Remote result directory is missing")
    if not directory and not path.is_file():
        raise ValueError("Remote result file is missing")
    return path


def _matrix(root: Path, row_count: int):
    import numpy as np

    candidates = []
    for file in sorted(root.rglob("theta*.npy")):
        try:
            value = np.load(file, allow_pickle=False, mmap_mode="r")
            if value.ndim == 2 and value.shape[0] == row_count and value.shape[1] > 0:
                candidates.append((file, value))
        except (OSError, ValueError):
            continue
    if not candidates:
        return None
    exact = [item for item in candidates if item[0].name == "theta.npy"]
    return (exact or candidates)[0][1]


def _source_data(result_root: Path, workspace: Path, normalized: Path, dataset: dict, plan: dict):
    import numpy as np

    source = verify_dataset(dataset)
    original = load_dataset(source, profile_limit=1_000_000)
    prepared = load_dataset(normalized, profile_limit=1_000_000)
    text_column = plan["textColumn"]
    if original.rows_truncated or prepared.rows_truncated:
        raise ValueError("Source row alignment exceeds the full-data reader limit")
    original_matches = (len(original.rows) == len(prepared.rows)
                        and all(str(a.get(text_column) or "") == str(b.get("text") or "")
                                for a, b in zip(original.rows, prepared.rows)))
    metadata_roots = [result_root, workspace]
    source_rows = next((file for root in metadata_roots for file in root.rglob("source_rows.npy")), None)
    texts_file = next((file for root in metadata_roots for file in root.rglob("texts.json")), None)
    if source_rows and texts_file:
        indices = np.load(source_rows, allow_pickle=False)
        texts = json.loads(texts_file.read_text(encoding="utf-8"))
        indices_valid = (indices.ndim == 1 and np.issubdtype(indices.dtype, np.integer)
                         and len(indices) == len(texts) and len(set(indices.tolist())) == len(indices)
                         and (indices >= 0).all() and (indices < len(prepared.rows)).all())
        exact_texts = indices_valid and all(
            str(prepared.rows[int(index)].get("text") or "") == str(text)
            for index, text in zip(indices, texts)
        )
    else:
        indices = np.arange(len(prepared.rows))
        indices_valid = True
        exact_texts = True
    theta = _matrix(result_root, len(indices))
    aligned = bool(original_matches and indices_valid and exact_texts and theta is not None)
    columns = list(dict.fromkeys([text_column, *plan.get("covariates", []),
        *[name for name in [plan.get("timeColumn"), plan.get("labelColumn"), "timestamp", "year", "date", "日期", "channel"]
          if name and name in original.columns]]))
    rows = []
    for matrix_row, index in enumerate(indices[:40] if aligned else np.arange(min(40, len(original.rows)))):
        row = original.rows[int(index)]
        item = {"sourceRow": int(index) + 1,
                **{name: excerpt(row.get(name), 500) for name in columns}}
        if aligned:
            key = "latentCoordinates" if plan.get("modelId") == "nvdm" else "membershipWeights" if plan.get("modelId") == "bertopic" else "topicWeights"
            item[key] = theta[matrix_row].tolist()
        rows.append(item)
    return {
        "datasetRef": dataset["datasetRef"], "datasetHash": dataset["sha256"],
        "textColumn": text_column, "rowCount": original.row_count,
        "matrixRowsAligned": aligned,
        "excludedSourceRows": len(original.rows) - len(indices) if aligned else None,
        "coverage": "all model rows" if aligned and len(indices) <= 40 else "first 40 rows, not representative",
        "rows": rows,
    }


def generate(payload: dict) -> dict:
    home = Path(payload["home"]).resolve(strict=True)
    job_id = payload["jobId"]
    bundle = _inside(home / "remote-results" / job_id, payload["bundleRoot"], directory=True)
    result_root = _inside(bundle, str(bundle / "result"), directory=True)
    workspace = _inside(bundle, str(bundle / "workspace"), directory=True)
    normalized = _inside(bundle, str(bundle / "input" / "data.csv"))
    if file_hash(normalized) != payload["normalizedInputSha256"]:
        raise ValueError("Normalized input hash does not match Worker integrity metadata")
    result_hash = tree_hash(result_root)
    evidence = read_result_evidence({
        "trainingRunId": job_id, "modelId": payload["plan"]["modelId"],
        "artifacts": [{"kind": "results", "path": str(result_root),
                       "sha256": result_hash, "artifactId": job_id + ":results"}],
    })
    missing = []
    matrix_names = [file.name for file in result_root.rglob("*.npy") if file.is_file() and not file.is_symlink()]
    complete = any(name.startswith("theta") for name in matrix_names) and any(name.startswith("beta") for name in matrix_names)
    if not complete:
        missing.append("Result bundle does not contain both verified theta and beta matrices; deep interpretation is incomplete.")
    try:
        source_data = _source_data(result_root, workspace, normalized, payload["dataset"], payload["plan"])
        if not source_data["matrixRowsAligned"]:
            missing.append("Original rows are not proven aligned to theta; no row/topic association may be inferred.")
    except Exception as error:
        source_data = None
        missing.append(f"原始文本行校验失败：{error}")

    destination = home / "reports" / job_id / ("remote-report-" + uuid4().hex)
    destination.mkdir(parents=True, exist_ok=False)
    files = []
    for file in sorted(result_root.rglob("*")):
        if not file.is_file() or file.is_symlink():
            continue
        suffix = file.suffix.lower()
        kind = "figure" if suffix in {".png", ".svg", ".jpg", ".jpeg", ".pdf", ".html"} else "table" if suffix in {".csv", ".tsv"} else "matrix" if suffix == ".npy" else "artifact"
        files.append({"name": "training/" + file.relative_to(result_root).as_posix(),
                      "path": str(file.resolve()), "kind": kind,
                      "sha256": file_hash(file), "sizeBytes": file.stat().st_size})
    worker_log = bundle / "worker.log"
    if worker_log.is_file() and not worker_log.is_symlink():
        files.append({"name": "worker.log", "path": str(worker_log.resolve()), "kind": "artifact",
                      "sha256": file_hash(worker_log), "sizeBytes": worker_log.stat().st_size})
    body = ["<!doctype html><html lang=\"zh\"><meta charset=\"utf-8\"><title>THETA 分布式结果报告</title>",
            "<style>body{font:16px/1.65 system-ui;max-width:1100px;margin:40px auto;padding:0 24px}img{max-width:100%}</style>",
            f"<h1>{html.escape(payload['plan']['modelId'].upper())} · 分布式训练结果</h1>",
            "<p>结果包已按 Worker 返回的 SHA-256 与长度校验；以下内容未重新训练。</p>",
            f"<p>任务：{html.escape(job_id)}</p><h2>可视化、表格与模型产物</h2>"]
    for item in files:
        uri = Path(item["path"]).as_uri()
        body.append(f'<p><a href="{html.escape(uri, quote=True)}">{html.escape(item["name"])}</a></p>')
        if item["kind"] == "figure" and Path(item["path"]).suffix.lower() in {".png", ".jpg", ".jpeg", ".svg"}:
            body.append(f'<img loading="lazy" src="{html.escape(uri, quote=True)}" alt="{html.escape(item["name"], quote=True)}">')
    if missing:
        body.append("<h2>限制</h2><ul>" + "".join(f"<li>{html.escape(item)}</li>" for item in missing) + "</ul>")
    page = destination / "index.html"
    page.write_text("\n".join(body) + "</html>", encoding="utf-8")
    files.append({"name": "index.html", "path": str(page.resolve()), "kind": "report",
                  "sha256": file_hash(page), "sizeBytes": page.stat().st_size})
    manifest = destination / "manifest.json"
    report = {
        "schemaVersion": "theta.result-report.v2", "jobId": job_id,
        "modelId": payload["plan"]["modelId"], "resultHash": result_hash,
        "inputSignatures": {str(normalized): file_hash(normalized)}, "files": files,
        "runtime": identity("reports"), "trainingPlan": payload["plan"],
        "reportStatus": "complete" if complete else "incomplete", "resultDir": str(result_root),
        "logPath": str(worker_log.resolve()) if worker_log.is_file() else None,
        "trainingLogPath": str(worker_log.resolve()) if worker_log.is_file() else None,
        "diagnostics": None, "evidence": evidence, "sourceData": source_data,
        "missingEvidence": missing, "nativeRunner": "distributed-worker-existing-artifacts",
        "reportPath": str(page.resolve()), "manifestPath": str(manifest.resolve()),
        "quality": {"source": "verified distributed Worker artifact bundle",
                    "rowAlignment": "verified" if source_data and source_data["matrixRowsAligned"] else "unverified"},
        "summary": {"evidenceFiles": len(evidence.get("evidence", [])),
                    "figures": len(evidence.get("figures", [])), "tables": len(evidence.get("tables", []))},
    }
    manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report
