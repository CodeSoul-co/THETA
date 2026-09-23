"""Local ComputeGateway runtime, reusing trainning.worker without editing the engine.

One SQLite claim per immutable job; no automatic re-execution of ambiguous jobs.
The distributed gateway uses Go task state instead of this local job store.
"""
from __future__ import annotations

import csv
from contextlib import contextmanager, closing
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from threading import Event, Thread
from datetime import datetime, timezone

from .capabilities import engine_root, verify_dataset, validate_plan, runtime_check
from .dataset.readers import load_dataset
from .results_reader import tree_hash, read_result_evidence, file_hash
from .execution_policy import assert_authorized, training_environment
from .diagnostics import failure_diagnostics
from .job_observation import observe


@contextmanager
def database(home: str):
    root = Path(home).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(root / "compute.sqlite", timeout=10)) as db, db:
        db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, request TEXT NOT NULL, state TEXT NOT NULL, value TEXT NOT NULL, cancel INTEGER NOT NULL DEFAULT 0, updated REAL NOT NULL)")
        yield db


def status(payload: dict) -> dict:
    with database(payload["home"]) as db:
        row = db.execute("SELECT value,state,updated FROM jobs WHERE id=?", (payload["jobId"],)).fetchone()
        if not row:
            raise ValueError("计算任务不存在")
        result = json.loads(row[0])
        if row[1] == 'failed':
            result['diagnostics'] = failure_diagnostics(payload['home'], payload['jobId'])
        if db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='external_usage'").fetchone():
            usage = db.execute('SELECT calls FROM external_usage WHERE job_id=?', (payload['jobId'],)).fetchone()
            if usage:
                result['externalRequestsUsed'] = usage[0]
        if result.get('status') == 'completed' and result.get('plan', {}).get('modelId') == 'bertopic' and result.get('resultDir'):
            result_root = Path(result['resultDir']).resolve()
            result_root.relative_to(Path(payload['home']).resolve() / 'compute' / result['id'])
            if not any(result_root.rglob('document_topics.npy')) or not any(result_root.rglob('result_manifest.json')):
                result['resultWarning'] = '历史任务记录为 completed，但 BERTopic 结果导出不完整；不能视为正常交付。已有文件保留，可申请诊断与结果整理。'
                result['diagnostics'] = failure_diagnostics(payload['home'], payload['jobId'])
        result["telemetry"] = observe(payload["home"], result, row[2])
        if row[1] in {"queued", "running"} and time.time() - row[2] > 30:
            result["error"] = "Worker 心跳已中断，任务状态不确定；不会自动重复启动训练。"
        return result


def submit(payload: dict) -> dict:
    job_id = payload["jobId"]
    if not job_id.startswith("job-") or len(job_id) != 68 or any(c not in "0123456789abcdef" for c in job_id[4:]):
        raise ValueError("无效任务标识")
    request = json.dumps(payload, sort_keys=True)
    with database(payload["home"]) as db:
        old = db.execute("SELECT request FROM jobs WHERE id=?", (job_id,)).fetchone()
        if old:
            previous = json.loads(old[0])
            if {key: value for key, value in previous.items() if key != 'authorization'} != {key: value for key, value in payload.items() if key != 'authorization'}:
                raise ValueError("相同请求 ID 不能绑定不同参数")
            return status(payload)
        assert_authorized(payload)
        validate_plan(payload)
        ready = runtime_check({"modelId": payload["plan"]["modelId"], "modelSize": payload["plan"]["params"].get("model_size"),
                               'embeddingProvider': payload['execution']['embedding']['mode'], 'mode': payload['plan']['params'].get('mode'),
                               'device': payload['plan'].get('device'), 'modelAssets': payload['execution'].get('modelAssets', {})})
        if not ready["ready"]:
            raise ValueError(" ".join(ready.get("issues", [])) or "训练依赖或本地模型资源未就绪，请检查模型设置和计算环境。")
        value = {"id": job_id, "status": "queued", "phase": "queued", "percent": 0, "createdAt": time.time(), "runtime": payload["execution"]["runtime"]}
        try:
            db.execute("INSERT INTO jobs (id,request,state,value,updated) VALUES (?,?,?,?,?)", (job_id, request, "queued", json.dumps(value), time.time()))
        except sqlite3.IntegrityError:
            return status(payload)
    try:
        root = Path(payload["home"]).resolve() / "compute" / job_id
        root.mkdir(parents=True, exist_ok=True)
        with (root / "host.log").open("ab") as log:
            subprocess.Popen([sys.executable, "-m", "workers", "compute.run", str(Path(payload["home"]).resolve()), job_id],
                             cwd=str(Path(__file__).resolve().parents[1]), stdin=subprocess.DEVNULL,
                             stdout=log, stderr=log, start_new_session=True, env=dict(os.environ))
    except Exception as exc:
        update(payload["home"], job_id, status="failed", phase="launch_failed", error=str(exc))
        raise
    return status(payload)


def update(home: str, job_id: str, **fields):
    with database(home) as db:
        row = db.execute("SELECT value FROM jobs WHERE id=?", (job_id,)).fetchone()
        value = json.loads(row[0])
        if fields.get('phase', value.get('phase')) != value.get('phase'):
            fields['phaseStartedAt'] = time.time()
            value['phaseHistory'] = [*value.get('phaseHistory', [])[-39:], {
                'phase': fields['phase'], 'at': fields['phaseStartedAt'],
            }]
        if fields.get('status') in {'completed', 'failed', 'cancelled'}:
            fields['finishedAt'] = time.time()
        value.update(fields)
        db.execute("UPDATE jobs SET state=?,value=?,updated=? WHERE id=?", (value["status"], json.dumps(value), time.time(), job_id))


def cancel(payload: dict) -> dict:
    with database(payload["home"]) as db:
        db.execute("UPDATE jobs SET cancel=1 WHERE id=? AND state IN ('queued','running')", (payload["jobId"],))
    return status(payload)


def cancelled(home: str, job_id: str) -> bool:
    with database(home) as db:
        return bool(db.execute("SELECT cancel FROM jobs WHERE id=?", (job_id,)).fetchone()[0])


def normalize_dataset(payload: dict, destination: Path) -> None:
    plan = payload["plan"]
    source = verify_dataset(payload["dataset"])
    table = load_dataset(source, profile_limit=1_000_000)
    if table.rows_truncated:
        raise ValueError("本地规范化最多支持一百万行；禁止用画像样本代替完整训练数据")
    mappings = {"text": plan["textColumn"]}
    if plan.get("labelColumn"):
        mappings["label"] = plan["labelColumn"]
    if plan.get("timeColumn"):
        mappings["year"] = plan["timeColumn"]
        mappings["timestamp"] = plan["timeColumn"]
    for index, column in enumerate(plan.get("covariates", [])):
        mappings[f"cov_{index}"] = column
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mappings))
        writer.writeheader()
        for row in table.rows:
            writer.writerow({key: row.get(column, "") for key, column in mappings.items()})
    verify_dataset(payload["dataset"])


def run(home: str, job_id: str) -> None:
    with database(home) as db:
        claimed = db.execute("UPDATE jobs SET state='running',updated=? WHERE id=? AND state='queued'", (time.time(), job_id)).rowcount
        if not claimed:
            return
        payload = json.loads(db.execute("SELECT request FROM jobs WHERE id=?", (job_id,)).fetchone()[0])
    update(home, job_id, status="running", phase="preparing", pid=os.getpid(), startedAt=time.time())
    stop_heartbeat = Event()
    def heartbeat():
        while not stop_heartbeat.wait(2):
            try:
                with database(home) as db:
                    db.execute("UPDATE jobs SET updated=? WHERE id=? AND state='running'", (time.time(), job_id))
            except sqlite3.Error:
                # A failed heartbeat is observable; do not claim liveness on a failed write.
                pass
    heartbeat_thread = Thread(target=heartbeat, daemon=True)
    heartbeat_thread.start()
    try:
        assert_authorized(payload)
        sys.path.insert(0, str(engine_root() / "trainning"))
        from dataclasses import replace
        from worker.config import WorkerConfig
        from .pipeline_adapter import AgentThetaPipeline
        from worker.protocol import ExecutionSpec
        from worker.storage import FilesystemObjectStorage
        from worker.errors import JobCancelled
        from worker.process import ProcessRunner

        plan = payload["plan"]
        root = Path(home).resolve() / "compute" / job_id
        objects = root / "objects"
        normalized = objects / "dataset/data.csv"
        if cancelled(home, job_id):
            raise JobCancelled("用户已取消训练")
        normalize_dataset(payload, normalized)
        params = {**plan["params"]}
        if plan["modelId"] == "theta":
            params.setdefault("embedding_provider", "local")
        gpu = plan.get('device', 'cpu').startswith('cuda:')
        spec = ExecutionSpec.from_dict({
            "schema_version": 2, "type": "job.ready", "event_id": job_id,
            "task_id": int(job_id[4:16], 16) + 1, "attempt": 1, "task_version": 1,
            "user_id": "local-agent", "priority": 5,
            "dataset": {"ref": payload["dataset"]["datasetRef"], "project_id": payload["runId"].removeprefix("run-"),
                        "object_key": "dataset/data.csv", "filename": "data.csv", "format": "csv", "sha256": file_hash(normalized)},
            "model": {"id": 1, "name": plan["modelId"], "framework": "theta"},
            "runtime": {"id": 1, "key": payload["execution"]["runtime"]["profile"], "version": payload["execution"]["runtime"]["revision"], "executor": "theta_pipeline", "image": "local"},
            "params": params,
            "resources": {"accelerator": "cuda" if gpu else "none", "gpu_count": 1 if gpu else 0, "gpu_memory_mb": 0,
                          "cpu_cores": 2, "memory_mb": 4096, "timeout_seconds": plan["timeoutSeconds"]},
            "output_prefix": job_id + "/", "created_at": datetime.now(timezone.utc).isoformat(),
        })
        config = replace(WorkerConfig.load(), project_root=engine_root(), job_root=root / "jobs",
                         python_executable=sys.executable, resource_class="gpu" if plan.get("device", "cpu").startswith("cuda:") else "cpu", gpu_id=int(plan["device"].split(":")[1]) if plan.get("device", "cpu").startswith("cuda:") else None,
                         keep_job_dir=True, heartbeat_interval_seconds=2)
        environment = training_environment(payload['execution'])
        environment.update({'THETA_PROJECT_ROOT': str(engine_root()), 'THETA_COMPUTE_DATABASE': str(Path(home).resolve() / 'compute.sqlite'), 'THETA_COMPUTE_JOB_ID': job_id})

        class ApprovedProcessRunner(ProcessRunner):
            def run(self, **kwargs):
                command = list(kwargs['command'])
                kwargs['command'] = [command[0], str(Path(__file__).with_name('engine_entry.py')), *command[1:]]
                # Retain per-job output directories from the shared pipeline, but replace ambient credentials.
                child_env = dict(kwargs['env'])
                for key in list(child_env):
                    if key not in environment:
                        child_env.pop(key)
                child_env.update(environment)
                for key in ['PROJECT_ROOT', 'DATA_DIR', 'WORKSPACE_DIR', 'RESULT_DIR', 'PYTHONUNBUFFERED', 'PYTHONUTF8', 'PYTHONIOENCODING']:
                    if key in kwargs['env']:
                        child_env[key] = kwargs['env'][key]
                kwargs['env'] = child_env
                return super().run(**kwargs)

        pipeline = AgentThetaPipeline(config, FilesystemObjectStorage(objects), ApprovedProcessRunner())
        pipeline.plan = plan
        result = pipeline.execute(spec, lambda: cancelled(home, job_id),
                                  lambda phase, percent, message: update(home, job_id, phase=phase, percent=percent, message=message))
        if cancelled(home, job_id):
            raise JobCancelled("用户已取消训练")
        digest = tree_hash(result.result_dir)
        update(home, job_id, status="completed", phase="completed", percent=100,
               resultDir=str(result.result_dir), resultHash=digest,
               sourceHash=payload["dataset"]["sha256"], preparedHash=spec.dataset.sha256,
               plan=plan, elapsedSeconds=result.elapsed_seconds)
    except Exception as exc:
        is_cancel = cancelled(home, job_id)
        reason = str(exc)[-3000:]
        with database(home) as db:
            if db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='external_denials'").fetchone():
                denial = db.execute('SELECT reason FROM external_denials WHERE job_id=?', (job_id,)).fetchone()
                if denial:
                    reason = denial[0]
        update(home, job_id, status="cancelled" if is_cancel else "failed", phase="cancelled" if is_cancel else "failed", error=reason)
    finally:
        stop_heartbeat.set()
        heartbeat_thread.join(timeout=3)


def results(payload: dict) -> dict:
    job = status(payload)
    if job["status"] != "completed":
        raise ValueError("任务尚未成功完成，不能读取成功结果")
    root = Path(job["resultDir"])
    expected = Path(payload["home"]).resolve() / "compute" / job["id"]
    root.resolve().relative_to(expected)
    if tree_hash(root) != job["resultHash"]:
        raise ValueError("Result artifact changed after verification")
    artifact = {"kind": "results", "path": str(root), "sha256": job["resultHash"], "artifactId": job["id"] + ":results"}
    if payload.get('view') == 'report':
        from .result_report import generate_report
        with database(payload['home']) as db:
            request = json.loads(db.execute('SELECT request FROM jobs WHERE id=?', (job['id'],)).fetchone()[0])
        workspaces = list((expected / 'jobs').glob('*/workspace'))
        if len(workspaces) != 1:
            raise ValueError('无法唯一定位本任务的预处理工作区，不能猜测词表或重训')
        return generate_report(root, job, Path(payload['home']).resolve() / 'reports', workspace=workspaces[0],
                               dataset=request['dataset'], prepared=expected / 'objects/dataset/data.csv', worker_log=workspaces[0].parent / 'worker.log')
    evidence = None
    # Follow-up reads must use the same corrected native presentation the user received,
    # rather than silently reverting to obsolete plots in the immutable training tree.
    manifests = (Path(payload['home']).resolve() / 'reports' / job['id']).glob('report-*/manifest.json')
    for manifest in sorted(manifests, key=lambda path: path.stat().st_mtime, reverse=True):
        report = json.loads(manifest.read_text())
        if (report.get('schemaVersion') == 'theta.result-report.v2' and report.get('resultHash') == job['resultHash']
                and all(Path(item['path']).is_file() and file_hash(Path(item['path'])) == item['sha256'] for item in report['files'])):
            evidence = report['evidence']
            evidence['reportPath'] = report['reportPath']
            evidence['missingEvidence'] = report['missingEvidence']
            evidence['reportStatus'] = report.get('reportStatus')
            evidence['resultDir'] = str(root)
            evidence['diagnostics'] = job.get('diagnostics')
            evidence['availableArtifacts'] = [{key: item[key] for key in ['name', 'kind', 'path']} for item in report['files']]
            break
    if evidence is None:
        evidence = read_result_evidence({"trainingRunId": job["id"], "modelId": job.get("plan", {}).get("modelId"), "artifacts": [artifact]})
    if payload.get("view") == "artifacts":
        return {"jobId": job["id"], "sha256": job["resultHash"], "resultDir": str(root),
                "reportPath": evidence.get('reportPath'), "reportStatus": evidence.get('reportStatus'),
                "diagnostics": job.get('diagnostics'), "missingEvidence": evidence.get('missingEvidence', []),
                "availableArtifacts": evidence.get('availableArtifacts', []), "files": [
                    {"name": p.relative_to(root).as_posix(), "path": str(p.resolve()), "sizeBytes": p.stat().st_size, "sha256": file_hash(p)}
                    for p in sorted(root.rglob("*")) if p.is_file()][:100],
                "instruction": "这是已授权结果的现有文件清单与诊断摘要，文件存在不等于可用于解读。直接说明现有证据，不需要为了查看日志摘要再申请 report 生成授权。"}
    view = payload.get('view', 'summary')
    offset = payload.get('offset') or 0
    if not isinstance(offset, int) or offset < 0:
        raise ValueError('无效结果分页位置')
    keys = {'tables': ['tables'], 'figures': ['figures'], 'distribution': ['matrices'], 'metrics': ['evidence']}.get(view)
    if keys:
        evidence = {key: value for key, value in evidence.items() if key in [*keys, 'trainingRunId', 'limitations', 'skipped', 'analysisSkipped', 'reportStatus', 'reportPath', 'resultDir', 'missingEvidence']}
        key = keys[0]
        all_items = evidence[key]
        page_size = 2 if view in {'tables', 'metrics'} else 10
        evidence[key] = all_items[offset:offset + page_size]
        evidence['nextOffset'] = offset + page_size if offset + page_size < len(all_items) else None
        evidence['itemCount'] = len(all_items)
    else:
        evidence['summaryCounts'] = {key: len(evidence[key]) for key in ['evidence', 'tables', 'figures', 'matrices']}
        for key, limit in [('evidence', 3), ('tables', 1), ('figures', 5)]:
            evidence[key] = evidence[key][:limit]
        evidence['instruction'] = '概览有界；用 metrics/tables/figures/distribution 分页读取剩余证据，再结合业务目标解读。缺失产物须明确说明，不自动重训或生成新图。'
    return {**evidence, "plan": job.get("plan"), "datasetHash": job.get("sourceHash"), "preparedHash": job.get("preparedHash")}
