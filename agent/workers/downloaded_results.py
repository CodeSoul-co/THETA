"""Register and read verified deliveries from a remote training worker.

The local capability service owns terminal compute records. Receiving a delivery
never starts training or executes code supplied by the remote service.
"""
import json
import re
import time
from pathlib import Path
from .results_reader import file_hash, tree_hash


def receive(payload):
    from .local_compute import database
    request, report = payload['request'], payload['report']
    job_id = request['jobId']
    if not re.fullmatch(r'job-[a-f0-9]{64}', job_id):
        raise ValueError('无效任务标识')
    root = Path(report['resultDir']).resolve()
    parent = Path(payload['home']).resolve() / 'compute' / job_id
    root.relative_to(parent)
    for item in report['files']:
        file = Path(item['path'])
        file.resolve().relative_to(root)
        if file.is_symlink() or file.stat().st_size != item['sizeBytes'] or file_hash(file) != item['sha256']:
            raise ValueError('远程结果文件校验失败：' + item['name'])
    report['resultHash'] = tree_hash(root)
    report['manifestPath'] = str(parent / 'delivery.json')
    Path(report['manifestPath']).write_text(json.dumps(report, ensure_ascii=False), encoding='utf-8')
    job = dict(id=job_id, status='completed', phase='completed', percent=100,
               resultDir=str(root), resultHash=report['resultHash'], remoteReport=report['manifestPath'])
    with database(payload['home']) as db:
        db.execute('INSERT INTO jobs(id,request,state,value,updated) VALUES(?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET request=excluded.request,state=excluded.state,value=excluded.value,updated=excluded.updated',
                   (job_id, json.dumps(request), 'completed', json.dumps(job), time.time()))
    return report


def read(job, payload):
    root = Path(job['resultDir']).resolve()
    parent = Path(payload['home']).resolve() / 'compute' / job['id']
    root.relative_to(parent)
    manifest = Path(job['remoteReport']).resolve()
    manifest.relative_to(parent)
    report = json.loads(manifest.read_text(encoding='utf-8'))
    if tree_hash(root) != job['resultHash']:
        raise ValueError('下载的训练结果已变化，不能读取未经校验的结果')
    if payload.get('view') == 'report':
        return report
    evidence = report.get('evidence', {})
    common = dict(trainingRunId=job['id'], jobId=job['id'], resultDir=str(root),
                  reportPath=report['reportPath'], reportStatus='complete', remoteDelivery=True,
                  missingEvidence=report.get('missingEvidence', []), availableArtifacts=report['files'])
    evidence.update(common)
    view, offset = payload.get('view', 'summary'), payload.get('offset') or 0
    if not isinstance(offset, int) or offset < 0:
        raise ValueError('无效结果分页位置')
    if view == 'artifacts':
        return dict(common, files=report['files'], sha256=job['resultHash'])
    key = dict(tables='tables', figures='figures', distribution='matrices', metrics='evidence').get(view)
    if key:
        values = evidence.get(key, [])
        size = 2 if view in ('tables', 'metrics') else 10
        return dict(common, **{key: values[offset:offset+size]}, itemCount=len(values),
                    nextOffset=offset+size if offset+size < len(values) else None)
    if view != 'analysis':
        evidence['summaryCounts'] = {k: len(evidence.get(k, [])) for k in ('evidence', 'tables', 'figures', 'matrices')}
        for key, size in [('evidence', 3), ('tables', 1), ('figures', 5)]:
            evidence[key] = evidence.get(key, [])[:size]
    return evidence
