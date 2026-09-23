"""Version-bound plans, one-shot host approvals, auditable results and paper exports."""
from __future__ import annotations
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import sqlite3
import time
import warnings
import numpy as np
import pandas as pd
from ..capabilities import verify_dataset
from ..runtime_environments import identity
from .registry import METHODS
from .validation import validate
from .report import clean,write


def load(dataset):
    file=verify_dataset(dataset)
    if file.stat().st_size>50*1024*1024: raise ValueError('Statistics worker limit: 50 MiB input')
    ext=file.suffix.lower()
    if ext in {'.csv','.tsv'}: frame=pd.read_csv(file,sep='\t' if ext=='.tsv' else ',',nrows=20001)
    elif ext=='.xlsx': frame=pd.read_excel(file,nrows=20001)
    elif ext=='.json': frame=pd.read_json(file)
    elif ext=='.jsonl': frame=pd.read_json(file,lines=True,nrows=20001)
    else: raise ValueError('Statistics worker accepts CSV, TSV, XLSX, JSON and JSONL; import/convert other formats explicitly')
    if not 2<=len(frame)<=20000 or len(frame.columns)>200: raise ValueError('Statistics input must contain 2..20000 rows and <=200 columns; no silent sampling')
    if frame.columns.duplicated().any(): raise ValueError('Duplicate columns are ambiguous')
    frame.columns=frame.columns.map(str)
    return frame.replace([np.inf,-np.inf],np.nan)


def data_for_plan(dataset,plan):
    if dataset is not None:return load(dataset)
    for step in plan.get('steps',[]):
        method=METHODS.get(step.get('method'),{})
        if method.get('family')!='optimization' and step.get('method')!='power_ttest':raise ValueError('This method requires an attached dataset')
        if any(key in step for key in ('x','y','factors','group','time','entity','event','weight','endogenous','instruments')):raise ValueError('Data-free mathematics cannot refer to observation columns')
    return pd.DataFrame()


def prepare(data,spec):
    method=validate(spec,list(data.columns)); family=method['family']
    cols=list(dict.fromkeys(spec.get('x',[])+spec.get('factors',[])+spec.get('endogenous',[])+spec.get('instruments',[])+[spec[k] for k in ('y','group','time','event','entity','weight') if k in spec]+([spec['params']['text_column']] if spec.get('params',{}).get('text_column') else [])))
    chosen=data[cols].copy()
    missing={str(c):int(chosen[c].isna().sum()) for c in cols}
    if spec.get('missing','drop')=='error' and chosen.isna().any().any(): raise ValueError('Missing/nonfinite values present and missing=error')
    if family=='timeseries' and chosen.isna().any().any(): raise ValueError('Time-series gaps require an explicit justified transformation; missing observations cannot be silently deleted')
    if family=='ml': subset=[spec['y']]+[spec[k] for k in ('group','time') if k in spec]; chosen=chosen.dropna(subset=subset)
    elif spec['method']=='impute_median' or spec['method']=='frequency': pass
    else: chosen=chosen.dropna()
    if family!='optimization' and spec['method']!='power_ttest' and len(chosen)<2: raise ValueError('Fewer than two eligible observations')
    return chosen,{'input_N':len(data),'eligible_N':len(chosen),'excluded_N':len(data)-len(chosen),'missing_by_column':missing,'policy':spec.get('missing','drop'),'source_row_definition':'1-based data record + 1 header; XLSX first sheet; not physical line number for multiline CSV cells'}


def preview(payload):
    dataset=payload.get('dataset'); plan=payload['plan']
    if not isinstance(plan,dict) or set(plan)-{'question','hypotheses','assumptions','steps','sensitivity','stoppingRule'}: raise ValueError('Invalid empirical plan fields')
    if not isinstance(plan.get('question'),str) or not 1<=len(plan['question'])<=2000: raise ValueError('Research question required')
    if not isinstance(plan.get('steps'),list) or not 1<=len(plan['steps'])<=6: raise ValueError('An approval covers 1..6 analyses')
    data=data_for_plan(dataset,plan)
    checks=[]
    for spec in plan['steps']:
        chosen,sample=prepare(data,spec); checks.append({'method':spec['method'],'name':METHODS[spec['method']]['name'],'sample':sample})
    runtime=identity('statistics')
    source=hashlib.sha256()
    for file in sorted(Path(__file__).parent.glob('*.py')):
        if file.name.startswith('test_'): continue
        source.update(file.name.encode()); source.update(file.read_bytes())
    runtime['sourceFingerprint']=source.hexdigest()
    # Core dependency availability is distinct from the method catalogue.
    dependencies={'numpy','pandas','scipy','statsmodels','scikit-learn','matplotlib','tabulate'}
    for spec in plan['steps']:
        family=METHODS[spec['method']]['family']
        if family=='survival': dependencies.add('lifelines')
        if spec['method'].startswith(('panel_','iv_')): dependencies.add('linearmodels')
        for prefix,package in [('xgboost','xgboost'),('lightgbm','lightgbm'),('catboost','catboost'),('factor','factor-analyzer'),('kmo_','factor-analyzer'),('sem','semopy'),('apriori','mlxtend'),('garch','arch'),('dunn','scikit-posthocs')]:
            if spec['method'].startswith(prefix): dependencies.add(package)
    for package in sorted(dependencies):
        metadata.version(package)
    return {'dataset':dataset,'plan':plan,'runtime':runtime,'checks':checks,'limits':{'seconds':120,'maxRows':20000,'maxSteps':6,'network':'No network calls or model downloads in registered implementations','execution':'registered functions only; Python venv is not an OS sandbox'}}


def analyze(data,spec):
    from . import inference,advanced,prediction,decision,design
    selected,sample=prepare(data,spec); m=METHODS[spec['method']]; family=m['family']
    dispatch={'design':design.run,'descriptive':inference.descriptive,'test':inference.tests,'group_test':inference.tests,'contingency':inference.contingency,'regression':inference.regression,'econometrics':advanced.econometrics,'survival':advanced.survival,'unsupervised':advanced.unsupervised,'timeseries':advanced.timeseries,'ml':prediction.run,'decision':decision.run,'preprocess':decision.preprocess}
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        result=decision.programming(spec) if family=='optimization' else dispatch[family](selected,spec)
    messages=list(dict.fromkeys(str(w.message) for w in captured))
    if any('did not converge' in text.lower() or 'failed to converge' in text.lower() or 'perfect separation' in text.lower() for text in messages): raise ValueError('Estimator convergence/identification failure: '+'; '.join(messages)[:1500])
    result.update(method=m['id'],name=spec.get('name',m['name']),spec=spec,sample=sample)
    result.setdefault('metrics',{}).setdefault('N',len(selected))
    result['warnings']=result.get('warnings',[])+messages+([m['limitation']] if m['limitation'] else [])
    if sample['excluded_N']: result['warnings'].append(f'{sample["excluded_N"]} rows excluded under the declared missing-data rule; selection bias remains possible.')
    result=clean(result)
    if any(row.get('coefficient') is None or row.get('std_error') is None or row.get('p_value') is None for row in result.get('coefficients',[])): raise ValueError('Nonfinite inferential result; refusing to publish a coefficient table')
    return result


def execute(payload):
    home=Path(payload['home']).resolve(); receipt=payload.get('authorization',{}); expected={k:v for k,v in payload.items() if k not in {'home','authorization'}}
    db=sqlite3.connect(home/'research.sqlite')
    try:
        row=db.execute('SELECT value FROM records WHERE kind=? AND id=?',('effect-approval',receipt.get('id',''))).fetchone()
        approval=json.loads(row[0]) if row else {}
        if approval.get('action')!='statistics.execute' or approval.get('target')!='statistics-local' or approval.get('status')!='approved' or approval.get('hash')!=receipt.get('hash') or approval.get('payload')!=expected or approval.get('expiresAt',0)<time.time()*1000: raise ValueError('Missing, expired, changed or unapproved statistics operation')
        current=preview({'dataset':expected['dataset'],'plan':expected['plan']})
        if current!=expected['preview']: raise ValueError('Data or worker environment changed; request a new approval')
        db.execute('CREATE TABLE IF NOT EXISTS statistics_claims (approval_id TEXT PRIMARY KEY, created_at REAL NOT NULL)')
        try:
            db.execute('INSERT INTO statistics_claims VALUES (?,?)',(receipt['id'],time.time())); db.commit()
        except sqlite3.IntegrityError: raise ValueError('This approval has already been consumed; no duplicate execution')
    finally: db.close()
    directory=home/'statistics'/expected['analysisId']; directory.mkdir(parents=True,exist_ok=False)
    progress_path=home/'statistics-progress'/f"{expected['analysisId']}.json"
    progress_path.parent.mkdir(parents=True,exist_ok=True)
    def progress(status,completed):
        state={'analysisId':expected['analysisId'],'pid':os.getpid(),'status':status,'completedSteps':completed,'totalSteps':len(expected['plan']['steps']),'updatedAt':time.time()}
        temporary=progress_path.with_suffix('.tmp');temporary.write_text(json.dumps(state));temporary.replace(progress_path)
    progress('running',0)
    data=data_for_plan(expected['dataset'],expected['plan']); started=time.time()
    try:
        results=[]
        for step in expected['plan']['steps']:
            if time.time()-started>100: raise TimeoutError('Batch budget exhausted before next analysis; split the plan into smaller approvals')
            results.append(analyze(data,step))
            (directory/f'step-{len(results):02d}.json').write_text(json.dumps(results[-1],ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
            progress('running',len(results))
        manifest={'schemaVersion':'theta.statistics.v1','question':expected['plan']['question'],'plan':expected['plan'],'dataset':expected['dataset'],'runtime':current['runtime'],'results':results,'elapsedSeconds':time.time()-started,'analysisId':expected['analysisId']}
        files=write(directory,manifest)
        # Reproduction reuses verified input version and serialized specification, never untrusted formulas.
        script='# Run from agent/ with: .local/runtimes/statistics/bin/python -s /path/to/reproduce.py\nimport json\nimport sys\nimport time\nfrom pathlib import Path\nif not (Path.cwd() / "workers" / "statistics" / "engine.py").is_file():\n    raise SystemExit("Run this script from the THETA agent/ directory with its statistics Python.")\nsys.path.insert(0, str(Path.cwd()))\nfrom workers.statistics.engine import data_for_plan, analyze, preview\nfrom workers.statistics.report import write\nmanifest = json.loads(Path(__file__).with_name("results.json").read_text())\ncurrent = preview({"dataset": manifest["dataset"], "plan": manifest["plan"]})\nmanifest["reproduction"] = {"originalRuntime": manifest["runtime"], "sameSource": current["runtime"]["sourceFingerprint"] == manifest["runtime"]["sourceFingerprint"]}\nmanifest["runtime"] = current["runtime"]\nstarted = time.time()\ndata = data_for_plan(manifest["dataset"], manifest["plan"])\nmanifest["results"] = [analyze(data, step) for step in manifest["plan"]["steps"]]\nmanifest["elapsedSeconds"] = time.time() - started\nwrite(Path(__file__).with_name("reproduced"), manifest)\n'
        (directory/'reproduce.py').write_text(script); files.append(directory/'reproduce.py')
        entries=[dict(name=f.name,path=str(f),kind=f.suffix[1:],sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in files]
        (directory/'manifest.json').write_text(json.dumps({'files':entries},ensure_ascii=False,indent=2))
        entries.append(dict(name='manifest.json',path=str(directory/'manifest.json'),kind='json'))
        # Full tables are archived; only a bounded preview enters the chat context.
        bounded=[{**r,'tables':{k:{'rows':v[:5],'totalRows':len(v)} for k,v in list(r.get('tables',{}).items())[:12]}} for r in results]
        progress('complete',len(results))
        return {'analysisId':expected['analysisId'],'status':'complete','reportPath':str(directory/'report.html'),'files':entries,'results':bounded,
                'instruction':'Use exact estimates, uncertainty and sample exclusions. Explain each delivered table and figure in relation to the question; distinguish association, prediction and causal identification. Full tables are downloadable; do not infer rows absent from this preview.'}
    except Exception as exc:
        progress('failed',len(results))
        (directory/'failure.json').write_text(json.dumps({'status':'failed','error':str(exc),'plan':expected['plan']},ensure_ascii=False,indent=2))
        raise
