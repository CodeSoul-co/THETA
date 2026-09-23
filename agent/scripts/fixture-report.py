"""Simulated model outputs for acceptance. Never trains or sends requests."""
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workers.result_report import generate_report
from workers.results_reader import tree_hash
home, job_id, model = sys.argv[1:]
root = Path(home) / 'simulated' / job_id
root.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng({'lda': 11, 'stm': 12, 'theta': 13}[model])
theta = rng.dirichlet([3, 2, 1], size=36)
np.save(root / 'theta_k3.npy', theta)
np.save(root / 'beta_k3.npy', np.array([[.7,.2,.1,0,0,0], [0,0,.1,.6,.3,0], [.1,0,0,0,.2,.7]]))
(root / 'topic_words_k3.json').write_text(json.dumps({'0':['delivery','parcel','late'], '1':['refund','billing','response'], '2':['login','application','error']}))
(root / 'metrics.json').write_text(json.dumps({'fixtureOnly':True,'coherence':{'lda':.42,'stm':.45,'theta':.48}[model]}))
job = {'id':job_id,'resultHash':tree_hash(root),'plan':{'modelId':model},'provenance':'simulated'}
print(json.dumps({'job':job,'report':generate_report(root,job,Path(home)/'reports')}))
