"""Real renderer smoke test for every model route; no mocks, training or network.

Run explicitly (not part of fast unittest discovery):
python tests/render_model_visualizations.py --output result/model-render-check
All matrices are synthetic; these figures are software checks, not research results.
"""
import argparse
import contextlib
import json
from pathlib import Path
import sys
import traceback

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/models'))
from visualization.run_visualization import run_baseline_visualization, run_all_visualizations

ROUTES = ['lda','hdp','btm','stm','dtm','etm','nvdm','gsm','prodlda','ctm',
          'ctm_zeroshot','ctm_combined','bertopic','theta:zero_shot','theta:supervised','theta:unsupervised']


def fixture(model):
    rng = np.random.default_rng(731)
    n, k = 48, 3
    vocab = ['政策','企业','创业','服务','技术','产品','资金','人才','培训','平台','市场','创新','数据','客户','社区']
    theta = rng.dirichlet([.7, 1.1, .9], n)
    beta = rng.dirichlet(np.ones(len(vocab)), k)
    words = [(i, [(vocab[j], float(beta[i,j])) for j in np.argsort(-beta[i])]) for i in range(k)]
    data = dict(theta=theta, beta=beta, vocab=vocab, topic_words=words,
                timestamps=np.array([pd.Timestamp(2021+i//12,1+(i%12),1).to_pydatetime() for i in range(n)]),
                dimension_values=np.array(['source A','source B','source C']*(n//3)),
                bow_matrix=rng.poisson(4,(n,len(vocab))),
                metrics={'TD':.8, 'iRBO':.75,'NPMI':.23,'C_V':.51,'UMass':-2.1,'Exclusivity':.62,'PPL':125.,
                         'NPMI_per_topic':[.11,.31,.27], 'C_V_per_topic':[.42,.52,.59],
                         'UMass_per_topic':[-2.3,-1.9,-2.1],'Exclusivity_per_topic':[.59,.61,.66]},
                plot_scope='SYNTHETIC SOFTWARE TEST — generated matrices, not OPC findings or trained models.')
    if model in {'etm','nvdm','gsm','prodlda','ctm','ctm_zeroshot','ctm_combined','theta'}:
        data['training_history']={'train_loss':[8.,5.,3.], 'val_loss':[8.2,5.5,3.7],
                                  'recon_loss':[7.,4.,2.], 'kl_loss':[1.,1.,1.], 'perplexity':[170.,150.,125.]}
    if model=='theta':
        data['training_history']={'stage1':data['training_history'], 'stage2':data['training_history'].copy()}
    if model=='stm':
        data.update(covariates=np.tile(np.arange(3),16).reshape(-1,1), covariate_names=['channel'],
                    Gamma=np.array([[0.,0.],[.2,-.3]]))
    if model=='dtm':
        data.update(beta_over_time=np.stack([rng.dirichlet(np.ones(len(vocab)),k) for _ in range(4)]),
                    time_slices_info={'unique_times':[2021,2022,2023,2024]})
    if model=='nvdm':data['theta']=rng.normal(0,1,(n,k))
    if model=='bertopic':
        data['document_topics']=theta.argmax(axis=1);data['document_topics'][:3]=-1
        data['theta'][:3]=0
    return data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--models',nargs='+',choices=ROUTES,default=ROUTES)
    parser.add_argument('--dpi',type=int,default=110)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    report=[]
    for route in args.models:
        family,_,mode=route.partition(':');out=args.output/route.replace(':','-');out.mkdir(exist_ok=True)
        data=fixture(family);original=data['theta'].copy()
        item={'route':route,'data':'synthetic, no training','status':'failed'}
        try:
            with (out/'render.log').open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                if family=='theta':run_all_visualizations(out,'synthetic',mode,output_dir=out,data=data,language='en',dpi=args.dpi)
                else:run_baseline_visualization(out,'synthetic',family,output_dir=out,data=data,language='en',dpi=args.dpi)
            np.testing.assert_array_equal(original,data['theta'])
            images=list(out.rglob('*.png'))
            assert len(images)>=10,(route,len(images))
            for f in images:
                with Image.open(f) as im:
                    assert im.width>100 and im.height>70,f
                    assert abs(im.info['dpi'][0]-args.dpi)<.1,f
                assert f.with_suffix('.pdf').read_bytes().startswith(b'%PDF'),f
                assert '<svg' in f.with_suffix('.svg').read_text(),f
            status=json.loads((out/'additional-chart-status.json').read_text())
            expected={'topic_words','topic_similarity','document_projection','wordclouds','wordcloud_grid'}
            if family not in {'nvdm','bertopic','dtm'}:expected|={'intertopic_distance','word_weights','pyldavis'}
            if family=='stm':expected.add('stm_covariates')
            if family=='dtm':expected.add('dtm_word_evolution')
            generated={x['chart'] for x in status if x['status']=='generated'}
            assert expected<=generated,(route,expected-generated,status)
            assert not any(x['status']=='failed' for x in status),status
            assert (out/'global/topic_wordcloud_grid_1.png').exists()
            if family!='nvdm':
                assert (out/'global/topic_network_circular.png').exists()
                for kind in ['year','source']:
                    assert (out/f'global/{kind}_topic_sankey.png').exists()
                    assert (out/f'global/{kind}_topic_sankey.html').exists()
            if family=='stm':assert (out/'global/stm_gamma_coefficients.png').exists()
            if family=='dtm':assert (out/'global/topic_word_evolution.png').exists()
            item.update(status='passed',figures=len(images),formats=['png','pdf','svg'],
                        generated=sorted(generated),skipped=[x for x in status if x['status']=='skipped'])
        except Exception:
            item['error']=traceback.format_exc()
        report.append(item)
        (args.output/'render-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print(route,item['status'],item.get('figures',''),flush=True)
    if any(x['status']=='failed' for x in report):raise SystemExit(1)


if __name__=='__main__':main()
