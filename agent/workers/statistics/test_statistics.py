"""Numerical invariants + executable catalogue smoke matrix (synthetic, not clinical evidence)."""
import json
from pathlib import Path
import tempfile
import unittest
import warnings
import numpy as np
import pandas as pd
from .engine import analyze
from .registry import METHODS
from .report import write,clean,rtf,tex


def fixture():
    rng=np.random.default_rng(731); n=180
    x=rng.normal(size=n); z=rng.normal(size=n); x2=rng.normal(size=n)
    group=np.tile([0,1],n//2); post=np.tile([0,0,1,1],n//4)
    noise=rng.normal(scale=.8,size=n)
    latent=.5+x+.3*x2+noise
    d=pd.DataFrame({'x':x,'x2':x2,'z':z,'endog':.9*z+.3*x+noise,'y':1+2*x+.5*x2+rng.normal(size=n),
      'binary':(latent>0).astype(int),'binary2':(latent+rng.normal(size=n)>.2).astype(int),
      'binary3':(latent+rng.normal(size=n)>.5).astype(int),'ordinal':np.digitize(latent,[-.5,.7]),
      'count':rng.poisson(np.exp(.4+.2*x)), 'positive':rng.gamma(2,np.exp(.1*x)),
      'group':group,'post':post,'entity':np.repeat(np.arange(30),6),'wave':np.tile(np.arange(6),30),
      'time':np.arange(n),'duration':rng.exponential(np.exp(-.3*x))*10+.01,'event':rng.binomial(1,.7,n),
      'weight':rng.uniform(.5,2,n),'score':rng.integers(0,11,n),'a':rng.uniform(.1,1,n),'b':rng.uniform(.1,1,n),'c':rng.uniform(.1,1,n),
      'text':['good experience' if v else 'bad experience' for v in group]})
    d['ydid']=2*group*post+.2*x+rng.normal(size=n)
    return d


def specification(method,d):
    family=METHODS[method]['family']; s={'method':method,'x':['x','x2'],'seed':42,'params':{}}
    if family=='descriptive':
        if method=='nps':s['x']=['score']
        if method in {'cronbach_alpha','kmo_bartlett'}:s['x']=['a','b','c']
    if family=='test':
        if method in {'ttest_one','runs','tost_one'}:s['x']=['x']
        if method=='binomial':s['x']=['binary']
        if method=='cochran_q':s['x']=['binary','binary2','binary3']
        if method=='friedman':s['x']=['a','b','c']
        if method=='kappa':s['x']=['binary','binary2']
    if family=='group_test':s={'method':method,'y':'y','group':'group'}
    if family=='contingency':s={'method':method,'x':['binary'],'group':'binary2'}
    if family in {'regression','econometrics','ml'}:
        s['y']='binary' if method.endswith('_classifier') or method in {'logit','probit','glm_binomial'} else 'y'
        if method in {'poisson','negative_binomial'}:s['y']='count' if method=='poisson' else 'positive_count';d['positive_count']=np.random.default_rng(5).negative_binomial(2,.4,len(d))
        if method in {'ordinal_logit','multinomial_logit'}:s['y']='ordinal'
        if method=='glm_gamma':s['y']='positive'
        if method in {'wls'}:s['weight']='weight'
        if method in {'gee','mixed_linear'}:s['group']='entity'; d['y']=d['y']+d['entity']*.1
        if method.startswith('panel_'):s.update(entity='entity',time='wave')
        if method.startswith('iv_'):s.update(endogenous=['endog'],instruments=['z'])
        if method=='did':s.update(group='group',time='post',y='ydid')
        if method=='rdd':s['params']={'cutoff':0,'bandwidth':2}
    if family=='survival':
        s.update(time='duration',event='event')
        if method in {'kaplan_meier','nelson_aalen','logrank'}:s.pop('x');s['group']='group'
    if family=='unsupervised':
        if method=='cca':s['endogenous']=['a','b']
        if method=='factor':s['x']=['a','b','c','x'];s['params']['components']=1
        if method=='correspondence':s['x']=['a','b','c']
    if family=='timeseries':
        s['time']='time';s['params']={'lags':2,'horizon':3}
        if method not in {'var','granger'}:s['x']=['x']
    if family=='decision':
        s['x']=['a','b','c']
        if method=='ahp':
            d=pd.DataFrame([[1,2,4],[.5,1,2],[.25,.5,1]],columns=['a','b','c'])
    if family=='preprocess' and method=='dummy':s['x']=['text']
    if family=='optimization':
        s={'method':method,'params':{'c':[-1,-2],'a_ub':[[1,1]],'b_ub':[5]}}
        if method=='quadratic_program':s['params']['q']=[[2,0],[0,2]]
    if family=='design':
        s={'method':method,'params':{}}
        if method in {'anova_factorial','ancova'}:s.update(factors=['group','post'],y='y');s['x']=['x'] if method=='ancova' else []
        if method=='anova_factorial':s.pop('x')
        if method=='repeated_anova':s.update(factors=['wave'],entity='entity',y='y')
        if method=='manova':s.update(factors=['group'],endogenous=['y','positive'])
        if method=='icc':s['x']=['a','b','c']
        if method=='moderation':s.update(x=['x','x2'],y='y')
        if method in {'mediation','parallel_mediation','chain_mediation'}:s.update(x=['x'],y='y',endogenous=['endog'] if method=='mediation' else ['endog','positive'],params={'bootstrap':100})
        if method=='psm':s.update(x=['x','x2'],y='y',group='binary')
        if method=='apriori':s['x']=['binary','binary2','binary3']
        if method=='sem':
            rng=np.random.default_rng(73); latent=rng.normal(size=len(d))
            for j in range(5):d[f'item{j}']=latent+rng.normal(size=len(d))
            s.update(x=[f'item{j}' for j in range(5)],params={'model':'F =~ item0 + item1 + item2 + item3 + item4'})
    if 'params' in s:s['params']={k:v for k,v in s['params'].items() if k in METHODS[method]['parameters']}
    return d,s


class StatisticsTests(unittest.TestCase):
    def test_published_examples_run_on_the_documented_csv_files(self):
        root=Path(__file__).resolve().parents[2]/'examples'/'statistics'
        examples=json.loads((root/'methods.json').read_text());inputs=json.loads((root/'inputs.json').read_text())
        self.assertEqual(set(examples),set(METHODS))
        for method,spec in examples.items():
            with self.subTest(method=method):
                result=analyze(pd.read_csv(root/inputs[method]),spec)
                self.assertEqual(result['method'],method)

    def test_every_registered_method_executes_and_serializes(self):
        for method in METHODS:
            with self.subTest(method=method):
                data,spec=specification(method,fixture())
                result=analyze(data,spec)
                self.assertEqual(result['method'],method)
                self.assertTrue(result.get('metrics') or result.get('tables'))
                json.dumps(result,allow_nan=False)

    def test_discrete_covariance_does_not_leak_empty_optimizer_arguments(self):
        for method in ['logit', 'probit', 'poisson', 'negative_binomial', 'multinomial_logit', 'ordinal_logit']:
            for covariance in ['nonrobust', 'HC1', 'cluster']:
                data=fixture(); data,spec=specification(method,data)
                spec['params']['covariance']=covariance
                if covariance=='cluster': spec['group']='entity'
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    result=analyze(data,spec)
                messages=[str(w.message) for w in caught]+result.get('warnings',[])
                self.assertFalse(any('cov_kwds' in message for message in messages), (method,covariance,messages))

    def test_ols_matches_closed_form_and_robust_se_changes(self):
        d=fixture();spec={'method':'ols','x':['x','x2'],'y':'y'}
        result=analyze(d,spec);matrix=np.column_stack([np.ones(len(d)),d[['x','x2']]])
        beta=np.linalg.solve(matrix.T@matrix,matrix.T@d.y)
        np.testing.assert_allclose([r['coefficient'] for r in result['coefficients']],beta,rtol=1e-11)
        sigma=((d.y-matrix@beta)**2).sum()/(len(d)-3);se=np.sqrt(np.diag(sigma*np.linalg.inv(matrix.T@matrix)))
        np.testing.assert_allclose([r['std_error'] for r in result['coefficients']],se,rtol=1e-11)
        robust=analyze(d,{**spec,'params':{'covariance':'HC3'}})
        np.testing.assert_allclose([r['coefficient'] for r in robust['coefficients']],beta)
        self.assertNotEqual(robust['coefficients'][1]['std_error'],result['coefficients'][1]['std_error'])
        from scipy.stats import t
        row=robust['coefficients'][1]
        self.assertAlmostEqual(row['p_value'],2*t.sf(abs(row['statistic']),len(d)-3),places=12)
        self.assertEqual(robust['metrics']['inference_distribution'],'Student t')
        self.assertNotIn('durbin_watson',robust['metrics'],'Row order alone is not a time-series design')

    def test_known_optimization_and_ahp(self):
        d=fixture();r=analyze(d,{'method':'linear_program','params':{'c':[-1,-2],'a_ub':[[1,1]],'b_ub':[5]}})
        self.assertAlmostEqual(r['metrics']['objective'],-10)
        d,s=specification('ahp',d);r=analyze(d,s)
        np.testing.assert_allclose([v['weight'] for v in r['tables']['weights']],np.array([4,2,1])/7)
        self.assertAlmostEqual(r['metrics']['CR'],0)

    def test_no_tuning_and_invalid_search_are_checked_before_execution(self):
        from .engine import prepare
        d,spec=specification('logistic_classifier',fixture())
        spec['params']={'tune':False,'cv':0,'test_size':.25}
        result=analyze(d,spec)
        self.assertEqual(result['metrics']['cv_folds'],0)
        self.assertEqual(result['metrics']['test_N'],45)
        for invalid in ({'tune':True},{'tune':{'C':[1]},'cv':0},{'cv':1},{'tune':{'unregistered':[1]},'cv':3}):
            with self.subTest(params=invalid),self.assertRaises(ValueError):
                prepare(d,{**spec,'params':invalid})

    def test_missingness_pairing_and_invalid_models(self):
        d=fixture();d.loc[0,'x']=np.nan
        r=analyze(d,{'method':'ols','x':['x'],'y':'y'});self.assertEqual(r['metrics']['N'],179);self.assertEqual(r['sample']['excluded_N'],1)
        with self.assertRaises(ValueError):analyze(d,{'method':'ols','x':['x'],'y':'y','missing':'error'})
        with self.assertRaises(ValueError):analyze(d,{'method':'ols','x':['y'],'y':'y'})
        with self.assertRaises(ValueError):analyze(d,{'method':'cox','x':['x'],'time':'duration','event':'score'})
        with self.assertRaises(ValueError):analyze(d,{'method':'logit','x':['x'],'y':'y'})
        with self.assertRaises(ValueError):analyze(d,{'method':'arima','x':['x'],'time':'time'})
        with self.assertRaises(ValueError):analyze(d,{'method':'ols','x':['x'],'y':'y','params':{'formula':'__import__("os")'}})

    def test_training_only_search_and_text(self):
        d=fixture();s={'method':'logistic_classifier','x':['x'],'y':'binary','params':{'text_column':'text','cv':3,'tune':{'C':[.1,1]}}}
        r=analyze(d,s);self.assertEqual(r['metrics']['train_N']+r['metrics']['test_N'],len(d));self.assertIn('cv_score',r['metrics'])
        self.assertEqual(len(r['tables']['test_predictions']),r['metrics']['test_N'])
        s['params']['split']='group';s['group']='entity';r=analyze(d,s)
        test_rows=[row['source_row']-2 for row in r['tables']['test_predictions']];test_groups=set(d.loc[test_rows,'entity']);train_groups=set(d.drop(test_rows).entity)
        self.assertFalse(test_groups&train_groups)

    def test_paper_outputs_share_values_escape_labels(self):
        d=fixture();r=analyze(d,{'method':'ols','name':'效果 <script> & 试验','x':['x'],'y':'y'})
        with tempfile.TemporaryDirectory() as tmp:
            manifest={'question':'A <script>','dataset':{'sha256':'a'*64},'plan':{'steps':[r['spec']]},'results':[r]}
            files=write(tmp,manifest);paths={p.name for p in files}
            self.assertTrue({'esttab.tex','esttab.rtf','esttab.csv','report.html','report.md','results.json','01-ols.pdf','01-ols.svg','01-ols.png'}<=paths)
            self.assertNotIn('<script>',Path(tmp,'report.html').read_text())
            self.assertIn('\\u',Path(tmp,'esttab.rtf').read_text())
            self.assertIn('\\&',Path(tmp,'esttab.tex').read_text())
            self.assertIn('confidence intervals',r['figure_caption'])


if __name__=='__main__':unittest.main()
