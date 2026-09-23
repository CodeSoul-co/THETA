"""Classical tests and explicitly parameterized statistical estimators."""
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.weightstats import ttost_ind, ttost_paired, ztest


def records(frame):
    return frame.reset_index().rename(columns={'index': 'variable'}).to_dict('records')


def fitted(result, names=None, alpha=.05):
    params = np.asarray(result.params)
    se, p = np.asarray(result.bse), np.asarray(result.pvalues)
    ci = np.asarray(result.conf_int(alpha=alpha))
    names = list(names if names is not None else result.params.index)
    rows = []
    if params.ndim == 1:
        for i, name in enumerate(names):
            rows.append(dict(term=str(name), coefficient=params[i], std_error=se[i], statistic=np.asarray(result.tvalues)[i], p_value=p[i], ci_low=ci[i, 0], ci_high=ci[i, 1]))
    else:
        ci = ci.reshape(params.shape[1], params.shape[0], 2)
        labels=getattr(result.model,'_ynames_map',{})
        for j in range(params.shape[1]):
            for i, name in enumerate(names):
                rows.append(dict(term=f'{labels.get(j+1,j+1)} vs {labels.get(0,0)}: {name}', coefficient=params[i,j], std_error=se[i,j], statistic=np.asarray(result.tvalues)[i,j], p_value=p[i,j], ci_low=ci[j,i,0], ci_high=ci[j,i,1]))
    metrics = {'N': int(result.nobs), 'covariance': getattr(result, 'cov_type', 'model-specific')}
    metrics['inference_distribution'] = 'Student t' if getattr(result,'use_t',False) else 'asymptotic normal (z)'
    if params.ndim>1: metrics['reference_category']=getattr(result.model,'_ynames_map',{}).get(0,0)
    if getattr(result,'use_t',False): metrics['inference_df'] = float(getattr(result,'df_resid_inference',result.df_resid))
    for attr in ('rsquared', 'rsquared_adj', 'prsquared', 'aic', 'bic', 'llf', 'df_resid'):
        try: value = getattr(result, attr, None)
        except NotImplementedError: value = None
        if value is not None and np.isscalar(value): metrics[attr] = float(value)
    return dict(coefficients=rows, metrics=metrics, tables={})


def descriptive(data, spec):
    method, cols, params = spec['method'], spec['x'], spec.get('params', {})
    d = data[cols]
    if method == 'frequency':
        return {'tables': {col: d[col].value_counts(dropna=False).rename('count').to_frame().assign(percent=lambda f: 100*f['count']/len(d)).reset_index().to_dict('records') for col in cols}}
    d = d.astype(float)
    if method == 'describe':
        table = d.describe(percentiles=[.25,.5,.75]).T
        table['skew'] = d.skew(); table['kurtosis_excess'] = d.kurtosis()
        return {'tables': {'descriptives': records(table)}}
    if method == 'normality':
        if not 3 <= len(d) <= 5000: raise ValueError('Shapiro requires 3..5000 cases; select another diagnostic for larger data')
        return {'tables': {'normality': [dict(variable=c, W=stats.shapiro(d[c]).statistic, p_value=stats.shapiro(d[c]).pvalue, N=len(d)) for c in cols]}}
    if method == 'correlation':
        function = {'pearson': stats.pearsonr, 'spearman': stats.spearmanr, 'kendall': stats.kendalltau}[params.get('correlation','pearson')]
        rows=[]
        for i,c in enumerate(cols):
            for other in cols[i+1:]:
                r,p=function(d[c],d[other]); rows.append(dict(x=c,y=other,coefficient=r,p_value=p,N=len(d)))
        return {'tables': {'correlations': rows}, 'warnings':['Pairwise tests are unadjusted; predefine the hypothesis family or adjust multiplicity.']}
    if method == 'cronbach_alpha':
        k=len(cols)
        if k<2 or d.sum(axis=1).var(ddof=1)<=0: raise ValueError('Alpha requires >=2 nonconstant items and nonzero total-score variance')
        value=k/(k-1)*(1-d.var(ddof=1).sum()/d.sum(axis=1).var(ddof=1))
        return {'metrics': {'alpha':value,'items':k}, 'warnings':['Reverse-code items explicitly first; alpha does not establish unidimensionality or validity.']}
    if method == 'kmo_bartlett':
        from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
        each,total=calculate_kmo(d); chi,p=calculate_bartlett_sphericity(d)
        return {'metrics':{'KMO':total,'bartlett_chi2':chi,'p_value':p}, 'tables':{'KMO_by_item':[dict(variable=c,KMO=v) for c,v in zip(cols,each)]}}
    if method == 'nps':
        if len(cols)!=1 or not d.iloc[:,0].between(0,10).all(): raise ValueError('NPS requires one 0..10 response column')
        x=d.iloc[:,0]; return {'metrics':{'NPS':100*((x>=9).mean()-(x<=6).mean()),'promoter_fraction':(x>=9).mean(),'detractor_fraction':(x<=6).mean()}}
    raise ValueError(method)


def tests(data, spec):
    m=spec['method']; p=spec.get('params',{}); a=spec.get('alpha',.05)
    cols=spec.get('x',[]); d=data[cols].astype(float) if cols else None
    out={'tables':{},'metrics':{}}
    if m in {'ttest_ind','mann_whitney','anova_one','kruskal','levene','tukey','dunn','tost_ind'}:
        groups=[g[spec['y']].astype(float).to_numpy() for _,g in data.groupby(spec['group'],sort=True)]
        if len(groups)<2 or min(map(len,groups))<2: raise ValueError('At least two groups with >=2 observations required')
        if m in {'ttest_ind','mann_whitney','tost_ind'} and len(groups)!=2: raise ValueError('Exactly two groups required')
        if m=='ttest_ind': r=stats.ttest_ind(*groups,equal_var=False); out['metrics']['mean_difference']=groups[0].mean()-groups[1].mean()
        elif m=='mann_whitney': r=stats.mannwhitneyu(*groups,alternative='two-sided')
        elif m=='anova_one': r=stats.f_oneway(*groups)
        elif m=='kruskal': r=stats.kruskal(*groups)
        elif m=='levene': r=stats.levene(*groups,center='median')
        elif m=='tukey':
            from statsmodels.stats.multicomp import pairwise_tukeyhsd
            r=pairwise_tukeyhsd(data[spec['y']],data[spec['group']],alpha=a)
            return {'tables':{'comparisons':[dict(zip(r.summary().data[0],row)) for row in r.summary().data[1:]]}}
        elif m=='dunn':
            import scikit_posthocs as sp
            return {'tables':{'adjusted_p_values':records(sp.posthoc_dunn(data,val_col=spec['y'],group_col=spec['group'],p_adjust=p.get('p_adjust','holm')))}}
        else:
            pv,lower,upper=ttost_ind(*groups,p.get('low',-.5),p.get('high',.5),usevar='unequal')
            return {'metrics':{'p_value':pv,'lower_test':list(lower),'upper_test':list(upper)}}
        out['metrics'].update(statistic=r.statistic,p_value=r.pvalue)
        if hasattr(r,'df'): out['metrics']['df']=r.df
        if hasattr(r,'confidence_interval'):
            ci=r.confidence_interval(confidence_level=1-a); out['metrics'].update(ci_low=ci.low,ci_high=ci.high)
        return out
    if d is None: raise ValueError('x required')
    x=d.iloc[:,0].to_numpy(); mu=p.get('mu',0)
    if m in {'ttest_paired','wilcoxon','tost_paired','kappa','bland_altman'} and len(cols)!=2: raise ValueError('Exactly two paired columns required')
    if m in {'ttest_one','ttest_paired'}:
        r=stats.ttest_1samp(x,mu) if m=='ttest_one' else stats.ttest_rel(x,d.iloc[:,1])
        ci=r.confidence_interval(confidence_level=1-a)
        out['metrics'].update(statistic=r.statistic,p_value=r.pvalue,df=r.df,ci_low=ci.low,ci_high=ci.high,
                              mean_difference=x.mean()-mu if m=='ttest_one' else (x-d.iloc[:,1]).mean())
        return out
    if m=='wilcoxon': r=stats.wilcoxon(x,d.iloc[:,1])
    elif m=='friedman': r=stats.friedmanchisquare(*[d[c] for c in cols])
    elif m=='kendall_w':
        if len(cols)<2 or len(d)<3: raise ValueError('Rows are objects and columns are judges; need >=3 objects, >=2 judges')
        r=stats.friedmanchisquare(*[row for row in d.to_numpy()]); out['metrics']['W']=r.statistic/(len(cols)*(len(d)-1))
    elif m=='cochran_q':
        from statsmodels.stats.contingency_tables import cochrans_q
        if not np.isin(d,[0,1]).all(): raise ValueError('Cochran Q requires binary items')
        r=cochrans_q(d)
    elif m=='runs':
        from statsmodels.sandbox.stats.runs import runstest_1samp
        z,pv=runstest_1samp(x); return {'metrics':{'z':z,'p_value':pv}}
    elif m=='binomial':
        if not np.isin(x,[0,1]).all(): raise ValueError('Binary 0/1 outcomes required')
        r=stats.binomtest(int(x.sum()),len(x),p.get('probability',.5)); ci=r.proportion_ci(1-a)
        return {'metrics':{'proportion':r.statistic,'p_value':r.pvalue,'ci_low':ci.low,'ci_high':ci.high}}
    elif m in {'tost_one','tost_paired'}:
        y=np.zeros(len(x)) if m=='tost_one' else d.iloc[:,1].to_numpy()
        pv,lo,hi=ttost_paired(x,y,p.get('low',-.5),p.get('high',.5))
        return {'metrics':{'p_value':pv,'lower_test':list(lo),'upper_test':list(hi)}}
    elif m=='kappa':
        from sklearn.metrics import cohen_kappa_score
        return {'metrics':{'kappa':cohen_kappa_score(x,d.iloc[:,1])}}
    elif m=='bland_altman':
        delta=x-d.iloc[:,1]; mean=delta.mean(); sd=delta.std(ddof=1)
        return {'metrics':{'bias':mean,'lower_95_agreement':mean-1.96*sd,'upper_95_agreement':mean+1.96*sd},'tables':{'agreement':pd.DataFrame({'mean':(x+d.iloc[:,1])/2,'difference':delta}).to_dict('records')},'warnings':['Limits of agreement are not confidence intervals; approximate normality of differences assumed.']}
    else: raise ValueError(m)
    out['metrics'].update(statistic=r.statistic,p_value=r.pvalue); return out


def contingency(data,spec):
    if len(spec['x'])!=1: raise ValueError('One x column required')
    table=pd.crosstab(data[spec['x'][0]],data[spec['group']]); m=spec['method']
    if min(table.shape)<2: raise ValueError('At least two observed categories per variable required')
    if m in {'fisher','mcnemar','chi_square_yates'} and table.shape!=(2,2): raise ValueError('This method requires a 2x2 table')
    output={'tables':{'counts':records(table)},'metrics':{}}
    if m=='fisher': r=stats.fisher_exact(table); output['metrics']={'odds_ratio':r.statistic,'p_value':r.pvalue}
    elif m=='mcnemar':
        from statsmodels.stats.contingency_tables import mcnemar
        if list(table.index)!=list(table.columns): raise ValueError('Paired category encodings must match')
        r=mcnemar(table,exact=True); output['metrics']={'statistic':r.statistic,'p_value':r.pvalue}
    else:
        chi,p,df,expected=stats.chi2_contingency(table,correction=m=='chi_square_yates')
        output['metrics']={'chi2':chi,'p_value':p,'df':df,'cramers_v':np.sqrt(chi/(table.to_numpy().sum()*(min(table.shape)-1)))}
        output['tables']['expected']=records(pd.DataFrame(expected,index=table.index,columns=table.columns))
        if (expected<5).any(): output['warnings']=['Expected count below 5; asymptotic chi-square may be unreliable.']
    return output


def regression(data,spec):
    m=spec['method']; p=spec.get('params',{}); alpha=spec.get('alpha',.05)
    y=data[spec['y']].astype(float); raw=data[spec['x']].astype(float)
    x=sm.add_constant(raw,has_constant='add'); cov=p.get('covariance','nonrobust'); kw={}
    if len(x)<=x.shape[1]+1 or np.linalg.matrix_rank(x)<x.shape[1]: raise ValueError('Insufficient residual degrees of freedom or collinear/constant predictors')
    if cov=='cluster':
        if not spec.get('group') or data[spec['group']].nunique()<2: raise ValueError('Cluster covariance needs group with >=2 clusters')
        kw={'groups':data[spec['group']]}
    if cov=='HAC': kw={'maxlags':p.get('max_lags',1)}
    if m in {'logit','probit','glm_binomial'} and not set(y.unique())<={0,1}: raise ValueError('Binary outcome must be coded 0/1')
    if m in {'poisson','negative_binomial'} and ((y<0).any() or (y!=np.floor(y)).any()): raise ValueError('Counts must be nonnegative integers')
    if m=='glm_gamma' and (y<=0).any(): raise ValueError('Gamma outcome must be positive')
    models={'ols':sm.OLS,'logit':sm.Logit,'probit':sm.Probit,'poisson':sm.Poisson,'negative_binomial':sm.NegativeBinomial,'multinomial_logit':sm.MNLogit}
    if m in models:
        kwargs={'disp':False,'maxiter':200} if m not in {'ols'} else {'use_t':True}
        # Empty cov_kwds leaks into discrete optimizers in statsmodels 0.14.
        if kw: kwargs['cov_kwds']=kw
        r=models[m](y,x).fit(cov_type=cov,**kwargs)
    elif m=='wls':
        if not spec.get('weight') or (data[spec['weight']]<=0).any(): raise ValueError('WLS requires positive inverse-variance weights')
        r=sm.WLS(y,x,weights=data[spec['weight']]).fit(cov_type=cov,cov_kwds=kw,use_t=True)
    elif m in {'glm_gamma','glm_binomial'}:
        family=sm.families.Gamma(sm.families.links.Log()) if m=='glm_gamma' else sm.families.Binomial()
        r=sm.GLM(y,x,family=family).fit(cov_type=cov,cov_kwds=kw)
    elif m=='ordinal_logit':
        from statsmodels.miscmodels.ordinal_model import OrderedModel
        r=OrderedModel(y,raw,distr='logit').fit(method='bfgs',disp=False,maxiter=200,cov_type=cov,**({'cov_kwds':kw} if kw else {}))
    elif m=='quantile':
        if cov not in {'nonrobust','HC1'}: raise ValueError('Quantile supports iid (nonrobust) or robust (HC1 label) density-based SE only')
        r=sm.QuantReg(y,x).fit(q=p.get('quantile',.5),vcov='iid' if cov=='nonrobust' else 'robust',max_iter=2000)
    elif m=='robust_linear':
        if cov!='nonrobust': raise ValueError('Huber RLM uses H1 covariance; custom covariance not supported')
        r=sm.RLM(y,x,M=sm.robust.norms.HuberT()).fit(maxiter=200)
    elif m in {'mixed_linear','gee'}:
        if not spec.get('group'): raise ValueError('group required')
        if cov!='nonrobust': raise ValueError('This estimator uses its native covariance; omit covariance')
        if m=='mixed_linear': r=sm.MixedLM(y,x,groups=data[spec['group']]).fit(reml=True)
        else: r=sm.GEE(y,x,groups=data[spec['group']],cov_struct=sm.cov_struct.Exchangeable()).fit()
    else: raise ValueError(m)
    if getattr(r,'mle_retvals',{}).get('converged') is False or getattr(r,'converged',True) is False: raise ValueError('Estimator did not converge; no valid inferential table delivered')
    out=fitted(r,alpha=alpha)
    if m in {'ols','wls'}:
        from statsmodels.stats.diagnostic import het_breuschpagan
        from statsmodels.stats.stattools import durbin_watson
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        lm,pv,_,_=het_breuschpagan(r.resid,x)
        jb=stats.jarque_bera(r.resid)
        out['metrics'].update(breusch_pagan_lm=lm,breusch_pagan_p=pv,residual_skew=stats.skew(r.resid,bias=False),residual_excess_kurtosis=stats.kurtosis(r.resid,bias=False),jarque_bera=jb.statistic,jarque_bera_p=jb.pvalue)
        if spec.get('time') and data[spec['time']].is_monotonic_increasing and not data[spec['time']].duplicated().any(): out['metrics']['durbin_watson']=durbin_watson(r.resid)
        bins=pd.qcut(r.fittedvalues,4,duplicates='drop')
        summary=pd.DataFrame({'fitted':r.fittedvalues,'residual':r.resid,'bin':bins}).groupby('bin',observed=True).agg(N=('residual','size'),mean_fitted=('fitted','mean'),mean_residual=('residual','mean'),sd_residual=('residual','std')).reset_index(drop=True)
        out['tables']['residual_bins']=summary.to_dict('records')
        out['tables']['VIF']=[dict(variable=c,VIF=variance_inflation_factor(x.to_numpy(),i)) for i,c in enumerate(x.columns) if c!='const']
        out['tables']['fitted']=[dict(source_row=int(i)+2,observed=float(y.loc[i]),fitted=float(v),residual=float(r.resid.loc[i])) for i,v in r.fittedvalues.items()]
    if m=='ordinal_logit':out['metrics']['ordinal_cutpoints']=list(r.model.transform_threshold_params(r.params)[1:-1])
    out['estimand']='Conditional association; coefficient scale is log-odds for logit, latent index for probit/ordinal, log count for Poisson/NB. NB alpha is dispersion, not an exposure coefficient; ordinal threshold rows are internal threshold parameters (actual cutpoints are recorded separately). No automatic causal interpretation.'
    return out
