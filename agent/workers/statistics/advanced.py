"""Econometrics, survival, latent structure, and ordered time-series methods."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from .inference import fitted, records


def econometrics(data,spec):
    m=spec['method']; p=spec.get('params',{}); cov=p.get('covariance','nonrobust'); alpha=spec.get('alpha',.05)
    if m in {'did','rdd'}:
        d=data.copy(); x=d[spec['x']].astype(float).copy()
        if m=='did':
            if not spec.get('group') or not spec.get('time'): raise ValueError('DID requires group=treated and time=post, both 0/1')
            g,t=d[spec['group']],d[spec['time']]
            if set(g)!={0,1} or set(t)!={0,1} or d.groupby([spec['group'],spec['time']]).size().size!=4: raise ValueError('All four binary treatment/time cells required')
            x['treated']=g; x['post']=t; x['treated_x_post']=g*t
            warning='Two-group two-period conditional DID; parallel trends, no anticipation and stable composition are assumptions, not tested facts.'
        else:
            running=x.iloc[:,0]-p.get('cutoff',0); band=p.get('bandwidth',1)
            use=running.abs()<=band; d=d.loc[use]; x=x.loc[use]; running=running.loc[use]
            if min((running<0).sum(),(running>=0).sum())<5: raise ValueError('RDD requires >=5 observations each side within the prespecified bandwidth')
            x.iloc[:,0]=running; x['above_cutoff']=(running>=0).astype(int); x['slope_change']=running*x['above_cutoff']
            warning='Sharp RDD with uniform kernel and prespecified bandwidth; no automatic bandwidth selection, bias correction or manipulation test.'
        x=sm.add_constant(x,has_constant='add')
        if np.linalg.matrix_rank(x)<x.shape[1] or len(x)<=x.shape[1]+1: raise ValueError('Rank-deficient design or insufficient observations')
        if cov=='HAC': raise ValueError('Use HC1/HC3 or valid cluster covariance for this design')
        kwargs={}
        if cov=='cluster':
            cluster=spec.get('entity')
            if not cluster or d[cluster].nunique()<2: raise ValueError('Clustered DID/RDD requires entity with >=2 independent clusters')
            kwargs={'groups':d[cluster]}
        r=sm.OLS(d[spec['y']].astype(float),x).fit(cov_type=cov,cov_kwds=kwargs)
        out=fitted(r,alpha=alpha); out['warnings']=[warning]; return out
    if cov not in {'nonrobust','HC1','cluster'}: raise ValueError('Panel/IV support nonrobust, HC1 or cluster only')
    covariance={'nonrobust':'unadjusted','HC1':'robust','cluster':'clustered'}[cov]; kwargs={}
    if m.startswith('panel_'):
        from linearmodels.panel import PanelOLS,RandomEffects
        if not spec.get('entity') or not spec.get('time'): raise ValueError('Panel requires entity and time')
        if data.duplicated([spec['entity'],spec['time']]).any(): raise ValueError('Duplicate entity-time observations')
        d=data.set_index([spec['entity'],spec['time']]).sort_index(); x=sm.add_constant(d[spec['x']].astype(float),has_constant='add'); y=d[spec['y']].astype(float)
        if cov=='cluster': kwargs={'cluster_entity':True}
        model=RandomEffects(y,x) if m=='panel_re' else PanelOLS(y,x,entity_effects=True,time_effects=m=='panel_twfe')
        r=model.fit(cov_type=covariance,**kwargs)
    else:
        from linearmodels.iv import IV2SLS,IVGMM
        if not spec.get('endogenous') or not spec.get('instruments'): raise ValueError('Explicit endogenous and excluded instrument columns required')
        if set(spec['x']) & (set(spec['endogenous'])|set(spec['instruments'])): raise ValueError('Exogenous, endogenous and excluded instruments must be distinct')
        if set(spec['endogenous']) & set(spec['instruments']): raise ValueError('An endogenous variable cannot instrument itself')
        if len(spec['instruments'])<len(spec['endogenous']): raise ValueError('Underidentified IV model')
        if cov=='cluster':
            if not spec.get('group'): raise ValueError('Cluster covariance requires group')
            kwargs={'clusters':data[spec['group']]}
        model=IV2SLS if m=='iv_2sls' else IVGMM
        r=model(data[spec['y']].astype(float),sm.add_constant(data[spec['x']].astype(float),has_constant='add'),data[spec['endogenous']].astype(float),data[spec['instruments']].astype(float)).fit(cov_type=covariance,**kwargs)
    ci=r.conf_int(level=1-alpha)
    out={'coefficients':[dict(term=str(k),coefficient=r.params[k],std_error=r.std_errors[k],statistic=r.tstats[k],p_value=r.pvalues[k],ci_low=ci.loc[k].iloc[0],ci_high=ci.loc[k].iloc[1]) for k in r.params.index],
         'metrics':{'N':r.nobs,'rsquared':r.rsquared,'covariance':covariance},'tables':{},'warnings':['Estimator equivalence does not establish instrument validity, exogeneity or causal identification.']}
    if m.startswith('iv_'):
        out['tables']['first_stage']=records(r.first_stage.diagnostics)
        if m=='iv_gmm': out['metrics'].update(J_statistic=r.j_stat.stat,J_p_value=r.j_stat.pval)
    return out


def survival(data,spec):
    import lifelines as ll
    from lifelines.statistics import multivariate_logrank_test,proportional_hazard_test
    m=spec['method']; t,e=spec['time'],spec['event']; alpha=spec.get('alpha',.05)
    if (data[t]<=0).any() or not set(data[e])<={0,1} or data[e].sum()==0: raise ValueError('Positive durations and binary event with at least one observed event required')
    metrics={'N':len(data),'events':int(data[e].sum()),'censored':int((data[e]==0).sum())}
    if m=='logrank':
        if not spec.get('group') or data[spec['group']].nunique()<2: raise ValueError('Log-rank needs >=2 groups')
        r=multivariate_logrank_test(data[t],data[spec['group']],data[e]); metrics.update(statistic=r.test_statistic,p_value=r.p_value)
        return {'metrics':metrics}
    if m in {'kaplan_meier','nelson_aalen'}:
        tables={}; groups=data.groupby(spec['group']) if spec.get('group') else [('all',data)]
        for name,d in groups:
            model=(ll.KaplanMeierFitter if m=='kaplan_meier' else ll.NelsonAalenFitter)(alpha=alpha).fit(d[t],d[e],label=str(name))
            curve=model.survival_function_ if m=='kaplan_meier' else model.cumulative_hazard_
            tables[str(name)+'_curve']=records(curve.join(model.confidence_interval_)); tables[str(name)+'_risk_table']=records(model.event_table)
        return {'metrics':metrics,'tables':tables}
    cols=spec.get('x',[])
    if not cols or set(cols)&{t,e}: raise ValueError('Survival regression requires separate x covariates')
    d=data[[t,e]+cols].astype(float)
    models={'cox':ll.CoxPHFitter,'weibull_aft':ll.WeibullAFTFitter,'lognormal_aft':ll.LogNormalAFTFitter,'loglogistic_aft':ll.LogLogisticAFTFitter}
    model=models[m](alpha=alpha).fit(d,duration_col=t,event_col=e)
    table=model.summary; rows=[]
    for name,row in table.iterrows():
        lower=next(k for k in row.index if k.startswith('coef lower')); upper=next(k for k in row.index if k.startswith('coef upper'))
        rows.append(dict(term=str(name),coefficient=row['coef'],std_error=row['se(coef)'],statistic=row['z'],p_value=row['p'],ci_low=row[lower],ci_high=row[upper],exp_coefficient=row['exp(coef)']))
    tables={'survival_regression':records(table)}
    if m=='cox': tables['PH_diagnostic']=records(proportional_hazard_test(model,d,time_transform='rank').summary)
    metrics.update(concordance=model.concordance_index_,log_likelihood=model.log_likelihood_)
    return {'coefficients':rows,'metrics':metrics,'tables':tables,'estimand':'Cox coefficient is log hazard ratio; AFT coefficient is log time ratio for the location parameter. Auxiliary distribution parameters are not exposure effects.'}


def unsupervised(data,spec):
    from sklearn import cluster,decomposition,manifold,mixture,cross_decomposition
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    m=spec['method']; p=spec.get('params',{}); raw=data[spec['x']].astype(float); x=raw.to_numpy(); seed=spec.get('seed',42); k=p.get('components',2)
    if m=='correspondence':
        if (x<0).any() or np.any(x.sum(axis=0)<=0) or np.any(x.sum(axis=1)<=0): raise ValueError('Correspondence requires a nonnegative contingency matrix with positive margins')
        proportions=x/x.sum(); r=proportions.sum(axis=1); c=proportions.sum(axis=0)
        u,s,v=np.linalg.svd((proportions-np.outer(r,c))/np.sqrt(np.outer(r,c)),full_matrices=False)
        coords=u[:,:k]*s[:k]/np.sqrt(r[:,None])
        return {'metrics':{'inertia':list(s*s)},'tables':{'row_coordinates':records(pd.DataFrame(coords,index=data.index)),'column_coordinates':records(pd.DataFrame(v[:k].T*s[:k]/np.sqrt(c[:,None]),index=raw.columns))}}
    if p.get('scale',True): x=StandardScaler().fit_transform(x)
    if m in {'pca','factor','mds','cca'} and k>min(x.shape): raise ValueError('Too many components')
    tables={}; metrics={}
    if m=='pca':
        model=decomposition.PCA(n_components=k,random_state=seed).fit(x); coords=model.transform(x)
        metrics['explained_variance_ratio']=list(model.explained_variance_ratio_); tables['loadings']=records(pd.DataFrame(model.components_.T,index=raw.columns))
    elif m=='factor':
        from factor_analyzer import FactorAnalyzer
        model=FactorAnalyzer(n_factors=k,method='ml',rotation='varimax' if k>1 else None).fit(x); coords=model.transform(x)
        tables['loadings']=records(pd.DataFrame(model.loadings_,index=raw.columns)); metrics['uniquenesses']=list(model.get_uniquenesses())
    elif m=='mds':
        if len(x)>1500: raise ValueError('MDS limited to 1500 cases')
        model=manifold.MDS(n_components=k,random_state=seed,n_init=2,max_iter=300).fit(x); coords=model.embedding_; metrics['stress']=model.stress_
    elif m=='cca':
        if not spec.get('endogenous'): raise ValueError('CCA requires endogenous as the second variable block')
        y=data[spec['endogenous']].astype(float)
        if k>y.shape[1]: raise ValueError('CCA components exceed second block width')
        model=cross_decomposition.CCA(n_components=k).fit(x,y); coords,cy=model.transform(x,y); metrics['canonical_correlations']=[np.corrcoef(coords[:,i],cy[:,i])[0,1] for i in range(k)]
    else:
        count=p.get('clusters',3)
        models={'kmeans':lambda:cluster.KMeans(n_clusters=count,n_init=10,random_state=seed),
                'hierarchical':lambda:cluster.AgglomerativeClustering(n_clusters=count,linkage='ward'),
                'dbscan':lambda:cluster.DBSCAN(eps=p.get('eps',.5),min_samples=p.get('min_samples',5)),
                'gmm_cluster':lambda:mixture.GaussianMixture(n_components=count,random_state=seed)}
        if m in {'hierarchical','dbscan'} and len(x)>5000: raise ValueError('Pairwise clustering limited to 5000 cases')
        model=models[m](); labels=model.fit_predict(x); metrics['clusters_found']=len(set(labels)-{-1}); metrics['noise_N']=int((labels==-1).sum())
        valid=labels!=-1
        if 1<len(set(labels[valid]))<valid.sum(): metrics['silhouette']=silhouette_score(x[valid],labels[valid],sample_size=min(1000,int(valid.sum())),random_state=seed)
        return {'metrics':metrics,'tables':{'assignments':[dict(source_row=int(i)+2,cluster=int(label)) for i,label in zip(data.index,labels)]}}
    tables['scores']=[dict(source_row=int(i)+2,**{f'component_{j+1}':v for j,v in enumerate(row)}) for i,row in zip(data.index,coords)]
    return {'tables':tables,'metrics':metrics}


def timeseries(data,spec):
    from statsmodels.tsa.stattools import adfuller,kpss,grangercausalitytests
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.api import VAR
    m=spec['method']; p=spec.get('params',{}); x=data[spec['x']].astype(float); lags=p.get('lags',5); horizon=p.get('horizon',10)
    if not spec.get('time'): raise ValueError('Explicit time column required to verify order and spacing')
    t=data[spec['time']]
    if not pd.api.types.is_numeric_dtype(t): t=pd.to_datetime(t,errors='raise').astype('int64')
    delta=np.diff(t.astype(float))
    if len(delta)==0 or (delta<=0).any() or not np.allclose(delta,delta[0]): raise ValueError('Time must be strictly increasing and equally spaced; encode a regular period index for monthly series')
    y=x.iloc[:,0]
    if m=='adf':
        r=adfuller(y,maxlag=lags,autolag='AIC'); return {'metrics':{'statistic':r[0],'p_value':r[1],'lags':r[2],'N':r[3],'critical_values':r[4]}}
    if m=='kpss':
        r=kpss(y,nlags=lags); return {'metrics':{'statistic':r[0],'p_value':r[1],'lags':r[2],'critical_values':r[3]}}
    if m=='ljung_box': return {'tables':{'portmanteau':records(acorr_ljungbox(y,lags=[lags],return_df=True))}}
    if m=='granger':
        if x.shape[1]!=2: raise ValueError('Granger requires [target, predictor] columns')
        r=grangercausalitytests(x,maxlag=lags,verbose=False)
        return {'tables':{'tests':[dict(lag=k,F=v[0]['ssr_ftest'][0],p_value=v[0]['ssr_ftest'][1]) for k,v in r.items()]},'warnings':['Multiple lag tests are unadjusted; reject causal interpretation without a design.']}
    if m=='var':
        r=VAR(x).fit(maxlags=lags); forecast=r.forecast(x.to_numpy()[-r.k_ar:],horizon)
        return {'metrics':{'aic':r.aic,'bic':r.bic,'lags':r.k_ar},'tables':{'forecast':records(pd.DataFrame(forecast,columns=x.columns))}}
    if m=='garch':
        from arch import arch_model
        r=arch_model(y,vol='GARCH',p=1,q=1,rescale=False).fit(disp='off')
        if r.convergence_flag!=0: raise ValueError('GARCH did not converge')
        return {'tables':{'coefficients':[dict(term=k,coefficient=r.params[k],std_error=r.std_err[k],p_value=r.pvalues[k]) for k in r.params.index],'forecast_variance':records(r.forecast(horizon=horizon).variance)},'metrics':{'aic':r.aic,'bic':r.bic}}
    if m=='arima':
        r=ARIMA(y.to_numpy(),order=tuple(p.get('order',[1,0,0]))).fit(); f=r.get_forecast(horizon)
        out=fitted(r,names=r.param_names,alpha=spec.get('alpha',.05)); out['tables']['forecast']=records(f.summary_frame(alpha=spec.get('alpha',.05))); return out
    r=ExponentialSmoothing(y.to_numpy(),trend='add').fit(); return {'metrics':{'aic':r.aic,'bic':r.bic},'tables':{'forecast':[dict(step=i+1,value=v) for i,v in enumerate(r.forecast(horizon))]}}
