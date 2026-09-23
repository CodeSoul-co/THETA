"""Factorial designs, empirical sensitivity tools, SEM, power and association rules."""
import itertools
import re
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from .inference import fitted,records


def least_squares(y,x):
    matrix=np.column_stack([np.ones(len(x)),x])
    if np.linalg.matrix_rank(matrix)<matrix.shape[1]: raise ValueError('Rank-deficient empirical design')
    return np.linalg.lstsq(matrix,y,rcond=None)[0]


def run(data,spec):
    m=spec['method'];p=spec.get('params',{});alpha=spec.get('alpha',.05);cols=spec.get('x',[])
    if m in {'anova_factorial','ancova','repeated_anova','manova'}:
        import statsmodels.formula.api as smf
        factors=spec.get('factors',[])
        if not factors or len(factors)>4: raise ValueError('One to four categorical factors required')
        if set(factors)&set(cols): raise ValueError('Numeric covariates and categorical factors must be distinct')
        d=pd.DataFrame(index=data.index);terms=[]
        for i,c in enumerate(factors):
            if not 2<=data[c].nunique()<=10: raise ValueError('Each factor needs 2..10 observed levels')
            d[f'f{i}']=data[c];terms.append(f'C(f{i})')
        for i,c in enumerate(cols):d[f'x{i}']=data[c].astype(float)
        rhs=' * '.join(terms)+''.join(f' + x{i}' for i in range(len(cols)))
        if m=='manova':
            from statsmodels.multivariate.manova import MANOVA
            outcomes=spec.get('endogenous',[])
            if len(outcomes)<2:raise ValueError('MANOVA requires >=2 endogenous outcome columns')
            for i,c in enumerate(outcomes):d[f'y{i}']=data[c].astype(float)
            result=MANOVA.from_formula(' + '.join(f'y{i}' for i in range(len(outcomes)))+' ~ '+rhs,d).mv_test()
            return {'tables':{name:records(v['stat']) for name,v in result.results.items()},'metrics':{'factor_mapping':dict(zip([f'f{i}' for i in range(len(factors))],factors))}}
        if not spec.get('y'):raise ValueError('y required')
        d['y']=data[spec['y']].astype(float)
        if m=='repeated_anova':
            from statsmodels.stats.anova import AnovaRM
            if not spec.get('entity'):raise ValueError('Repeated ANOVA requires entity identifier')
            if cols:raise ValueError('Repeated ANOVA covariates are not supported')
            d['subject']=data[spec['entity']]
            r=AnovaRM(d,'y','subject',within=[f'f{i}' for i in range(len(factors))]).fit()
            return {'tables':{'ANOVA':records(r.anova_table)},'warnings':['Balanced fully within-subject design; no Greenhouse–Geisser correction. For >2 levels, sphericity remains an untested assumption.']}
        model=smf.ols('y ~ '+rhs,d).fit()
        if np.linalg.matrix_rank(model.model.exog)<model.model.exog.shape[1]:raise ValueError('Empty/confounded design cells: requested ANOVA effects are not estimable')
        table=sm.stats.anova_lm(model,typ=2);residual=table.loc['Residual','sum_sq'];table['partial_eta_squared']=table.sum_sq/(table.sum_sq+residual);table.loc['Residual','partial_eta_squared']=np.nan
        return {'tables':{'ANOVA_type_II':records(table)},'metrics':{'N':model.nobs,'rsquared':model.rsquared,'factor_mapping':dict(zip([f'f{i}' for i in range(len(factors))],factors))},'warnings':['Type II sums of squares; interpret main effects cautiously when interactions are present. Observational group contrasts do not imply random assignment.']}
    if m=='icc':
        x=data[cols].astype(float).to_numpy();n,k=x.shape
        if n<3 or k<2:raise ValueError('Rows are subjects, columns are raters; need >=3 subjects and >=2 raters')
        grand=x.mean();ssr=k*((x.mean(axis=1)-grand)**2).sum();ssc=n*((x.mean(axis=0)-grand)**2).sum();sse=((x-x.mean(axis=1)[:,None]-x.mean(axis=0)+grand)**2).sum()
        msr,msc,mse=ssr/(n-1),ssc/(k-1),sse/((n-1)*(k-1))
        return {'metrics':{'ICC_2_1_absolute':(msr-mse)/(msr+(k-1)*mse+k*(msc-mse)/n),'ICC_3_1_consistency':(msr-mse)/(msr+(k-1)*mse),'subjects':n,'raters':k},'warnings':['Balanced complete ratings only; point estimates do not include interval uncertainty.']}
    if m=='moderation':
        if len(cols)!=2 or not spec.get('y'):raise ValueError('Moderation requires x=[exposure, moderator] and y')
        x=data[cols].astype(float).copy();x['interaction']=x.iloc[:,0]*x.iloc[:,1];r=sm.OLS(data[spec['y']],sm.add_constant(x)).fit(cov_type='HC3');out=fitted(r,alpha=alpha);out['warnings']=['Uncentered linear product interaction; interpret exposure effect at moderator=0.'];return out
    if m in {'mediation','parallel_mediation','chain_mediation'}:
        mediators=spec.get('endogenous',[])
        if len(cols)!=1 or not mediators or not spec.get('y'):raise ValueError('Mediation requires one exposure x, mediator columns endogenous, and outcome y')
        if m=='mediation' and len(mediators)!=1:raise ValueError('Single mediation requires exactly one mediator')
        if m=='chain_mediation' and len(mediators)<2:raise ValueError('Chain mediation requires >=2 ordered mediators')
        if len(mediators)>4:raise ValueError('At most four mediators')
        x=data[cols].to_numpy(float);M=data[mediators].to_numpy(float);y=data[spec['y']].to_numpy(float)
        def effects(index):
            xx,mm,yy=x[index],M[index],y[index]
            a=[least_squares(mm[:,j],np.column_stack([xx,mm[:,:j]]) if m=='chain_mediation' else xx) for j in range(mm.shape[1])]
            outcome=least_squares(yy,np.column_stack([xx,mm]))
            if m=='chain_mediation':return np.array([a[0][1]*np.prod([a[j][-1] for j in range(1,len(a))])*outcome[-1]])
            return np.array([coef[1]*outcome[j+2] for j,coef in enumerate(a)])
        n=len(data);point=effects(np.arange(n));rng=np.random.default_rng(spec.get('seed',42));b=p.get('bootstrap',500);boot=np.array([effects(rng.integers(0,n,n)) for _ in range(b)]);ci=np.quantile(boot,[alpha/2,1-alpha/2],axis=0)
        return {'tables':{'indirect_effects':[dict(path=' -> '.join(mediators) if m=='chain_mediation' else mediators[j],effect=v,ci_low=ci[0,j],ci_high=ci[1,j]) for j,v in enumerate(point)]},'metrics':{'bootstrap_resamples':b,'N':n},'warnings':['Percentile bootstrap, independent rows, linear models, no unmeasured exposure–mediator/outcome confounding assumed. Chain reports the full ordered chain only, not every shorter indirect path.']}
    if m=='psm':
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.neighbors import NearestNeighbors
        if not spec.get('group') or not spec.get('y') or not cols:raise ValueError('PSM needs group=treatment, y=outcome, x=pre-treatment confounders')
        t=data[spec['group']].to_numpy();y=data[spec['y']].to_numpy(float);x=data[cols].to_numpy(float)
        if set(t)!={0,1}:raise ValueError('Treatment must be binary 0/1')
        propensity=make_pipeline(StandardScaler(),LogisticRegression(C=1e6,max_iter=1000)).fit(x,t).predict_proba(x)[:,1]
        logits=np.log(np.clip(propensity,1e-8,1-1e-8)/(1-np.clip(propensity,1e-8,1-1e-8)));treated=np.flatnonzero(t==1);control=np.flatnonzero(t==0)
        distance,match=NearestNeighbors(n_neighbors=1).fit(logits[control,None]).kneighbors(logits[treated,None]);keep=distance[:,0]<=p.get('caliper',.2)*logits.std(ddof=1)
        ti=treated[keep];co=control[match[:,0][keep]]
        if len(ti)<2:raise ValueError('Fewer than two treated units matched within caliper')
        denom=np.sqrt((x[t==1].var(axis=0,ddof=1)+x[t==0].var(axis=0,ddof=1))/2)
        before=(x[t==1].mean(axis=0)-x[t==0].mean(axis=0))/denom;after=(x[ti].mean(axis=0)-x[co].mean(axis=0))/denom
        return {'metrics':{'ATT_matched':np.mean(y[ti]-y[co]),'matched_treated':len(ti),'unmatched_treated':len(treated)-len(ti),'replacement':True},'tables':{'balance':[dict(variable=c,SMD_before=a,SMD_after=b) for c,a,b in zip(cols,before,after)],'matches':[dict(treated_row=int(data.index[i])+2,control_row=int(data.index[j])+2) for i,j in zip(ti,co)]},'warnings':['Nearest neighbor on logit propensity, with replacement and a SD-based caliper. ATT applies to matched treated units only; no valid matching SE/CI is implemented. Examine overlap/balance; hidden confounding remains possible.']}
    if m=='sem':
        import semopy
        model=p.get('model','')
        if not isinstance(model,str) or len(model)>3000:raise ValueError('Bounded SEM model string required')
        # Only named linear relations/covariances/loadings; no expressions, commands, paths or function calls.
        lines=[]
        for line in model.splitlines():
            if not re.fullmatch(r'\s*[A-Za-z][A-Za-z0-9_]*\s*(?:=~|~~|~)\s*[A-Za-z][A-Za-z0-9_]*(?:\s*\+\s*[A-Za-z][A-Za-z0-9_]*)*\s*',line):raise ValueError('SEM supports identifier-only ~, ~~, =~ relations; no constraints or expression evaluation')
            lines.append(line)
        if not lines:raise ValueError('SEM model required')
        observed=set(re.findall(r'[A-Za-z][A-Za-z0-9_]*',model))-set(re.findall(r'^\s*([A-Za-z][A-Za-z0-9_]*)\s*=~',model,re.M))
        if not observed<=set(cols):raise ValueError('All observed SEM variables must be explicitly listed in x')
        estimator=semopy.Model('\n'.join(lines));r=estimator.fit(data[cols].astype(float),obj='MLW')
        if not r.success:raise ValueError('SEM optimization did not converge')
        params=estimator.inspect(std_est=True);fit=semopy.calc_stats(estimator)
        if fit['DoF'].iloc[0]<=0:raise ValueError('SEM fit assessment requires positive degrees of freedom; simplify the identified model')
        return {'tables':{'SEM_parameters':params.to_dict('records'),'fit_indices':records(fit)},'warnings':['MLW assumes suitable continuous indicators; ordered categorical WLSMV, invariance and multilevel SEM are not provided. Global fit cannot prove the proposed causal structure.']}
    if m=='apriori':
        from mlxtend.frequent_patterns import apriori,association_rules
        if not cols or len(cols)>25 or not np.isin(data[cols],[0,1,False,True]).all():raise ValueError('Apriori requires <=25 binary item columns')
        items=apriori(data[cols].astype(bool),min_support=p.get('min_support',.2),use_colnames=True,max_len=3)
        if items.empty:return {'tables':{'rules':[]},'metrics':{'frequent_itemsets':0}}
        rules=association_rules(items,metric='confidence',min_threshold=p.get('min_confidence',.6))
        for c in ['antecedents','consequents']:rules[c]=rules[c].map(lambda v:', '.join(sorted(v)))
        return {'tables':{'rules':rules.to_dict('records')},'metrics':{'frequent_itemsets':len(items)},'warnings':['Max itemset length 3; associations are descriptive and selected from many candidates. No causal or confirmatory significance interpretation.']}
    if m=='power_ttest':
        from statsmodels.stats.power import TTestIndPower
        effect=p.get('effect_size',.5);ratio=p.get('ratio',1);nobs=p.get('nobs');power=p.get('power',.8)
        if nobs is None: n=TTestIndPower().solve_power(effect_size=effect,alpha=alpha,power=power,ratio=ratio);return {'metrics':{'N_group1':int(np.ceil(n)),'N_group2':int(np.ceil(n*ratio)),'target_power':power,'effect_size':effect,'alpha':alpha}}
        return {'metrics':{'power':TTestIndPower().power(effect,nobs,alpha,ratio=ratio),'N_group1':nobs,'effect_size':effect,'alpha':alpha}}
    raise ValueError(m)
