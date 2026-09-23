"""Transparent numeric decision models and bounded mathematical programming."""
import numpy as np
import pandas as pd
from scipy import optimize,stats
from .inference import records


def weights(values,n):
    w=np.ones(n)/n if values is None else np.asarray(values,dtype=float)
    if w.shape!=(n,) or (w<0).any() or not np.isfinite(w).all() or w.sum()<=0: raise ValueError('Need nonnegative weights of matching width with positive sum')
    return w/w.sum()


def run(data,spec):
    m=spec['method']; p=spec.get('params',{}); x=data[spec['x']].astype(float).to_numpy(); n,k=x.shape
    if m=='ahp':
        if n!=k or not 2<=n<=10 or (x<=0).any() or not np.allclose(x*x.T,1,rtol=.01) or not np.allclose(np.diag(x),1): raise ValueError('AHP requires a positive reciprocal 2..10 square judgment matrix with unit diagonal')
        vals,vecs=np.linalg.eig(x); i=int(np.argmax(vals.real)); maximum=vals[i].real; w=np.abs(vecs[:,i].real); w/=w.sum(); ci=(maximum-n)/(n-1)
        ri=[0,0,0,.58,.90,1.12,1.24,1.32,1.41,1.45,1.49][n]
        return {'metrics':{'lambda_max':maximum,'CI':ci,'CR':ci/ri if ri else 0},'tables':{'weights':[dict(variable=c,weight=v) for c,v in zip(spec['x'],w)]},'warnings':['CR < 0.10 is a conventional consistency heuristic, not inferential significance.']}
    if m=='coupling':
        if (x<0).any() or (x>1).any(): raise ValueError('Coupling requires prespecified dimensionless subsystem scores within [0,1]')
        w=weights(p.get('weights'),k); mean=x.mean(axis=1); c=np.divide(np.prod(x,axis=1)**(1/k),mean,out=np.zeros(n),where=mean>0); t=x@w; score=np.sqrt(c*t)
        return {'tables':{'scores':[dict(source_row=int(i)+2,coupling=a,development=b,coordination=d) for i,a,b,d in zip(data.index,c,t,score)]}}
    if n<2 or (np.ptp(x,axis=0)<=0).any(): raise ValueError('Decision metrics need >=2 alternatives and nonconstant criteria')
    cost=p.get('cost_columns',[])
    if not isinstance(cost,list) or not set(cost)<=set(spec['x']): raise ValueError('cost_columns must be selected criteria')
    z=(x-x.min(axis=0))/np.ptp(x,axis=0)
    for j,c in enumerate(spec['x']):
        if c in cost: z[:,j]=1-z[:,j]
    if m in {'entropy_weight','critic','cv_weight'}:
        if m=='entropy_weight':
            q=z/z.sum(axis=0); terms=np.zeros_like(q); np.log(q,out=terms,where=q>0); w=1+(q*terms).sum(axis=0)/np.log(n)
        elif m=='critic': w=z.std(axis=0,ddof=1)*(1-np.corrcoef(z,rowvar=False)).sum(axis=0)
        else:
            if (x.mean(axis=0)<=0).any(): raise ValueError('CV weights require positive nonzero means')
            w=x.std(axis=0,ddof=1)/x.mean(axis=0)
        w=weights(w,k)
        return {'tables':{'weights':[dict(variable=c,weight=v) for c,v in zip(spec['x'],w)]}}
    w=weights(p.get('weights'),k)
    if m=='topsis':
        norm=np.linalg.norm(x,axis=0)
        if (norm<=0).any(): raise ValueError('Zero vector norm')
        v=x/norm*w; best=v.max(axis=0); worst=v.min(axis=0)
        for j,c in enumerate(spec['x']):
            if c in cost: best[j],worst[j]=worst[j],best[j]
        plus=np.linalg.norm(v-best,axis=1); minus=np.linalg.norm(v-worst,axis=1); score=minus/(plus+minus)
    elif m=='rsr': score=np.column_stack([stats.rankdata(z[:,j],method='average') for j in range(k)])@w/n
    elif m=='vikor':
        regret=(1-z)*w; S=regret.sum(axis=1); R=regret.max(axis=1); v=p.get('v',.5)
        score=v*(S-S.min())/(np.ptp(S) or 1)+(1-v)*(R-R.min())/(np.ptp(R) or 1)
    elif m=='grey_relation':
        delta=1-z; score=((delta.min()+.5*delta.max())/(delta+.5*delta.max()))@w
    else: raise ValueError(m)
    ranks=stats.rankdata(score if m=='vikor' else -score,method='min')
    return {'tables':{'ranking':[dict(source_row=int(i)+2,score=s,rank=int(r)) for i,s,r in zip(data.index,score,ranks)]},'metrics':{'weights':list(w),'preferred_direction':'lower' if m=='vikor' else 'higher'}}


def programming(spec):
    p=spec.get('params',{}); m=spec['method']; c=np.asarray(p.get('c',[]),dtype=float); n=len(c)
    if c.ndim!=1 or not 1<=n<=100: raise ValueError('c must be a vector with 1..100 variables')
    def constraints(matrix,rhs):
        a,b=p.get(matrix),p.get(rhs)
        if (a is None)!=(b is None): raise ValueError('Constraint matrix and right side must be supplied together')
        if a is None: return None,None
        a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
        if a.ndim!=2 or a.shape[1]!=n or b.shape!=(len(a),) or len(a)>500: raise ValueError('Invalid constraint dimensions')
        return a,b
    au,bu=constraints('a_ub','b_ub'); ae,be=constraints('a_eq','b_eq')
    bounds=p.get('bounds',[[0,None]]*n)
    if len(bounds)!=n or any(not isinstance(pair,list) or len(pair)!=2 for pair in bounds): raise ValueError('One bound pair per variable required')
    lower=np.array([-np.inf if pair[0] is None else pair[0] for pair in bounds],dtype=float); upper=np.array([np.inf if pair[1] is None else pair[1] for pair in bounds],dtype=float)
    if (lower>upper).any(): raise ValueError('Inconsistent bounds')
    if m=='linear_program': r=optimize.linprog(c,A_ub=au,b_ub=bu,A_eq=ae,b_eq=be,bounds=bounds,method='highs',options={'time_limit':30})
    elif m=='mixed_integer_program':
        integers=p.get('integrality',[1]*n)
        if len(integers)!=n or any(type(v) is not int or v not in (0,1) for v in integers): raise ValueError('integrality must be 0/1 vector')
        cons=[]
        if au is not None: cons.append(optimize.LinearConstraint(au,-np.inf,bu))
        if ae is not None: cons.append(optimize.LinearConstraint(ae,be,be))
        r=optimize.milp(c,integrality=integers,bounds=optimize.Bounds(lower,upper),constraints=cons,options={'time_limit':30,'node_limit':10000})
    else:
        q=np.asarray(p.get('q'),dtype=float)
        if q.shape!=(n,n) or not np.allclose(q,q.T) or np.linalg.eigvalsh(q).min() < -1e-8: raise ValueError('Q must be symmetric positive semidefinite; objective .5*xᵀQx+cᵀx')
        cons=[]
        if au is not None: cons.append(optimize.LinearConstraint(au,-np.inf,bu))
        if ae is not None: cons.append(optimize.LinearConstraint(ae,be,be))
        start=np.clip(np.zeros(n),lower,upper)
        r=optimize.minimize(lambda x:.5*x@q@x+c@x,start,jac=lambda x:q@x+c,method='SLSQP',bounds=optimize.Bounds(lower,upper),constraints=cons,options={'maxiter':1000,'ftol':1e-10})
    if not r.success: raise ValueError(f'Optimization not certified successful: {r.message}')
    if (au is not None and (au@r.x>bu+1e-6).any()) or (ae is not None and not np.allclose(ae@r.x,be,atol=1e-6)): raise ValueError('Solver returned infeasible constraints')
    return {'metrics':{'objective':r.fun,'status':str(r.message),'success':True},'tables':{'solution':[dict(variable=f'x{i+1}',value=v) for i,v in enumerate(r.x)]}}


def preprocess(data,spec):
    from sklearn.preprocessing import StandardScaler,MinMaxScaler
    m=spec['method']; p=spec.get('params',{}); d=data[spec['x']].copy()
    if m=='dummy': d=pd.get_dummies(d,drop_first=True,dtype=int)
    else:
        d=d.astype(float)
        if m in {'standardize','minmax'}:
            values=(StandardScaler() if m=='standardize' else MinMaxScaler()).fit_transform(d); d=pd.DataFrame(values,columns=d.columns,index=d.index)
        elif m=='winsorize': d=d.clip(lower=d.quantile(p.get('tail',.01)),upper=d.quantile(1-p.get('tail',.01)),axis=1)
        elif m=='impute_median':
            if d.median().isna().any(): raise ValueError('Cannot median-impute an entirely missing column')
            d=d.fillna(d.median())
        elif m=='outliers_iqr':
            q1,q3=d.quantile(.25),d.quantile(.75); d=((d<q1-1.5*(q3-q1))|(d>q3+1.5*(q3-q1))).astype(int)
        elif m=='rolling_mean': d=d.rolling(p.get('window',3),min_periods=p.get('window',3)).mean()
        else: raise ValueError(m)
    d.insert(0,'source_row',data.index+2)
    return {'tables':{'derived_data':d.to_dict('records')},'warnings':['Derived dataset only. No source mutation. Refit preprocessing inside training folds for prediction.']}
