"""Evidence-backed tables; HTML/LaTeX/RTF use the same coefficient cells."""
import html
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd


def clean(value):
    if isinstance(value,dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)): return [clean(v) for v in value]
    if isinstance(value,(np.integer,np.bool_)): return value.item()
    if isinstance(value,(float,np.floating)): return float(value) if math.isfinite(value) else None
    if value is pd.NA or value is pd.NaT: return None
    return value


def number(value):
    return f'{value:.3f}' if isinstance(value,(int,float)) and math.isfinite(value) else '—'


def star(p): return '***' if p is not None and p<.001 else '**' if p is not None and p<.01 else '*' if p is not None and p<.05 else ''


def tex(value):
    replacements={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    return ''.join(replacements.get(c,c) for c in str(value))


def rtf(value):
    output=''
    for c in str(value):
        if c in '\\{}': output+='\\'+c
        elif ord(c)<128: output+=c if c!='\n' else r'\line '
        else:
            encoded=c.encode('utf-16-le')
            for i in range(0,len(encoded),2):
                unit=int.from_bytes(encoded[i:i+2],'little'); output+=f'\\u{unit if unit<32768 else unit-65536}?'
    return output


NOTE='Coefficients; standard errors in parentheses. * p<0.05, ** p<0.01, *** p<0.001; two-sided, unadjusted. Confidence level, estimator, covariance and sample exclusions are recorded per model. Blank cells mean not estimated. Formatting follows esttab conventions; this is not Stata output.'


def esttab(results):
    models=[r for r in results if r.get('coefficients')]
    terms=list(dict.fromkeys(row['term'] for result in models for row in result['coefficients']))
    headers=['Variable']+[f'({i+1}) {r["name"]}' for i,r in enumerate(models)]
    rows=[]
    for term in terms:
        coefficient=[term]; errors=['']
        for result in models:
            row=next((v for v in result['coefficients'] if v['term']==term),None)
            coefficient.append(number(row['coefficient'])+star(row.get('p_value')) if row else '')
            errors.append('('+number(row['std_error'])+')' if row else '')
        rows.extend([coefficient,errors])
    for key in ['N','rsquared','rsquared_adj','prsquared','aic','bic','covariance']:
        if any(key in r.get('metrics',{}) for r in models):
            rows.append([key]+[str(r['metrics'][key]) if key in {'N','covariance'} and key in r['metrics'] else number(r['metrics'].get(key)) if key in r['metrics'] else '' for r in models])
    return headers,rows


def write(directory,manifest):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    results=manifest['results']; headers,rows=esttab(results)
    pd.DataFrame(rows,columns=headers).to_csv(directory/'esttab.csv',index=False)
    latex='\\begin{tabular}{l'+'c'*(len(headers)-1)+'}\n\\toprule\n'+' & '.join(map(tex,headers))+r' \\'+'\n\\midrule\n'
    latex+='\n'.join(' & '.join(map(tex,row))+r' \\' for row in rows)+'\n\\bottomrule\n\\end{tabular}\n% '+NOTE+'\n'
    (directory/'esttab.tex').write_text(latex,encoding='utf-8')
    rich=r'{\rtf1\ansi\deff0{\fonttbl{\f0 Times New Roman;}}\f0\fs21 ' +'\n'
    for index,row in enumerate([headers]+rows):
        rich+=r'\trowd\trgaph90'+(''.join((r'\clbrdrt\brdrs\brdrw15\clbrdrb\brdrs\brdrw10' if index==0 else r'\clbrdrb\brdrs\brdrw15' if index==len(rows) else '')+f'\\cellx{(i+1)*max(900,9000//len(headers))}' for i in range(len(headers))))+'\n'
        rich+=''.join(r'\intbl '+(r'\b ' if index==0 else '')+rtf(cell)+(r'\b0 ' if index==0 else '')+r'\cell ' for cell in row)+r'\row '+'\n'
    rich+=r'\pard '+rtf(NOTE)+'}'
    (directory/'esttab.rtf').write_text(rich,encoding='ascii')
    style='body{font:16px Georgia,serif;max-width:1100px;margin:40px auto;padding:24px;color:#17212b}table{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0;border-top:2px solid;border-bottom:2px solid}th{border-bottom:1px solid;text-align:left}td,th{padding:7px 10px;border:0}thead th{border-bottom:1px solid}h1{font-size:28px}p{line-height:1.65}.table-wrap{overflow-x:auto}td:not(:first-child){text-align:right}pre{white-space:pre-wrap;font:13px monospace}.note{color:#495260;font-size:13px}img{max-width:100%}'
    fragments=[f'<!doctype html><meta charset="utf-8"><title>THETA empirical analysis</title><style>{style}</style><h1>统计分析报告</h1><p>{html.escape(manifest["question"])}</p>', '<p class="note">'+html.escape(NOTE)+'</p>']
    if len(headers)>1: fragments.append(pd.DataFrame(rows,columns=headers).to_html(index=False,escape=True,border=0,float_format=lambda v:format(v,'.4g')))
    markdown=['# '+manifest['question'],'',NOTE,'',f'Dataset SHA-256: `{(manifest.get("dataset") or {}).get("sha256", "not applicable: data-free mathematical specification")}`','']
    for i,result in enumerate(results):
        prefix=f'{i+1:02d}-{result["method"]}'
        fragments.append('<h2>'+html.escape(result['name'])+'</h2>')
        markdown+=['## '+result['name'],'',f'Method: `{result["method"]}`; input N={result["sample"]["input_N"]}, complete/eligible N={result["sample"]["eligible_N"]}, excluded N={result["sample"]["excluded_N"]}.', '',result.get('estimand','Evidence is conditional on the method assumptions and selected sample.'),'']
        summary=json.dumps(result.get('metrics',{}),ensure_ascii=False,indent=2)
        fragments.append('<p class="note">'+html.escape(f"Input N={result['sample']['input_N']}; eligible N={result['sample']['eligible_N']}; excluded N={result['sample']['excluded_N']}")+'</p>')
        fragments.append('<pre>'+html.escape(summary)+'</pre>'); markdown+=['```json',summary,'```','']
        for warning in result.get('warnings',[]):
            fragments.append('<p class="note">'+html.escape(warning)+'</p>'); markdown+=['- '+warning]
        for j,(title,table) in enumerate(result.get('tables',{}).items()):
            name=f'{prefix}-table-{j+1:02d}.csv'; pd.DataFrame(table).to_csv(directory/name,index=False)
            fragments.append(f'<h3>{html.escape(title)}</h3><p><a href="{name}">完整 CSV</a></p>'+pd.DataFrame(table[:20]).to_html(index=False,escape=True,border=0,float_format=lambda v:format(v,'.4g')))
            markdown+=['',f'[{title} — full table]({name})', '',pd.DataFrame(table[:10]).to_markdown(index=False),'']
        if result.get('coefficients'):
            pd.DataFrame(result['coefficients']).to_csv(directory/f'{prefix}-coefficients.csv',index=False)
        figure=plot(directory,prefix,result)
        if figure:
            fragments.append(f'<figure><img src="{figure}.svg"><figcaption>'+html.escape(result['figure_caption'])+'</figcaption></figure>')
            markdown+=['',f'![{result["figure_caption"]}]({figure}.svg)','',result['figure_caption'],'']
        diagnostic=diagnostics(directory,prefix,result)
        if diagnostic:
            file,caption=diagnostic;result['diagnostic_caption']=caption
            fragments.append(f'<figure><img src="{file}.svg"><figcaption>'+html.escape(caption)+'</figcaption></figure>')
            markdown+=['',f'![Residual diagnostics]({file}.svg)','',caption,'']
    limitations='Interpretation contract: report the direction, magnitude, units, interval and denominator of each substantive finding; discuss assumptions, alternative explanations, multiplicity, uncertainty and external validity. Nonsignificance is not equivalence; prediction is not causation; exploratory same-data checks are not independent confirmation.'
    fragments.append('<p class="note">'+html.escape(limitations)+'</p>'); markdown+=['',limitations]
    (directory/'report.html').write_text('\n'.join(fragments),encoding='utf-8')
    (directory/'report.md').write_text('\n'.join(markdown),encoding='utf-8')
    (directory/'results.json').write_text(json.dumps(clean(manifest),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    (directory/'analysis-spec.json').write_text(json.dumps(manifest['plan'],ensure_ascii=False,indent=2),encoding='utf-8')
    return sorted(p for p in directory.iterdir() if p.is_file())


def plot(directory,prefix,result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fonts={font.name for font in font_manager.fontManager.ttflist}
    family=next((name for name in ['Noto Sans CJK SC','Arial Unicode MS','PingFang SC','Heiti SC'] if name in fonts),'DejaVu Sans')
    plt.rcParams.update({'font.family':family,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    coefficients=[r for r in result.get('coefficients',[]) if r['coefficient'] is not None and r.get('ci_low') is not None and r.get('ci_high') is not None]
    tables=result.get('tables',{}); roc=tables.get('roc'); forecast=tables.get('forecast'); scores=tables.get('ranking'); curves=[(k,v) for k,v in tables.items() if k.endswith('_curve')]
    if not any([coefficients,roc,forecast,scores,curves]): return None
    fig,ax=plt.subplots(figsize=(7,max(3,min(len(coefficients),40)*.27)))
    if coefficients:
        c=coefficients[:40]; y=np.arange(len(c)); values=np.array([r['coefficient'] for r in c]); lo=np.array([r['ci_low'] for r in c]); hi=np.array([r['ci_high'] for r in c]); ax.hlines(y,lo,hi,color='#244f78'); ax.scatter(values,y,color='#244f78',s=18); ax.set_yticks(y,[r['term'] for r in c]); ax.axvline(0,color='#888888',linestyle='--',linewidth=.8); ax.invert_yaxis(); ax.set_xlabel('Coefficient and confidence interval')
        confidence=100*(1-result['spec'].get('alpha',.05)); caption=f'Horizontal axis: coefficient; vertical axis: model term. Points are estimated coefficients; horizontal bars are {confidence:g}% confidence intervals under the recorded covariance estimator. The dashed line is zero. Scale follows the estimator; intervals do not establish causal identification. First {len(c)} of {len(coefficients)} terms shown.'
    elif roc:
        ax.plot([r['false_positive_rate'] for r in roc],[r['true_positive_rate'] for r in roc]); ax.plot([0,1],[0,1],'--',color='gray'); ax.set(xlabel='False positive rate',ylabel='True positive rate'); caption=f'Held-out ROC curve; AUC={result["metrics"]["roc_auc"]:.3f}. Positive class: {result["metrics"]["positive_class"]}. This single test partition does not quantify external validation uncertainty.'
    elif curves:
        for name,rows in curves:
            frame=pd.DataFrame(rows); ax.step(frame.iloc[:,0],frame.iloc[:,1],where='post',label=name.removesuffix('_curve'))
            if frame.shape[1]>=4: ax.fill_between(frame.iloc[:,0],frame.iloc[:,2],frame.iloc[:,3],alpha=.12,step='post')
        ax.legend(); ax.set(xlabel='Duration',ylabel='Survival probability' if result['method']=='kaplan_meier' else 'Cumulative hazard'); caption='Step estimates and pointwise confidence bands for right-censored data. Consult the accompanying at-risk table; tail estimates with few participants are unstable. Curves alone are not an adjusted group-effect test.'
    elif scores:
        rows=scores[:40]; ax.bar(range(len(rows)),[r['score'] for r in rows]); ax.set(xlabel='Alternative (input order)',ylabel='Decision score'); caption=f'Deterministic scores under the declared criterion directions and weights; these are not probabilities or significance tests. First {len(rows)} alternatives shown.'
    else:
        frame=pd.DataFrame(forecast); numeric=frame.select_dtypes('number'); series=numeric['mean'] if 'mean' in numeric else numeric.iloc[:,-1]; ax.plot(range(1,len(series)+1),series); ax.set(xlabel='Forecast horizon',ylabel='Forecast'); caption='Conditional forecast from the specified series and fitted model; no held-out forecast accuracy is implied. Consult the full forecast table for available intervals.'
    result['figure_caption']=caption
    fig.tight_layout()
    for suffix in ('svg','pdf','png'): fig.savefig(directory/f'{prefix}.{suffix}',dpi=300,bbox_inches='tight')
    plt.close(fig); return prefix


def diagnostics(directory,prefix,result):
    rows=result.get('tables',{}).get('fitted')
    if not rows:return None
    import matplotlib.pyplot as plt
    from scipy import stats
    frame=pd.DataFrame(rows);residual=frame['residual'].to_numpy();fitted=frame['fitted'].to_numpy()
    fig,axes=plt.subplots(1,2,figsize=(8,3.2))
    axes[0].scatter(fitted,residual,s=12,alpha=.55,color='#244f78');axes[0].axhline(0,color='gray',linestyle='--',linewidth=.8);axes[0].set(xlabel='Fitted outcome',ylabel='Residual')
    (theoretical,ordered),(slope,intercept,_)=stats.probplot(residual,dist='norm')
    axes[1].scatter(theoretical,ordered,s=12,alpha=.55,color='#244f78');axes[1].plot(theoretical,slope*theoretical+intercept,color='gray',linewidth=.8);axes[1].set(xlabel='Theoretical normal quantile',ylabel='Ordered residual')
    fig.tight_layout();name=prefix+'-diagnostics'
    for suffix in ('svg','pdf','png'):fig.savefig(Path(directory)/f'{name}.{suffix}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    return name,'Residual–fitted plot checks patterns and changing dispersion; Q–Q plot checks tail/shape departures. These are visual diagnostics, not proof of independence or causal identification. HC3 addresses heteroskedastic standard errors, not misspecified conditional means or confounding.'
