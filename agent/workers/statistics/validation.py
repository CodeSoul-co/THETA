"""Bounded, explicit contracts shared by discovery, planning and execution."""
from __future__ import annotations
import math
from .registry import METHODS

PARAMETERS = {
 'bootstrap': [100,2000,500], 'caliper':[.01,2,.2], 'model':'Identifier-only linear SEM relations ~, ~~, =~; no formulas/functions', 'min_support':[.05,1,.2], 'min_confidence':[.1,1,.6], 'effect_size':[.01,5,.5], 'power':[.5,.999,.8], 'nobs':[3,100000,100], 'ratio':[.1,10,1],
 'correlation': ['pearson', 'spearman', 'kendall'], 'covariance': ['nonrobust', 'HC1', 'HC3', 'cluster', 'HAC'],
 'mu': [-1e12, 1e12, 0], 'low': [-1e12, 1e12, -0.5], 'high': [-1e12, 1e12, 0.5],
 'probability': [0.000001, .999999, .5], 'quantile': [.01, .99, .5], 'max_lags': [1, 40, 1],
 'cutoff': [-1e12, 1e12, 0], 'bandwidth': [.000001, 1e12, 1], 'components': [1, 20, 2],
 'clusters': [2, 20, 3], 'eps': [.000001, 1e6, .5], 'min_samples': [2, 1000, 5],
 'scale': [True, False], 'lags': [1, 40, 5], 'horizon': [1, 100, 10],
 'p_adjust': ['holm', 'bonferroni', 'fdr_bh'], 'test_size': [.1, .4, .25], 'cv': [0, 5, 0],
 'search': ['grid', 'random'], 'split': ['random', 'group', 'time'], 'tail': [.001, .2, .01],
 'window': [2, 1000, 3], 'v': [0, 1, .5],
 'weights': 'nonnegative numeric vector', 'cost_columns': 'list of x column names',
 'order': '[p,d,q], 0<=p,q<=5, 0<=d<=2', 'text_column': 'one exact text column name; optional TF-IDF feature',
 'tune': 'false or {} disables tuning; otherwise object mapping allowlisted estimator parameter -> 1..4 values; <=12 combinations, cv=2..5 required',
 'c': 'objective vector, 1..100 variables', 'q': 'symmetric positive semidefinite matrix',
 'a_ub': 'matrix A: A*x <= b', 'b_ub': 'vector b', 'a_eq': 'matrix A: A*x = b', 'b_eq': 'vector b',
 'bounds': 'one [lower,upper] pair per variable; null means unbounded; default x>=0',
 'integrality': '0 (continuous) or 1 (integer) per variable',
}
INTEGER = {'bootstrap','nobs','max_lags', 'components', 'clusters', 'min_samples', 'lags', 'horizon', 'cv', 'window'}
FIELDS = {'method', 'name', 'x', 'y', 'group', 'time', 'event', 'entity', 'weight', 'endogenous', 'instruments', 'factors', 'params', 'missing', 'alpha', 'seed'}


def parameter_contract(method): return {k: PARAMETERS[k] for k in method['parameters']}


def example(method):
    spec = {'method': method['id'], 'params': {}, 'missing': 'drop', 'alpha': .05, 'seed': 42}
    for key in method['required']: spec[key] = ['x', 'x2'] if key == 'x' else [key] if key in {'factors','endogenous','instruments'} else key
    if method['family'] == 'optimization': spec['params'] = {'c': [-1, -2], 'a_ub': [[1, 1]], 'b_ub': [5]}
    if method['id'] == 'wls': spec['weight'] = 'weight'
    if method['id'] in {'mixed_linear', 'gee'}: spec['group'] = 'group'
    if method['id'].startswith('panel_'): spec.update(entity='id', time='wave')
    if method['id'].startswith('iv_'): spec.update(endogenous=['endog'], instruments=['instrument'])
    if method['id'] == 'did': spec.update(group='treated', time='post')
    if method['id'] == 'rdd': spec['params'].update(cutoff=0, bandwidth=1)
    if method['id'] == 'logrank': spec['group'] = 'group'
    if method['family'] == 'survival': spec['time'] = 'duration'
    if method['id'] == 'quadratic_program': spec['params']['q'] = [[2,0],[0,2]]
    if method['id'] == 'sem': spec.update(x=['item0','item1','item2','item3','item4'], params={'model':'F =~ item0 + item1 + item2 + item3 + item4'})
    return spec


def finite(value):
    if isinstance(value, dict): return all(finite(v) for v in value.values())
    if isinstance(value, list): return all(finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def validate(spec, columns):
    if not isinstance(spec, dict) or set(spec) - FIELDS: raise ValueError('Unknown analysis fields')
    method = METHODS.get(spec.get('method'))
    if not method: raise ValueError(f"未实现方法: {spec.get('method')}")
    if not finite(spec): raise ValueError('Parameters must be finite')
    for field in method['required']:
        if not spec.get(field): raise ValueError(f'{method["id"]} requires {field}')
    for field in ('x', 'endogenous', 'instruments', 'factors'):
        if field in spec:
            value = spec[field]
            if not isinstance(value, list) or not 1 <= len(value) <= 40 or len(set(value)) != len(value): raise ValueError(f'Invalid {field}')
            if any(not isinstance(v, str) or v not in columns for v in value): raise ValueError(f'Unknown {field} column')
    for field in ('y', 'group', 'time', 'event', 'entity', 'weight'):
        if field in spec and spec[field] not in columns: raise ValueError(f'Unknown {field} column')
    if spec.get('y') in spec.get('x', []): raise ValueError('Outcome must not also be a predictor (target leakage)')
    if spec.get('missing', 'drop') not in {'drop', 'error'}: raise ValueError('missing must be drop or error')
    if not .001 <= spec.get('alpha', .05) <= .2: raise ValueError('Invalid alpha')
    if type(spec.get('seed', 42)) is not int or not 0 <= spec.get('seed', 42) <= 2147483647: raise ValueError('Invalid seed')
    params = spec.get('params', {})
    if not isinstance(params, dict) or set(params) - set(method['parameters']): raise ValueError('Unknown method parameter; inspect the method contract')
    for key, value in params.items():
        contract = PARAMETERS[key]
        if isinstance(contract, list):
            if len(contract) == 3 and type(contract[0]) in (float, int):
                if type(value) not in (float, int) or not contract[0] <= value <= contract[1]: raise ValueError(f'Invalid {key}')
                if key in INTEGER and type(value) is not int: raise ValueError(f'{key} must be integer')
            elif value not in contract: raise ValueError(f'Invalid {key}')
    if method['family']=='ml':
        from .prediction import estimator, search_grid
        grid=search_grid(params.get('tune',{}),estimator(method['id'],spec.get('seed',42)))
        folds=params.get('cv',0)
        if folds==1: raise ValueError('cv is 0 (off) or 2..5')
        if grid and folds<2: raise ValueError('Tuning requires train-only CV >=2')
    if 'text_column' in params and (params['text_column'] not in columns or params['text_column'] == spec.get('y') or params['text_column'] in spec.get('x', [])): raise ValueError('text_column must be a distinct input column')
    if 'order' in params:
        order = params['order']
        if not isinstance(order, list) or len(order) != 3 or any(type(v) is not int or not 0 <= v <= lim for v, lim in zip(order, [5,2,5])): raise ValueError('Invalid ARIMA order')
    for field in ('low', 'high'):
        if field in params and params.get('low', -.5) >= params.get('high', .5): raise ValueError('TOST low must be below high')
    return method
