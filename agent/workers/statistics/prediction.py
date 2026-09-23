"""Train-only preprocessing, bounded CV search, held-out prediction evidence."""
import math
import numpy as np
import pandas as pd
from sklearn import ensemble, linear_model, svm, neighbors, tree, neural_network, naive_bayes
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GroupShuffleSplit, GroupKFold, TimeSeriesSplit, StratifiedKFold, KFold, GridSearchCV, RandomizedSearchCV
from sklearn import metrics


def estimator(method, seed):
    classifier=method.endswith('_classifier'); algorithm=method.rsplit('_',1)[0]
    pairs={
      'tree':(tree.DecisionTreeClassifier,tree.DecisionTreeRegressor),
      'random_forest':(ensemble.RandomForestClassifier,ensemble.RandomForestRegressor),
      'extra_trees':(ensemble.ExtraTreesClassifier,ensemble.ExtraTreesRegressor),
      'adaboost':(ensemble.AdaBoostClassifier,ensemble.AdaBoostRegressor),
      'gradient_boosting':(ensemble.GradientBoostingClassifier,ensemble.GradientBoostingRegressor),
      'mlp':(neural_network.MLPClassifier,neural_network.MLPRegressor),
      'svm':(svm.SVC,svm.SVR), 'knn':(neighbors.KNeighborsClassifier,neighbors.KNeighborsRegressor),
    }
    if algorithm in pairs:
        cls=pairs[algorithm][0 if classifier else 1]; kw={}
        if algorithm not in {'svm','knn'}: kw['random_state']=seed
        if algorithm in {'random_forest','extra_trees'}: kw.update(n_estimators=100,n_jobs=1)
        if algorithm=='mlp': kw.update(hidden_layer_sizes=(32,),max_iter=500)
        return cls(**kw)
    if algorithm=='logistic': return linear_model.LogisticRegression(max_iter=1000,random_state=seed)
    if algorithm=='naive_bayes': return naive_bayes.GaussianNB()
    if algorithm=='linear': return linear_model.LinearRegression()
    if algorithm=='ridge': return linear_model.Ridge()
    if algorithm=='lasso': return linear_model.Lasso(max_iter=5000)
    if algorithm=='elastic_net': return linear_model.ElasticNet(max_iter=5000)
    if algorithm=='ransac': return linear_model.RANSACRegressor(random_state=seed)
    if algorithm=='pls':
        from sklearn.cross_decomposition import PLSRegression
        return PLSRegression(n_components=2,scale=False)
    if algorithm=='xgboost':
        from xgboost import XGBClassifier,XGBRegressor
        return (XGBClassifier if classifier else XGBRegressor)(n_estimators=100,max_depth=4,n_jobs=1,random_state=seed,tree_method='hist')
    if algorithm=='lightgbm':
        from lightgbm import LGBMClassifier,LGBMRegressor
        return (LGBMClassifier if classifier else LGBMRegressor)(n_estimators=100,n_jobs=1,random_state=seed,verbosity=-1)
    if algorithm=='catboost':
        from catboost import CatBoostClassifier,CatBoostRegressor
        return (CatBoostClassifier if classifier else CatBoostRegressor)(iterations=100,depth=4,thread_count=1,random_seed=seed,verbose=False,allow_writing_files=False)
    raise ValueError(method)


def search_grid(grid, model):
    if grid is False: return {}  # Explicit no-tuning, equivalent to an omitted grid.
    allowed={'C':(.001,1000),'alpha':(.000001,100),'l1_ratio':(0,1),'max_depth':(1,20),
             'n_estimators':(10,300),'n_neighbors':(1,30),'learning_rate':(.001,1),
             'min_samples_leaf':(1,50),'subsample':(.2,1),'n_components':(1,10)}
    if not isinstance(grid,dict) or len(grid)>3: raise ValueError('tune must contain <=3 estimator parameters')
    combinations=1
    out={}
    for key,values in grid.items():
        if key not in allowed or key not in model.get_params() or not isinstance(values,list) or not 1<=len(values)<=4: raise ValueError(f'Unsupported search parameter {key}')
        lo,hi=allowed[key]
        if any(type(v) not in (float,int) or not lo<=v<=hi for v in values): raise ValueError('Search value outside resource bound')
        if key in {'max_depth','n_estimators','n_neighbors','min_samples_leaf','n_components'} and any(type(v) is not int for v in values): raise ValueError('Integer search parameter required')
        combinations*=len(values); out['model__'+key]=values
    if combinations>12: raise ValueError('At most 12 search candidates')
    return out


def run(data,spec):
    p=spec.get('params',{}); seed=spec.get('seed',42); classifier=spec['method'].endswith('_classifier')
    cols=spec['x']; text=p.get('text_column'); cols=cols+([text] if text else [])
    X=data[cols].copy(); y=data[spec['y']].copy(); encoder=None
    if classifier:
        encoder=LabelEncoder(); y=pd.Series(encoder.fit_transform(y.astype(str)),index=y.index)
        if not 2<=y.nunique()<=30: raise ValueError('Classification requires 2..30 classes')
    else: y=y.astype(float)
    test_size=p.get('test_size',.25); split=p.get('split','random'); indices=np.arange(len(data)); groups=None
    if split=='group':
        if not spec.get('group') or data[spec['group']].isna().any(): raise ValueError('Group split needs nonmissing group column')
        groups=data[spec['group']].to_numpy()
        train,test=next(GroupShuffleSplit(n_splits=1,test_size=test_size,random_state=seed).split(X,y,groups))
    elif split=='time':
        if not spec.get('time') or not data[spec['time']].is_monotonic_increasing or data[spec['time']].duplicated().any(): raise ValueError('Time split needs strictly increasing unique time column')
        cut=int(len(data)*(1-test_size)); train,test=indices[:cut],indices[cut:]
    else: train,test=train_test_split(indices,test_size=test_size,random_state=seed,stratify=y if classifier else None)
    if min(len(train),len(test))<5: raise ValueError('Need at least 5 train and test cases')
    if classifier and set(y.iloc[train])!=set(y): raise ValueError('Training partition does not contain every outcome class')
    numeric=[c for c in spec['x'] if pd.api.types.is_numeric_dtype(X[c])]
    category=[c for c in spec['x'] if c not in numeric]
    # Vocabulary, categories, imputation values and scaling are learned by each training fold.
    transforms=[]
    if numeric: transforms.append(('numeric',Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler())]),numeric))
    if category:
        X[category]=X[category].fillna('__MISSING__').astype(str)
        transforms.append(('category',OneHotEncoder(handle_unknown='ignore',max_categories=32,sparse_output=False),category))
    if text:
        X[text]=X[text].fillna('').astype(str)
        transforms.append(('text',TfidfVectorizer(analyzer='char',ngram_range=(2,3),max_features=1000,min_df=2),text))
    preprocessing=ColumnTransformer(transforms,sparse_threshold=0)
    model=estimator(spec['method'],seed)
    pipe=Pipeline([('prepare',preprocessing),('model',model)])
    grid=search_grid(p.get('tune',{}),model); folds=p.get('cv',0)
    if folds==1: raise ValueError('cv is 0 (off) or 2..5')
    if grid and folds<2: raise ValueError('Tuning requires train-only CV >=2')
    evidence={'train_N':len(train),'test_N':len(test),'split':split,'seed':seed,'cv_folds':folds}
    if folds:
        if split=='group': cross=GroupKFold(folds)
        elif split=='time': cross=TimeSeriesSplit(folds)
        elif classifier: cross=StratifiedKFold(folds,shuffle=True,random_state=seed)
        else: cross=KFold(folds,shuffle=True,random_state=seed)
        kwargs=dict(scoring='balanced_accuracy' if classifier else 'neg_root_mean_squared_error',cv=cross,n_jobs=1,error_score='raise')
        search=(RandomizedSearchCV(pipe,grid,n_iter=min(12,math.prod(map(len,grid.values()))),random_state=seed,**kwargs) if p.get('search')=='random' else GridSearchCV(pipe,grid or {},**kwargs))
        search.fit(X.iloc[train],y.iloc[train],**({'groups':groups[train]} if groups is not None else {})); pipe=search.best_estimator_
        evidence.update(cv_score=search.best_score_,best_params=search.best_params_,cv_scoring=kwargs['scoring'])
    else: pipe.fit(X.iloc[train],y.iloc[train])
    predicted=np.asarray(pipe.predict(X.iloc[test])).reshape(-1)
    truth=y.iloc[test].to_numpy(); tables={}
    if classifier:
        evidence.update(accuracy=metrics.accuracy_score(truth,predicted),balanced_accuracy=metrics.balanced_accuracy_score(truth,predicted),f1_macro=metrics.f1_score(truth,predicted,average='macro',zero_division=0))
        labels=list(range(len(encoder.classes_)))
        matrix=metrics.confusion_matrix(truth,predicted,labels=labels)
        tables['confusion_matrix']=[dict(actual=str(encoder.classes_[i]),**{str(encoder.classes_[j]):int(v) for j,v in enumerate(row)}) for i,row in enumerate(matrix)]
        report=metrics.classification_report(truth,predicted,labels=labels,target_names=encoder.classes_,output_dict=True,zero_division=0)
        tables['classification']=[dict(class_name=k,**v) for k,v in report.items() if isinstance(v,dict)]
        if len(labels)==2 and len(np.unique(truth))==2:
            score=pipe.predict_proba(X.iloc[test])[:,1] if hasattr(pipe,'predict_proba') else pipe.decision_function(X.iloc[test]) if hasattr(pipe,'decision_function') else None
            if score is not None:
                evidence.update(roc_auc=metrics.roc_auc_score(truth,score),positive_class=str(encoder.classes_[1]))
                fpr,tpr,_=metrics.roc_curve(truth,score); tables['roc']=[dict(false_positive_rate=f,true_positive_rate=t) for f,t in zip(fpr,tpr)]
        truth=encoder.inverse_transform(truth.astype(int)); predicted=encoder.inverse_transform(predicted.astype(int))
    else: evidence.update(RMSE=np.sqrt(metrics.mean_squared_error(truth,predicted)),MAE=metrics.mean_absolute_error(truth,predicted),R2=metrics.r2_score(truth,predicted))
    tables['test_predictions']=[dict(source_row=int(data.index[i])+2,observed=t,predicted=v) for i,t,v in zip(test,truth,predicted)]
    return {'metrics':evidence,'tables':tables,'warnings':['Single held-out split; generalization uncertainty and external validity are not established. Do not report predictive coefficients/importances as causal effects.'], 'estimand':'Out-of-sample predictive performance on the stated held-out partition.'}
