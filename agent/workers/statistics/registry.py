"""Executable catalogue. Only registered methods can reach a numerical implementation."""
from __future__ import annotations

METHODS = {}


def register(family, entries, required=('x',), parameters=(), limitation=''):
    for item in entries.strip().splitlines():
        key, title = item.strip().split('|', 1)
        METHODS[key] = dict(id=key, name=title, family=family, required=list(required),
                            parameters=list(parameters), status='implemented', limitation=limitation)


register('descriptive', '''describe|描述性统计
frequency|频数与百分比
correlation|Pearson / Spearman / Kendall 相关
normality|Shapiro-Wilk 正态性
cronbach_alpha|Cronbach α 信度
kmo_bartlett|KMO 与 Bartlett 球形检验
nps|净推荐值 NPS''', parameters=('correlation',))
register('test', '''ttest_one|单样本 t 检验
wilcoxon|配对 Wilcoxon 符号秩检验
ttest_paired|配对 t 检验
friedman|Friedman 重复测量秩检验
cochran_q|Cochran Q 二分类重复测量检验
kendall_w|Kendall W 一致性
kappa|Cohen Kappa 一致性
bland_altman|Bland–Altman 一致性
runs|游程检验
binomial|精确二项检验
tost_one|单样本等效性 TOST
tost_paired|配对等效性 TOST''', parameters=('mu', 'low', 'high', 'probability'))
register('group_test', '''ttest_ind|Welch 独立样本 t 检验
mann_whitney|Mann–Whitney U 检验
anova_one|单因素方差分析
kruskal|Kruskal–Wallis H 检验
levene|Levene 方差齐性检验
tukey|Tukey HSD 多重比较
dunn|Dunn 秩和事后检验
tost_ind|独立样本等效性 TOST''', required=('y', 'group'), parameters=('low', 'high', 'p_adjust'))
register('contingency', '''chi_square|Pearson 卡方独立性检验
chi_square_yates|Yates 连续性校正卡方
fisher|Fisher 精确检验
mcnemar|McNemar 配对二分类检验''', required=('x', 'group'))
register('regression', '''ols|普通最小二乘回归
wls|加权最小二乘回归
logit|二项 Logistic 回归
probit|二项 Probit 回归
poisson|Poisson 计数回归
negative_binomial|负二项 NB2 回归
ordinal_logit|有序 Logistic 回归
multinomial_logit|多项 Logistic 回归
quantile|分位数回归
robust_linear|Huber M 稳健回归
glm_gamma|Gamma GLM（log 链接）
glm_binomial|Binomial GLM（logit 链接）
gee|交换相关 Gaussian GEE
mixed_linear|随机截距线性混合模型''', required=('y', 'x'), parameters=('covariance', 'max_lags', 'quantile'), limitation='分类自变量须显式哑变量编码；默认含截距。仅支持登记的协方差选项；不接受任意统计命令。')
register('econometrics', '''panel_fe|个体固定效应面板回归
panel_twfe|个体与时间双向固定效应
panel_re|随机效应面板回归
iv_2sls|工具变量 2SLS
iv_gmm|线性工具变量 GMM
did|两组两期 DID（交互项 OLS）
rdd|局部线性 sharp RDD''', required=('y', 'x'), parameters=('covariance', 'cutoff', 'bandwidth'), limitation='DID 仅两组两期、无错位处理；RDD 需预先指定带宽，不自动证明识别假设；GMM 为线性 IV GMM，不是任意矩条件。')
register('survival', '''kaplan_meier|Kaplan–Meier 生存估计
nelson_aalen|Nelson–Aalen 累积风险
logrank|多组 Log-rank 检验
cox|Cox 比例风险回归
weibull_aft|Weibull AFT 生存回归
lognormal_aft|Log-normal AFT 生存回归
loglogistic_aft|Log-logistic AFT 生存回归''', required=('time', 'event'), limitation='仅右删失；event=1 为事件、0 为删失，时间必须为正。无左截断、竞争风险或时变协变量。')
register('unsupervised', '''kmeans|K-means 聚类
hierarchical|Ward 层次聚类
dbscan|DBSCAN 密度聚类
gmm_cluster|Gaussian mixture 聚类
pca|主成分分析
factor|最大似然探索性因子分析
mds|度量多维尺度分析
cca|典型相关分析
correspondence|对应分析''', parameters=('components', 'clusters', 'eps', 'min_samples', 'scale'), limitation='数值列完整案例；默认标准化（对应分析除外）；聚类标签并非真实类别。')
register('timeseries', '''adf|ADF 单位根检验
kpss|KPSS 平稳性检验
ljung_box|Ljung–Box 自相关检验
arima|ARIMA 时间序列预测
exponential_smoothing|Holt 趋势指数平滑
var|向量自回归 VAR
granger|Granger 预测性检验
garch|GARCH(1,1) 条件波动''', parameters=('lags', 'order', 'horizon'), limitation='输入必须按时间排序且等间隔；缺失时间不静默补齐；Granger 是预测先后关系而非因果识别。')
register('decision', '''entropy_weight|熵权法
critic|CRITIC 客观权重
cv_weight|变异系数权重
topsis|TOPSIS 综合评价
rsr|秩和比 RSR
vikor|VIKOR 折衷排序
grey_relation|灰色关联分析
ahp|层次分析 AHP
coupling|耦合协调度''', parameters=('weights', 'cost_columns', 'v'), limitation='输入及指标方向须预先确定；结果是给定权重下的排序，不是统计显著性证据。')
register('optimization', '''linear_program|线性规划
mixed_integer_program|混合整数线性规划
quadratic_program|有界凸二次规划''', required=(), parameters=('c', 'a_ub', 'b_ub', 'a_eq', 'b_eq', 'bounds', 'integrality', 'q'), limitation='最小化；最多 100 个变量。二次规划仅接受半正定 Q；无任意 Python 或符号表达式执行。')
register('preprocess', '''standardize|Z-score 标准化
minmax|Min-max 归一化
winsorize|分位数缩尾
impute_median|中位数缺失插补
dummy|分类哑变量编码
outliers_iqr|IQR 异常值标记
rolling_mean|顺序滑动均值''', parameters=('tail', 'window'), limitation='输出派生数据，不修改源文件；预测建模的预处理必须只在训练集内拟合。')
ML = {
    'logistic': 'Logistic', 'svm': 'SVM', 'knn': 'KNN', 'tree': '决策树',
    'random_forest': '随机森林', 'extra_trees': '极端随机树', 'adaboost': 'AdaBoost',
    'gradient_boosting': '梯度提升', 'xgboost': 'XGBoost', 'lightgbm': 'LightGBM',
    'catboost': 'CatBoost', 'mlp': '多层感知机', 'naive_bayes': 'Gaussian Naive Bayes',
    'linear': '线性', 'ridge': '岭', 'lasso': 'Lasso', 'elastic_net': 'Elastic Net',
    'pls': '偏最小二乘', 'ransac': 'RANSAC',
}
for task in ('classifier', 'regressor'):
    for algorithm, label in ML.items():
        if task == 'classifier' and algorithm in {'linear', 'ridge', 'lasso', 'elastic_net', 'pls', 'ransac'}: continue
        if task == 'regressor' and algorithm in {'logistic', 'naive_bayes'}: continue
        register('ml', f'{algorithm}_{task}|{label}{"分类" if task == "classifier" else "回归"}',
                 required=('x', 'y'), parameters=('test_size', 'cv', 'search', 'text_column', 'split', 'tune'),
                 limitation='训练集内预处理与调参；独立测试集只评分。随机拆分只适合独立观测；分组或时间数据须指定 split。超参数网格有预算上限。预测效果不构成因果证据。')


def catalog(payload):
    family = payload.get('family')
    query = str(payload.get('query', '')).lower()
    rows = [v for v in METHODS.values() if (not family or v['family'] == family)
            and (not query or query in (v['id'] + ' ' + v['name']).lower())]
    offset, limit = int(payload.get('offset', 0)), min(int(payload.get('limit', 30)), 50)
    if offset < 0 or limit < 1: raise ValueError('Invalid pagination')
    return {'total': len(rows), 'methods': rows[offset:offset + limit],
            'nextOffset': offset + limit if offset + limit < len(rows) else None,
            'compatibility': 'Registered Python methods; supported parameters and numerical defaults are defined per method.'}


def inspect(payload):
    method = METHODS.get(payload['method'])
    if not method: raise ValueError('未实现的方法；先调用 statistics_methods 查看已登记能力')
    from .validation import parameter_contract, example
    return {**method, 'parameterContract': parameter_contract(method), 'example': example(method),
            'common': {'missing': 'drop or error; ML drops missing outcomes and imputes predictors on training only',
                       'alpha': '0.001..0.2, default 0.05', 'seed': '0..2147483647, default 42',
                       'columns': 'exact column names, never executable formulas'}}

register('design', '''anova_factorial|二/三/多因素方差分析
ancova|协方差分析
repeated_anova|平衡被试内重复测量 ANOVA
manova|多元方差分析
icc|组内相关 ICC(2,1) 与 ICC(3,1)
moderation|调节效应（交互项回归）
mediation|单中介 Bootstrap 间接效应
parallel_mediation|并行中介 Bootstrap
chain_mediation|链式中介 Bootstrap
psm|倾向得分最近邻匹配 ATT
sem|结构方程/路径/CFA（MLW）
apriori|Apriori 关联规则
power_ttest|独立两样本 t 检验功效/样本量''', required=(), parameters=('bootstrap', 'caliper', 'model', 'min_support', 'min_confidence', 'effect_size', 'power', 'nobs', 'ratio'), limitation='仅登记规格；中介/匹配不证明可忽略性；SEM 须论证识别与测量模型；功效不能用观察到的 p 值倒推研究质量。')

# Do not advertise/accept parameters silently ignored by a sibling estimator.
_OVERRIDES = {
 'correlation':['correlation'], 'ttest_one':['mu'], 'binomial':['probability'],
 'tost_one':['low','high'], 'tost_paired':['low','high'], 'tost_ind':['low','high'], 'dunn':['p_adjust'],
 'ols':['covariance','max_lags'], 'wls':['covariance','max_lags'],
 'quantile':['quantile','covariance'], 'robust_linear':[], 'gee':[], 'mixed_linear':[],
 'panel_fe':['covariance'], 'panel_twfe':['covariance'], 'panel_re':['covariance'],
 'iv_2sls':['covariance'], 'iv_gmm':['covariance'], 'did':['covariance'], 'rdd':['covariance','cutoff','bandwidth'],
 'kmeans':['clusters','scale'], 'hierarchical':['clusters','scale'], 'dbscan':['eps','min_samples','scale'],
 'gmm_cluster':['clusters','scale'], 'pca':['components','scale'], 'factor':['components','scale'], 'mds':['components','scale'], 'cca':['components','scale'], 'correspondence':['components'],
 'adf':['lags'], 'kpss':['lags'], 'ljung_box':['lags'], 'granger':['lags'], 'arima':['order','horizon'], 'var':['lags','horizon'], 'garch':['horizon'], 'exponential_smoothing':['horizon'],
 'entropy_weight':['cost_columns'], 'critic':['cost_columns'], 'cv_weight':[], 'topsis':['weights','cost_columns'], 'rsr':['weights','cost_columns'], 'vikor':['weights','cost_columns','v'], 'grey_relation':['weights','cost_columns'], 'ahp':[], 'coupling':['weights'],
 'linear_program':['c','a_ub','b_ub','a_eq','b_eq','bounds'], 'mixed_integer_program':['c','a_ub','b_ub','a_eq','b_eq','bounds','integrality'], 'quadratic_program':['c','a_ub','b_ub','a_eq','b_eq','bounds','q'],
 'winsorize':['tail'], 'rolling_mean':['window'],
 'mediation':['bootstrap'], 'parallel_mediation':['bootstrap'], 'chain_mediation':['bootstrap'], 'psm':['caliper'], 'sem':['model'], 'apriori':['min_support','min_confidence'], 'power_ttest':['effect_size','power','nobs','ratio'],
}
for key, method in METHODS.items():
    if key in _OVERRIDES: method['parameters']=_OVERRIDES[key]
    elif method['family'] in {'descriptive','test','group_test','preprocess','design'}:method['parameters']=[]
    elif method['family']=='regression':method['parameters']=['covariance','max_lags']
_REQUIRED = {'anova_factorial':['factors','y'], 'ancova':['factors','x','y'], 'repeated_anova':['factors','entity','y'], 'manova':['factors','endogenous'], 'icc':['x'], 'moderation':['x','y'], 'mediation':['x','endogenous','y'], 'parallel_mediation':['x','endogenous','y'], 'chain_mediation':['x','endogenous','y'], 'psm':['group','x','y'], 'sem':['x'], 'apriori':['x']}
for key,required in _REQUIRED.items():METHODS[key]['required']=required
