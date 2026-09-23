# THETA 可执行统计方法与使用规格

此表由注册表和实际执行矩阵生成；示例对应 `agent/examples/statistics/demo.csv`（AHP 使用 `ahp.csv`，混合模型使用 `mixed-linear.csv`）。所有数据是合成验收数据，不是实证发现。

通用参数：`alpha=0.05`，`seed=42`，`missing="drop"` 或 `"error"`。完整病例与插补规则见报告，分类预测在训练集内编码/插补/缩放。每项示例是 `statistics_plan.plan.steps` 的一个元素。

| 方法 ID | 名称 | 类别 |
|---|---|---|
| `describe` | 描述性统计 | descriptive |
| `frequency` | 频数与百分比 | descriptive |
| `correlation` | Pearson / Spearman / Kendall 相关 | descriptive |
| `normality` | Shapiro-Wilk 正态性 | descriptive |
| `cronbach_alpha` | Cronbach α 信度 | descriptive |
| `kmo_bartlett` | KMO 与 Bartlett 球形检验 | descriptive |
| `nps` | 净推荐值 NPS | descriptive |
| `ttest_one` | 单样本 t 检验 | test |
| `wilcoxon` | 配对 Wilcoxon 符号秩检验 | test |
| `ttest_paired` | 配对 t 检验 | test |
| `friedman` | Friedman 重复测量秩检验 | test |
| `cochran_q` | Cochran Q 二分类重复测量检验 | test |
| `kendall_w` | Kendall W 一致性 | test |
| `kappa` | Cohen Kappa 一致性 | test |
| `bland_altman` | Bland–Altman 一致性 | test |
| `runs` | 游程检验 | test |
| `binomial` | 精确二项检验 | test |
| `tost_one` | 单样本等效性 TOST | test |
| `tost_paired` | 配对等效性 TOST | test |
| `ttest_ind` | Welch 独立样本 t 检验 | group_test |
| `mann_whitney` | Mann–Whitney U 检验 | group_test |
| `anova_one` | 单因素方差分析 | group_test |
| `kruskal` | Kruskal–Wallis H 检验 | group_test |
| `levene` | Levene 方差齐性检验 | group_test |
| `tukey` | Tukey HSD 多重比较 | group_test |
| `dunn` | Dunn 秩和事后检验 | group_test |
| `tost_ind` | 独立样本等效性 TOST | group_test |
| `chi_square` | Pearson 卡方独立性检验 | contingency |
| `chi_square_yates` | Yates 连续性校正卡方 | contingency |
| `fisher` | Fisher 精确检验 | contingency |
| `mcnemar` | McNemar 配对二分类检验 | contingency |
| `ols` | 普通最小二乘回归 | regression |
| `wls` | 加权最小二乘回归 | regression |
| `logit` | 二项 Logistic 回归 | regression |
| `probit` | 二项 Probit 回归 | regression |
| `poisson` | Poisson 计数回归 | regression |
| `negative_binomial` | 负二项 NB2 回归 | regression |
| `ordinal_logit` | 有序 Logistic 回归 | regression |
| `multinomial_logit` | 多项 Logistic 回归 | regression |
| `quantile` | 分位数回归 | regression |
| `robust_linear` | Huber M 稳健回归 | regression |
| `glm_gamma` | Gamma GLM（log 链接） | regression |
| `glm_binomial` | Binomial GLM（logit 链接） | regression |
| `gee` | 交换相关 Gaussian GEE | regression |
| `mixed_linear` | 随机截距线性混合模型 | regression |
| `panel_fe` | 个体固定效应面板回归 | econometrics |
| `panel_twfe` | 个体与时间双向固定效应 | econometrics |
| `panel_re` | 随机效应面板回归 | econometrics |
| `iv_2sls` | 工具变量 2SLS | econometrics |
| `iv_gmm` | 线性工具变量 GMM | econometrics |
| `did` | 两组两期 DID（交互项 OLS） | econometrics |
| `rdd` | 局部线性 sharp RDD | econometrics |
| `kaplan_meier` | Kaplan–Meier 生存估计 | survival |
| `nelson_aalen` | Nelson–Aalen 累积风险 | survival |
| `logrank` | 多组 Log-rank 检验 | survival |
| `cox` | Cox 比例风险回归 | survival |
| `weibull_aft` | Weibull AFT 生存回归 | survival |
| `lognormal_aft` | Log-normal AFT 生存回归 | survival |
| `loglogistic_aft` | Log-logistic AFT 生存回归 | survival |
| `kmeans` | K-means 聚类 | unsupervised |
| `hierarchical` | Ward 层次聚类 | unsupervised |
| `dbscan` | DBSCAN 密度聚类 | unsupervised |
| `gmm_cluster` | Gaussian mixture 聚类 | unsupervised |
| `pca` | 主成分分析 | unsupervised |
| `factor` | 最大似然探索性因子分析 | unsupervised |
| `mds` | 度量多维尺度分析 | unsupervised |
| `cca` | 典型相关分析 | unsupervised |
| `correspondence` | 对应分析 | unsupervised |
| `adf` | ADF 单位根检验 | timeseries |
| `kpss` | KPSS 平稳性检验 | timeseries |
| `ljung_box` | Ljung–Box 自相关检验 | timeseries |
| `arima` | ARIMA 时间序列预测 | timeseries |
| `exponential_smoothing` | Holt 趋势指数平滑 | timeseries |
| `var` | 向量自回归 VAR | timeseries |
| `granger` | Granger 预测性检验 | timeseries |
| `garch` | GARCH(1,1) 条件波动 | timeseries |
| `entropy_weight` | 熵权法 | decision |
| `critic` | CRITIC 客观权重 | decision |
| `cv_weight` | 变异系数权重 | decision |
| `topsis` | TOPSIS 综合评价 | decision |
| `rsr` | 秩和比 RSR | decision |
| `vikor` | VIKOR 折衷排序 | decision |
| `grey_relation` | 灰色关联分析 | decision |
| `ahp` | 层次分析 AHP | decision |
| `coupling` | 耦合协调度 | decision |
| `linear_program` | 线性规划 | optimization |
| `mixed_integer_program` | 混合整数线性规划 | optimization |
| `quadratic_program` | 有界凸二次规划 | optimization |
| `standardize` | Z-score 标准化 | preprocess |
| `minmax` | Min-max 归一化 | preprocess |
| `winsorize` | 分位数缩尾 | preprocess |
| `impute_median` | 中位数缺失插补 | preprocess |
| `dummy` | 分类哑变量编码 | preprocess |
| `outliers_iqr` | IQR 异常值标记 | preprocess |
| `rolling_mean` | 顺序滑动均值 | preprocess |
| `logistic_classifier` | Logistic分类 | ml |
| `svm_classifier` | SVM分类 | ml |
| `knn_classifier` | KNN分类 | ml |
| `tree_classifier` | 决策树分类 | ml |
| `random_forest_classifier` | 随机森林分类 | ml |
| `extra_trees_classifier` | 极端随机树分类 | ml |
| `adaboost_classifier` | AdaBoost分类 | ml |
| `gradient_boosting_classifier` | 梯度提升分类 | ml |
| `xgboost_classifier` | XGBoost分类 | ml |
| `lightgbm_classifier` | LightGBM分类 | ml |
| `catboost_classifier` | CatBoost分类 | ml |
| `mlp_classifier` | 多层感知机分类 | ml |
| `naive_bayes_classifier` | Gaussian Naive Bayes分类 | ml |
| `svm_regressor` | SVM回归 | ml |
| `knn_regressor` | KNN回归 | ml |
| `tree_regressor` | 决策树回归 | ml |
| `random_forest_regressor` | 随机森林回归 | ml |
| `extra_trees_regressor` | 极端随机树回归 | ml |
| `adaboost_regressor` | AdaBoost回归 | ml |
| `gradient_boosting_regressor` | 梯度提升回归 | ml |
| `xgboost_regressor` | XGBoost回归 | ml |
| `lightgbm_regressor` | LightGBM回归 | ml |
| `catboost_regressor` | CatBoost回归 | ml |
| `mlp_regressor` | 多层感知机回归 | ml |
| `linear_regressor` | 线性回归 | ml |
| `ridge_regressor` | 岭回归 | ml |
| `lasso_regressor` | Lasso回归 | ml |
| `elastic_net_regressor` | Elastic Net回归 | ml |
| `pls_regressor` | 偏最小二乘回归 | ml |
| `ransac_regressor` | RANSAC回归 | ml |
| `anova_factorial` | 二/三/多因素方差分析 | design |
| `ancova` | 协方差分析 | design |
| `repeated_anova` | 平衡被试内重复测量 ANOVA | design |
| `manova` | 多元方差分析 | design |
| `icc` | 组内相关 ICC(2,1) 与 ICC(3,1) | design |
| `moderation` | 调节效应（交互项回归） | design |
| `mediation` | 单中介 Bootstrap 间接效应 | design |
| `parallel_mediation` | 并行中介 Bootstrap | design |
| `chain_mediation` | 链式中介 Bootstrap | design |
| `psm` | 倾向得分最近邻匹配 ATT | design |
| `sem` | 结构方程/路径/CFA（MLW） | design |
| `apriori` | Apriori 关联规则 | design |
| `power_ttest` | 独立两样本 t 检验功效/样本量 | design |

## describe · 描述性统计

```json
{
  "method": "describe",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {}
}
```

支持参数：无方法专用参数。
按登记输入与统计假设使用，参数/样本/依赖版本随结果归档。

## frequency · 频数与百分比

```json
{
  "method": "frequency",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {}
}
```

支持参数：无方法专用参数。
按登记输入与统计假设使用，参数/样本/依赖版本随结果归档。

## correlation · Pearson / Spearman / Kendall 相关

```json
{
  "method": "correlation",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {}
}
```

支持参数：correlation。
按登记输入与统计假设使用，参数/样本/依赖版本随结果归档。

## normality · Shapiro-Wilk 正态性

```json
{
  "method": "normality",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {}
}
```

支持参数：无方法专用参数。
按登记输入与统计假设使用，参数/样本/依赖版本随结果归档。

## cronbach_alpha · Cronbach α 信度

```json
{
  "method": "cronbach_alpha",
  "x": [
    "a",
    "b",
    "c"
  ],
  "seed": 42,
  "params": {}
}
```

支持参数：无方法专用参数。
按登记输入与统计假设使用，参数/样本/依赖版本随结果归档。

## kmo_bartlett · KMO 与 Bartlett 球形检验

```json
{
  "method": "kmo_bartlett",
  "x": [
    "a",
    "b",
    "c"
  ],
  "seed": 42,
  "params": {}
}
```

支持参数：无方法专用参数。
按登记输入与统计假设使用，参数/样本/依赖版本随结果归档。

## nps · 净推荐值 NPS

```json
{
  "method": "nps",
  "x": [
    "score"
  ],
  "seed": 42,
  "params": {}
}
```

支持参数：无方法专用参数。
按登记输入与统计假设使用，参数/样本/依赖版本随结果归档。

## ttest_one · 单样本 t 检验

```json
{
  "method": "ttest_one",
  "x": [
    "x"
  ],
  "seed": 42,
  "params": {}
}
```

支持参数：mu。
按登记输入与统计假设使用，参数/样本/依赖版本随结果归档。

## wilcoxon · 配对 Wilcoxon 符号秩检验

```json
{
  "method": "wilcoxon",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {}
}
```

支持参数：无方法专用参数。
按登记输入与统计假设使用，参数/样本/依赖版本随结果归档。

## ttest_paired · 配对 t 检验

```json
{
  "method": "ttest_paired",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {}
}
```

支持参数：无方法专用参数。
按登记输入与统计假设使用，参数/样本/依赖版本随结果归档。

## friedman · Friedman 重复测量秩检验

```json
{
  "method": "friedman",
  "x": [
    "a",
    "b",
    "c"
  ],
  "seed": 42,
  "params": {}
}
```

支持参数：无方法专用参数。
按登记输入与统计假设使用，参数/样本/依赖版本随结果归档。

## cochran_q · Cochran Q 二分类重复测量检验

```json
{
  "method": "cochran_q",
  "x": [
    "binary",
    "binary2",
    "binary3"
  ],
  "seed": 42,
  "params": {}
}
```

支持参数：无方法专用参数。
按登记输入与统计假设使用，参数/样本/依赖版本随结果归档。

## kendall_w · Kendall W 一致性

```json
{
  "method": "kendall_w",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {}
}
```

支持参数：无方法专用参数。
按登记输入与统计假设使用，参数/样本/依赖版本随结果归档。

## kappa · Cohen Kappa 一致性

```json
{
  "method": "kappa",
  "x": [
    "binary",
    "binary2"
  ],
  "seed": 42,
  "params": {}
}
```

支持参数：无方法专用参数。
按登记输入与统计假设使用，参数/样本/依赖版本随结果归档。

## bland_altman · Bland–Altman 一致性

```json
{
  "method": "bland_altman",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {}
}
```

支持参数：无方法专用参数。
按登记输入与统计假设使用，参数/样本/依赖版本随结果归档。

## runs · 游程检验

```json
{
  "method": "runs",
  "x": [
    "x"
  ],
  "seed": 42,
  "params": {}
}
```

支持参数：无方法专用参数。
按登记输入与统计假设使用，参数/样本/依赖版本随结果归档。

## binomial · 精确二项检验

```json
{
  "method": "binomial",
  "x": [
    "binary"
  ],
  "seed": 42,
  "params": {}
}
```

支持参数：probability。
按登记输入与统计假设使用，参数/样本/依赖版本随结果归档。

## tost_one · 单样本等效性 TOST

```json
{
  "method": "tost_one",
  "x": [
    "x"
  ],
  "seed": 42,
  "params": {}
}
```

支持参数：low, high。
按登记输入与统计假设使用，参数/样本/依赖版本随结果归档。

## tost_paired · 配对等效性 TOST

```json
{
  "method": "tost_paired",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {}
}
```

支持参数：low, high。
按登记输入与统计假设使用，参数/样本/依赖版本随结果归档。

## ttest_ind · Welch 独立样本 t 检验

```json
{
  "method": "ttest_ind",
  "y": "y",
  "group": "group"
}
```

支持参数：无方法专用参数。
按登记输入与统计假设使用，参数/样本/依赖版本随结果归档。

## mann_whitney · Mann–Whitney U 检验

```json
{
  "method": "mann_whitney",
  "y": "y",
  "group": "group"
}
```

支持参数：无方法专用参数。
按登记输入与统计假设使用，参数/样本/依赖版本随结果归档。

## anova_one · 单因素方差分析

```json
{
  "method": "anova_one",
  "y": "y",
  "group": "group"
}
```

支持参数：无方法专用参数。
按登记输入与统计假设使用，参数/样本/依赖版本随结果归档。

## kruskal · Kruskal–Wallis H 检验

```json
{
  "method": "kruskal",
  "y": "y",
  "group": "group"
}
```

支持参数：无方法专用参数。
按登记输入与统计假设使用，参数/样本/依赖版本随结果归档。

## levene · Levene 方差齐性检验

```json
{
  "method": "levene",
  "y": "y",
  "group": "group"
}
```

支持参数：无方法专用参数。
按登记输入与统计假设使用，参数/样本/依赖版本随结果归档。

## tukey · Tukey HSD 多重比较

```json
{
  "method": "tukey",
  "y": "y",
  "group": "group"
}
```

支持参数：无方法专用参数。
按登记输入与统计假设使用，参数/样本/依赖版本随结果归档。

## dunn · Dunn 秩和事后检验

```json
{
  "method": "dunn",
  "y": "y",
  "group": "group"
}
```

支持参数：p_adjust。
按登记输入与统计假设使用，参数/样本/依赖版本随结果归档。

## tost_ind · 独立样本等效性 TOST

```json
{
  "method": "tost_ind",
  "y": "y",
  "group": "group"
}
```

支持参数：low, high。
按登记输入与统计假设使用，参数/样本/依赖版本随结果归档。

## chi_square · Pearson 卡方独立性检验

```json
{
  "method": "chi_square",
  "x": [
    "binary"
  ],
  "group": "binary2"
}
```

支持参数：无方法专用参数。
按登记输入与统计假设使用，参数/样本/依赖版本随结果归档。

## chi_square_yates · Yates 连续性校正卡方

```json
{
  "method": "chi_square_yates",
  "x": [
    "binary"
  ],
  "group": "binary2"
}
```

支持参数：无方法专用参数。
按登记输入与统计假设使用，参数/样本/依赖版本随结果归档。

## fisher · Fisher 精确检验

```json
{
  "method": "fisher",
  "x": [
    "binary"
  ],
  "group": "binary2"
}
```

支持参数：无方法专用参数。
按登记输入与统计假设使用，参数/样本/依赖版本随结果归档。

## mcnemar · McNemar 配对二分类检验

```json
{
  "method": "mcnemar",
  "x": [
    "binary"
  ],
  "group": "binary2"
}
```

支持参数：无方法专用参数。
按登记输入与统计假设使用，参数/样本/依赖版本随结果归档。

## ols · 普通最小二乘回归

```json
{
  "method": "ols",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "y": "y"
}
```

支持参数：covariance, max_lags。
分类自变量须显式哑变量编码；默认含截距。仅支持登记的协方差选项；不是 Stata 命令解析器。

## wls · 加权最小二乘回归

```json
{
  "method": "wls",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "y": "y",
  "weight": "weight"
}
```

支持参数：covariance, max_lags。
分类自变量须显式哑变量编码；默认含截距。仅支持登记的协方差选项；不是 Stata 命令解析器。

## logit · 二项 Logistic 回归

```json
{
  "method": "logit",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "y": "binary"
}
```

支持参数：covariance, max_lags。
分类自变量须显式哑变量编码；默认含截距。仅支持登记的协方差选项；不是 Stata 命令解析器。

## probit · 二项 Probit 回归

```json
{
  "method": "probit",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "y": "binary"
}
```

支持参数：covariance, max_lags。
分类自变量须显式哑变量编码；默认含截距。仅支持登记的协方差选项；不是 Stata 命令解析器。

## poisson · Poisson 计数回归

```json
{
  "method": "poisson",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "y": "count"
}
```

支持参数：covariance, max_lags。
分类自变量须显式哑变量编码；默认含截距。仅支持登记的协方差选项；不是 Stata 命令解析器。

## negative_binomial · 负二项 NB2 回归

```json
{
  "method": "negative_binomial",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "y": "positive_count"
}
```

支持参数：covariance, max_lags。
分类自变量须显式哑变量编码；默认含截距。仅支持登记的协方差选项；不是 Stata 命令解析器。

## ordinal_logit · 有序 Logistic 回归

```json
{
  "method": "ordinal_logit",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "y": "ordinal"
}
```

支持参数：covariance, max_lags。
分类自变量须显式哑变量编码；默认含截距。仅支持登记的协方差选项；不是 Stata 命令解析器。

## multinomial_logit · 多项 Logistic 回归

```json
{
  "method": "multinomial_logit",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "y": "ordinal"
}
```

支持参数：covariance, max_lags。
分类自变量须显式哑变量编码；默认含截距。仅支持登记的协方差选项；不是 Stata 命令解析器。

## quantile · 分位数回归

```json
{
  "method": "quantile",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "y": "y"
}
```

支持参数：quantile, covariance。
分类自变量须显式哑变量编码；默认含截距。仅支持登记的协方差选项；不是 Stata 命令解析器。

## robust_linear · Huber M 稳健回归

```json
{
  "method": "robust_linear",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "y": "y"
}
```

支持参数：无方法专用参数。
分类自变量须显式哑变量编码；默认含截距。仅支持登记的协方差选项；不是 Stata 命令解析器。

## glm_gamma · Gamma GLM（log 链接）

```json
{
  "method": "glm_gamma",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "y": "positive"
}
```

支持参数：covariance, max_lags。
分类自变量须显式哑变量编码；默认含截距。仅支持登记的协方差选项；不是 Stata 命令解析器。

## glm_binomial · Binomial GLM（logit 链接）

```json
{
  "method": "glm_binomial",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "y": "binary"
}
```

支持参数：covariance, max_lags。
分类自变量须显式哑变量编码；默认含截距。仅支持登记的协方差选项；不是 Stata 命令解析器。

## gee · 交换相关 Gaussian GEE

```json
{
  "method": "gee",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "y": "y",
  "group": "entity"
}
```

支持参数：无方法专用参数。
分类自变量须显式哑变量编码；默认含截距。仅支持登记的协方差选项；不是 Stata 命令解析器。

## mixed_linear · 随机截距线性混合模型

```json
{
  "method": "mixed_linear",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "y": "y",
  "group": "entity"
}
```

支持参数：无方法专用参数。
分类自变量须显式哑变量编码；默认含截距。仅支持登记的协方差选项；不是 Stata 命令解析器。

## panel_fe · 个体固定效应面板回归

```json
{
  "method": "panel_fe",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "y": "y",
  "entity": "entity",
  "time": "wave"
}
```

支持参数：covariance。
DID 仅两组两期、无错位处理；RDD 需预先指定带宽，不自动证明识别假设；GMM 为线性 IV GMM，不是任意矩条件。

## panel_twfe · 个体与时间双向固定效应

```json
{
  "method": "panel_twfe",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "y": "y",
  "entity": "entity",
  "time": "wave"
}
```

支持参数：covariance。
DID 仅两组两期、无错位处理；RDD 需预先指定带宽，不自动证明识别假设；GMM 为线性 IV GMM，不是任意矩条件。

## panel_re · 随机效应面板回归

```json
{
  "method": "panel_re",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "y": "y",
  "entity": "entity",
  "time": "wave"
}
```

支持参数：covariance。
DID 仅两组两期、无错位处理；RDD 需预先指定带宽，不自动证明识别假设；GMM 为线性 IV GMM，不是任意矩条件。

## iv_2sls · 工具变量 2SLS

```json
{
  "method": "iv_2sls",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "y": "y",
  "endogenous": [
    "endog"
  ],
  "instruments": [
    "z"
  ]
}
```

支持参数：covariance。
DID 仅两组两期、无错位处理；RDD 需预先指定带宽，不自动证明识别假设；GMM 为线性 IV GMM，不是任意矩条件。

## iv_gmm · 线性工具变量 GMM

```json
{
  "method": "iv_gmm",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "y": "y",
  "endogenous": [
    "endog"
  ],
  "instruments": [
    "z"
  ]
}
```

支持参数：covariance。
DID 仅两组两期、无错位处理；RDD 需预先指定带宽，不自动证明识别假设；GMM 为线性 IV GMM，不是任意矩条件。

## did · 两组两期 DID（交互项 OLS）

```json
{
  "method": "did",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "y": "ydid",
  "group": "group",
  "time": "post"
}
```

支持参数：covariance。
DID 仅两组两期、无错位处理；RDD 需预先指定带宽，不自动证明识别假设；GMM 为线性 IV GMM，不是任意矩条件。

## rdd · 局部线性 sharp RDD

```json
{
  "method": "rdd",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {
    "cutoff": 0,
    "bandwidth": 2
  },
  "y": "y"
}
```

支持参数：covariance, cutoff, bandwidth。
DID 仅两组两期、无错位处理；RDD 需预先指定带宽，不自动证明识别假设；GMM 为线性 IV GMM，不是任意矩条件。

## kaplan_meier · Kaplan–Meier 生存估计

```json
{
  "method": "kaplan_meier",
  "seed": 42,
  "params": {},
  "time": "duration",
  "event": "event",
  "group": "group"
}
```

支持参数：无方法专用参数。
仅右删失；event=1 为事件、0 为删失，时间必须为正。无左截断、竞争风险或时变协变量。

## nelson_aalen · Nelson–Aalen 累积风险

```json
{
  "method": "nelson_aalen",
  "seed": 42,
  "params": {},
  "time": "duration",
  "event": "event",
  "group": "group"
}
```

支持参数：无方法专用参数。
仅右删失；event=1 为事件、0 为删失，时间必须为正。无左截断、竞争风险或时变协变量。

## logrank · 多组 Log-rank 检验

```json
{
  "method": "logrank",
  "seed": 42,
  "params": {},
  "time": "duration",
  "event": "event",
  "group": "group"
}
```

支持参数：无方法专用参数。
仅右删失；event=1 为事件、0 为删失，时间必须为正。无左截断、竞争风险或时变协变量。

## cox · Cox 比例风险回归

```json
{
  "method": "cox",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "time": "duration",
  "event": "event"
}
```

支持参数：无方法专用参数。
仅右删失；event=1 为事件、0 为删失，时间必须为正。无左截断、竞争风险或时变协变量。

## weibull_aft · Weibull AFT 生存回归

```json
{
  "method": "weibull_aft",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "time": "duration",
  "event": "event"
}
```

支持参数：无方法专用参数。
仅右删失；event=1 为事件、0 为删失，时间必须为正。无左截断、竞争风险或时变协变量。

## lognormal_aft · Log-normal AFT 生存回归

```json
{
  "method": "lognormal_aft",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "time": "duration",
  "event": "event"
}
```

支持参数：无方法专用参数。
仅右删失；event=1 为事件、0 为删失，时间必须为正。无左截断、竞争风险或时变协变量。

## loglogistic_aft · Log-logistic AFT 生存回归

```json
{
  "method": "loglogistic_aft",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "time": "duration",
  "event": "event"
}
```

支持参数：无方法专用参数。
仅右删失；event=1 为事件、0 为删失，时间必须为正。无左截断、竞争风险或时变协变量。

## kmeans · K-means 聚类

```json
{
  "method": "kmeans",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {}
}
```

支持参数：clusters, scale。
数值列完整案例；默认标准化（对应分析除外）；聚类标签并非真实类别。

## hierarchical · Ward 层次聚类

```json
{
  "method": "hierarchical",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {}
}
```

支持参数：clusters, scale。
数值列完整案例；默认标准化（对应分析除外）；聚类标签并非真实类别。

## dbscan · DBSCAN 密度聚类

```json
{
  "method": "dbscan",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {}
}
```

支持参数：eps, min_samples, scale。
数值列完整案例；默认标准化（对应分析除外）；聚类标签并非真实类别。

## gmm_cluster · Gaussian mixture 聚类

```json
{
  "method": "gmm_cluster",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {}
}
```

支持参数：clusters, scale。
数值列完整案例；默认标准化（对应分析除外）；聚类标签并非真实类别。

## pca · 主成分分析

```json
{
  "method": "pca",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {}
}
```

支持参数：components, scale。
数值列完整案例；默认标准化（对应分析除外）；聚类标签并非真实类别。

## factor · 最大似然探索性因子分析

```json
{
  "method": "factor",
  "x": [
    "a",
    "b",
    "c",
    "x"
  ],
  "seed": 42,
  "params": {
    "components": 1
  }
}
```

支持参数：components, scale。
数值列完整案例；默认标准化（对应分析除外）；聚类标签并非真实类别。

## mds · 度量多维尺度分析

```json
{
  "method": "mds",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {}
}
```

支持参数：components, scale。
数值列完整案例；默认标准化（对应分析除外）；聚类标签并非真实类别。

## cca · 典型相关分析

```json
{
  "method": "cca",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "endogenous": [
    "a",
    "b"
  ]
}
```

支持参数：components, scale。
数值列完整案例；默认标准化（对应分析除外）；聚类标签并非真实类别。

## correspondence · 对应分析

```json
{
  "method": "correspondence",
  "x": [
    "a",
    "b",
    "c"
  ],
  "seed": 42,
  "params": {}
}
```

支持参数：components。
数值列完整案例；默认标准化（对应分析除外）；聚类标签并非真实类别。

## adf · ADF 单位根检验

```json
{
  "method": "adf",
  "x": [
    "x"
  ],
  "seed": 42,
  "params": {
    "lags": 2
  },
  "time": "time"
}
```

支持参数：lags。
输入必须按时间排序且等间隔；缺失时间不静默补齐；Granger 是预测先后关系而非因果识别。

## kpss · KPSS 平稳性检验

```json
{
  "method": "kpss",
  "x": [
    "x"
  ],
  "seed": 42,
  "params": {
    "lags": 2
  },
  "time": "time"
}
```

支持参数：lags。
输入必须按时间排序且等间隔；缺失时间不静默补齐；Granger 是预测先后关系而非因果识别。

## ljung_box · Ljung–Box 自相关检验

```json
{
  "method": "ljung_box",
  "x": [
    "x"
  ],
  "seed": 42,
  "params": {
    "lags": 2
  },
  "time": "time"
}
```

支持参数：lags。
输入必须按时间排序且等间隔；缺失时间不静默补齐；Granger 是预测先后关系而非因果识别。

## arima · ARIMA 时间序列预测

```json
{
  "method": "arima",
  "x": [
    "x"
  ],
  "seed": 42,
  "params": {
    "horizon": 3
  },
  "time": "time"
}
```

支持参数：order, horizon。
输入必须按时间排序且等间隔；缺失时间不静默补齐；Granger 是预测先后关系而非因果识别。

## exponential_smoothing · Holt 趋势指数平滑

```json
{
  "method": "exponential_smoothing",
  "x": [
    "x"
  ],
  "seed": 42,
  "params": {
    "horizon": 3
  },
  "time": "time"
}
```

支持参数：horizon。
输入必须按时间排序且等间隔；缺失时间不静默补齐；Granger 是预测先后关系而非因果识别。

## var · 向量自回归 VAR

```json
{
  "method": "var",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {
    "lags": 2,
    "horizon": 3
  },
  "time": "time"
}
```

支持参数：lags, horizon。
输入必须按时间排序且等间隔；缺失时间不静默补齐；Granger 是预测先后关系而非因果识别。

## granger · Granger 预测性检验

```json
{
  "method": "granger",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {
    "lags": 2
  },
  "time": "time"
}
```

支持参数：lags。
输入必须按时间排序且等间隔；缺失时间不静默补齐；Granger 是预测先后关系而非因果识别。

## garch · GARCH(1,1) 条件波动

```json
{
  "method": "garch",
  "x": [
    "x"
  ],
  "seed": 42,
  "params": {
    "horizon": 3
  },
  "time": "time"
}
```

支持参数：horizon。
输入必须按时间排序且等间隔；缺失时间不静默补齐；Granger 是预测先后关系而非因果识别。

## entropy_weight · 熵权法

```json
{
  "method": "entropy_weight",
  "x": [
    "a",
    "b",
    "c"
  ],
  "seed": 42,
  "params": {}
}
```

支持参数：cost_columns。
输入及指标方向须预先确定；结果是给定权重下的排序，不是统计显著性证据。

## critic · CRITIC 客观权重

```json
{
  "method": "critic",
  "x": [
    "a",
    "b",
    "c"
  ],
  "seed": 42,
  "params": {}
}
```

支持参数：cost_columns。
输入及指标方向须预先确定；结果是给定权重下的排序，不是统计显著性证据。

## cv_weight · 变异系数权重

```json
{
  "method": "cv_weight",
  "x": [
    "a",
    "b",
    "c"
  ],
  "seed": 42,
  "params": {}
}
```

支持参数：无方法专用参数。
输入及指标方向须预先确定；结果是给定权重下的排序，不是统计显著性证据。

## topsis · TOPSIS 综合评价

```json
{
  "method": "topsis",
  "x": [
    "a",
    "b",
    "c"
  ],
  "seed": 42,
  "params": {}
}
```

支持参数：weights, cost_columns。
输入及指标方向须预先确定；结果是给定权重下的排序，不是统计显著性证据。

## rsr · 秩和比 RSR

```json
{
  "method": "rsr",
  "x": [
    "a",
    "b",
    "c"
  ],
  "seed": 42,
  "params": {}
}
```

支持参数：weights, cost_columns。
输入及指标方向须预先确定；结果是给定权重下的排序，不是统计显著性证据。

## vikor · VIKOR 折衷排序

```json
{
  "method": "vikor",
  "x": [
    "a",
    "b",
    "c"
  ],
  "seed": 42,
  "params": {}
}
```

支持参数：weights, cost_columns, v。
输入及指标方向须预先确定；结果是给定权重下的排序，不是统计显著性证据。

## grey_relation · 灰色关联分析

```json
{
  "method": "grey_relation",
  "x": [
    "a",
    "b",
    "c"
  ],
  "seed": 42,
  "params": {}
}
```

支持参数：weights, cost_columns。
输入及指标方向须预先确定；结果是给定权重下的排序，不是统计显著性证据。

## ahp · 层次分析 AHP

```json
{
  "method": "ahp",
  "x": [
    "a",
    "b",
    "c"
  ],
  "seed": 42,
  "params": {}
}
```

支持参数：无方法专用参数。
输入及指标方向须预先确定；结果是给定权重下的排序，不是统计显著性证据。

## coupling · 耦合协调度

```json
{
  "method": "coupling",
  "x": [
    "a",
    "b",
    "c"
  ],
  "seed": 42,
  "params": {}
}
```

支持参数：weights。
输入及指标方向须预先确定；结果是给定权重下的排序，不是统计显著性证据。

## linear_program · 线性规划

```json
{
  "method": "linear_program",
  "params": {
    "c": [
      -1,
      -2
    ],
    "a_ub": [
      [
        1,
        1
      ]
    ],
    "b_ub": [
      5
    ]
  }
}
```

支持参数：c, a_ub, b_ub, a_eq, b_eq, bounds。
最小化；最多 100 个变量。二次规划仅接受半正定 Q；无任意 Python 或符号表达式执行。

## mixed_integer_program · 混合整数线性规划

```json
{
  "method": "mixed_integer_program",
  "params": {
    "c": [
      -1,
      -2
    ],
    "a_ub": [
      [
        1,
        1
      ]
    ],
    "b_ub": [
      5
    ]
  }
}
```

支持参数：c, a_ub, b_ub, a_eq, b_eq, bounds, integrality。
最小化；最多 100 个变量。二次规划仅接受半正定 Q；无任意 Python 或符号表达式执行。

## quadratic_program · 有界凸二次规划

```json
{
  "method": "quadratic_program",
  "params": {
    "c": [
      -1,
      -2
    ],
    "a_ub": [
      [
        1,
        1
      ]
    ],
    "b_ub": [
      5
    ],
    "q": [
      [
        2,
        0
      ],
      [
        0,
        2
      ]
    ]
  }
}
```

支持参数：c, a_ub, b_ub, a_eq, b_eq, bounds, q。
最小化；最多 100 个变量。二次规划仅接受半正定 Q；无任意 Python 或符号表达式执行。

## standardize · Z-score 标准化

```json
{
  "method": "standardize",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {}
}
```

支持参数：无方法专用参数。
输出派生数据，不修改源文件；预测建模的预处理必须只在训练集内拟合。

## minmax · Min-max 归一化

```json
{
  "method": "minmax",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {}
}
```

支持参数：无方法专用参数。
输出派生数据，不修改源文件；预测建模的预处理必须只在训练集内拟合。

## winsorize · 分位数缩尾

```json
{
  "method": "winsorize",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {}
}
```

支持参数：tail。
输出派生数据，不修改源文件；预测建模的预处理必须只在训练集内拟合。

## impute_median · 中位数缺失插补

```json
{
  "method": "impute_median",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {}
}
```

支持参数：无方法专用参数。
输出派生数据，不修改源文件；预测建模的预处理必须只在训练集内拟合。

## dummy · 分类哑变量编码

```json
{
  "method": "dummy",
  "x": [
    "text"
  ],
  "seed": 42,
  "params": {}
}
```

支持参数：无方法专用参数。
输出派生数据，不修改源文件；预测建模的预处理必须只在训练集内拟合。

## outliers_iqr · IQR 异常值标记

```json
{
  "method": "outliers_iqr",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {}
}
```

支持参数：无方法专用参数。
输出派生数据，不修改源文件；预测建模的预处理必须只在训练集内拟合。

## rolling_mean · 顺序滑动均值

```json
{
  "method": "rolling_mean",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {}
}
```

支持参数：window。
输出派生数据，不修改源文件；预测建模的预处理必须只在训练集内拟合。

## logistic_classifier · Logistic分类

```json
{
  "method": "logistic_classifier",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "y": "binary"
}
```

支持参数：test_size, cv, search, text_column, split, tune。
训练集内预处理与调参；独立测试集只评分。随机拆分只适合独立观测；分组或时间数据须指定 split。超参数网格有预算上限。预测效果不构成因果证据。

## svm_classifier · SVM分类

```json
{
  "method": "svm_classifier",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "y": "binary"
}
```

支持参数：test_size, cv, search, text_column, split, tune。
训练集内预处理与调参；独立测试集只评分。随机拆分只适合独立观测；分组或时间数据须指定 split。超参数网格有预算上限。预测效果不构成因果证据。

## knn_classifier · KNN分类

```json
{
  "method": "knn_classifier",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "y": "binary"
}
```

支持参数：test_size, cv, search, text_column, split, tune。
训练集内预处理与调参；独立测试集只评分。随机拆分只适合独立观测；分组或时间数据须指定 split。超参数网格有预算上限。预测效果不构成因果证据。

## tree_classifier · 决策树分类

```json
{
  "method": "tree_classifier",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "y": "binary"
}
```

支持参数：test_size, cv, search, text_column, split, tune。
训练集内预处理与调参；独立测试集只评分。随机拆分只适合独立观测；分组或时间数据须指定 split。超参数网格有预算上限。预测效果不构成因果证据。

## random_forest_classifier · 随机森林分类

```json
{
  "method": "random_forest_classifier",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "y": "binary"
}
```

支持参数：test_size, cv, search, text_column, split, tune。
训练集内预处理与调参；独立测试集只评分。随机拆分只适合独立观测；分组或时间数据须指定 split。超参数网格有预算上限。预测效果不构成因果证据。

## extra_trees_classifier · 极端随机树分类

```json
{
  "method": "extra_trees_classifier",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "y": "binary"
}
```

支持参数：test_size, cv, search, text_column, split, tune。
训练集内预处理与调参；独立测试集只评分。随机拆分只适合独立观测；分组或时间数据须指定 split。超参数网格有预算上限。预测效果不构成因果证据。

## adaboost_classifier · AdaBoost分类

```json
{
  "method": "adaboost_classifier",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "y": "binary"
}
```

支持参数：test_size, cv, search, text_column, split, tune。
训练集内预处理与调参；独立测试集只评分。随机拆分只适合独立观测；分组或时间数据须指定 split。超参数网格有预算上限。预测效果不构成因果证据。

## gradient_boosting_classifier · 梯度提升分类

```json
{
  "method": "gradient_boosting_classifier",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "y": "binary"
}
```

支持参数：test_size, cv, search, text_column, split, tune。
训练集内预处理与调参；独立测试集只评分。随机拆分只适合独立观测；分组或时间数据须指定 split。超参数网格有预算上限。预测效果不构成因果证据。

## xgboost_classifier · XGBoost分类

```json
{
  "method": "xgboost_classifier",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "y": "binary"
}
```

支持参数：test_size, cv, search, text_column, split, tune。
训练集内预处理与调参；独立测试集只评分。随机拆分只适合独立观测；分组或时间数据须指定 split。超参数网格有预算上限。预测效果不构成因果证据。

## lightgbm_classifier · LightGBM分类

```json
{
  "method": "lightgbm_classifier",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "y": "binary"
}
```

支持参数：test_size, cv, search, text_column, split, tune。
训练集内预处理与调参；独立测试集只评分。随机拆分只适合独立观测；分组或时间数据须指定 split。超参数网格有预算上限。预测效果不构成因果证据。

## catboost_classifier · CatBoost分类

```json
{
  "method": "catboost_classifier",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "y": "binary"
}
```

支持参数：test_size, cv, search, text_column, split, tune。
训练集内预处理与调参；独立测试集只评分。随机拆分只适合独立观测；分组或时间数据须指定 split。超参数网格有预算上限。预测效果不构成因果证据。

## mlp_classifier · 多层感知机分类

```json
{
  "method": "mlp_classifier",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "y": "binary"
}
```

支持参数：test_size, cv, search, text_column, split, tune。
训练集内预处理与调参；独立测试集只评分。随机拆分只适合独立观测；分组或时间数据须指定 split。超参数网格有预算上限。预测效果不构成因果证据。

## naive_bayes_classifier · Gaussian Naive Bayes分类

```json
{
  "method": "naive_bayes_classifier",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "y": "binary"
}
```

支持参数：test_size, cv, search, text_column, split, tune。
训练集内预处理与调参；独立测试集只评分。随机拆分只适合独立观测；分组或时间数据须指定 split。超参数网格有预算上限。预测效果不构成因果证据。

## svm_regressor · SVM回归

```json
{
  "method": "svm_regressor",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "y": "y"
}
```

支持参数：test_size, cv, search, text_column, split, tune。
训练集内预处理与调参；独立测试集只评分。随机拆分只适合独立观测；分组或时间数据须指定 split。超参数网格有预算上限。预测效果不构成因果证据。

## knn_regressor · KNN回归

```json
{
  "method": "knn_regressor",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "y": "y"
}
```

支持参数：test_size, cv, search, text_column, split, tune。
训练集内预处理与调参；独立测试集只评分。随机拆分只适合独立观测；分组或时间数据须指定 split。超参数网格有预算上限。预测效果不构成因果证据。

## tree_regressor · 决策树回归

```json
{
  "method": "tree_regressor",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "y": "y"
}
```

支持参数：test_size, cv, search, text_column, split, tune。
训练集内预处理与调参；独立测试集只评分。随机拆分只适合独立观测；分组或时间数据须指定 split。超参数网格有预算上限。预测效果不构成因果证据。

## random_forest_regressor · 随机森林回归

```json
{
  "method": "random_forest_regressor",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "y": "y"
}
```

支持参数：test_size, cv, search, text_column, split, tune。
训练集内预处理与调参；独立测试集只评分。随机拆分只适合独立观测；分组或时间数据须指定 split。超参数网格有预算上限。预测效果不构成因果证据。

## extra_trees_regressor · 极端随机树回归

```json
{
  "method": "extra_trees_regressor",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "y": "y"
}
```

支持参数：test_size, cv, search, text_column, split, tune。
训练集内预处理与调参；独立测试集只评分。随机拆分只适合独立观测；分组或时间数据须指定 split。超参数网格有预算上限。预测效果不构成因果证据。

## adaboost_regressor · AdaBoost回归

```json
{
  "method": "adaboost_regressor",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "y": "y"
}
```

支持参数：test_size, cv, search, text_column, split, tune。
训练集内预处理与调参；独立测试集只评分。随机拆分只适合独立观测；分组或时间数据须指定 split。超参数网格有预算上限。预测效果不构成因果证据。

## gradient_boosting_regressor · 梯度提升回归

```json
{
  "method": "gradient_boosting_regressor",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "y": "y"
}
```

支持参数：test_size, cv, search, text_column, split, tune。
训练集内预处理与调参；独立测试集只评分。随机拆分只适合独立观测；分组或时间数据须指定 split。超参数网格有预算上限。预测效果不构成因果证据。

## xgboost_regressor · XGBoost回归

```json
{
  "method": "xgboost_regressor",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "y": "y"
}
```

支持参数：test_size, cv, search, text_column, split, tune。
训练集内预处理与调参；独立测试集只评分。随机拆分只适合独立观测；分组或时间数据须指定 split。超参数网格有预算上限。预测效果不构成因果证据。

## lightgbm_regressor · LightGBM回归

```json
{
  "method": "lightgbm_regressor",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "y": "y"
}
```

支持参数：test_size, cv, search, text_column, split, tune。
训练集内预处理与调参；独立测试集只评分。随机拆分只适合独立观测；分组或时间数据须指定 split。超参数网格有预算上限。预测效果不构成因果证据。

## catboost_regressor · CatBoost回归

```json
{
  "method": "catboost_regressor",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "y": "y"
}
```

支持参数：test_size, cv, search, text_column, split, tune。
训练集内预处理与调参；独立测试集只评分。随机拆分只适合独立观测；分组或时间数据须指定 split。超参数网格有预算上限。预测效果不构成因果证据。

## mlp_regressor · 多层感知机回归

```json
{
  "method": "mlp_regressor",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "y": "y"
}
```

支持参数：test_size, cv, search, text_column, split, tune。
训练集内预处理与调参；独立测试集只评分。随机拆分只适合独立观测；分组或时间数据须指定 split。超参数网格有预算上限。预测效果不构成因果证据。

## linear_regressor · 线性回归

```json
{
  "method": "linear_regressor",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "y": "y"
}
```

支持参数：test_size, cv, search, text_column, split, tune。
训练集内预处理与调参；独立测试集只评分。随机拆分只适合独立观测；分组或时间数据须指定 split。超参数网格有预算上限。预测效果不构成因果证据。

## ridge_regressor · 岭回归

```json
{
  "method": "ridge_regressor",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "y": "y"
}
```

支持参数：test_size, cv, search, text_column, split, tune。
训练集内预处理与调参；独立测试集只评分。随机拆分只适合独立观测；分组或时间数据须指定 split。超参数网格有预算上限。预测效果不构成因果证据。

## lasso_regressor · Lasso回归

```json
{
  "method": "lasso_regressor",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "y": "y"
}
```

支持参数：test_size, cv, search, text_column, split, tune。
训练集内预处理与调参；独立测试集只评分。随机拆分只适合独立观测；分组或时间数据须指定 split。超参数网格有预算上限。预测效果不构成因果证据。

## elastic_net_regressor · Elastic Net回归

```json
{
  "method": "elastic_net_regressor",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "y": "y"
}
```

支持参数：test_size, cv, search, text_column, split, tune。
训练集内预处理与调参；独立测试集只评分。随机拆分只适合独立观测；分组或时间数据须指定 split。超参数网格有预算上限。预测效果不构成因果证据。

## pls_regressor · 偏最小二乘回归

```json
{
  "method": "pls_regressor",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "y": "y"
}
```

支持参数：test_size, cv, search, text_column, split, tune。
训练集内预处理与调参；独立测试集只评分。随机拆分只适合独立观测；分组或时间数据须指定 split。超参数网格有预算上限。预测效果不构成因果证据。

## ransac_regressor · RANSAC回归

```json
{
  "method": "ransac_regressor",
  "x": [
    "x",
    "x2"
  ],
  "seed": 42,
  "params": {},
  "y": "y"
}
```

支持参数：test_size, cv, search, text_column, split, tune。
训练集内预处理与调参；独立测试集只评分。随机拆分只适合独立观测；分组或时间数据须指定 split。超参数网格有预算上限。预测效果不构成因果证据。

## anova_factorial · 二/三/多因素方差分析

```json
{
  "method": "anova_factorial",
  "params": {},
  "factors": [
    "group",
    "post"
  ],
  "y": "y"
}
```

支持参数：无方法专用参数。
仅登记规格；中介/匹配不证明可忽略性；SEM 须论证识别与测量模型；功效不能用观察到的 p 值倒推研究质量。

## ancova · 协方差分析

```json
{
  "method": "ancova",
  "params": {},
  "factors": [
    "group",
    "post"
  ],
  "y": "y",
  "x": [
    "x"
  ]
}
```

支持参数：无方法专用参数。
仅登记规格；中介/匹配不证明可忽略性；SEM 须论证识别与测量模型；功效不能用观察到的 p 值倒推研究质量。

## repeated_anova · 平衡被试内重复测量 ANOVA

```json
{
  "method": "repeated_anova",
  "params": {},
  "factors": [
    "wave"
  ],
  "entity": "entity",
  "y": "y"
}
```

支持参数：无方法专用参数。
仅登记规格；中介/匹配不证明可忽略性；SEM 须论证识别与测量模型；功效不能用观察到的 p 值倒推研究质量。

## manova · 多元方差分析

```json
{
  "method": "manova",
  "params": {},
  "factors": [
    "group"
  ],
  "endogenous": [
    "y",
    "positive"
  ]
}
```

支持参数：无方法专用参数。
仅登记规格；中介/匹配不证明可忽略性；SEM 须论证识别与测量模型；功效不能用观察到的 p 值倒推研究质量。

## icc · 组内相关 ICC(2,1) 与 ICC(3,1)

```json
{
  "method": "icc",
  "params": {},
  "x": [
    "a",
    "b",
    "c"
  ]
}
```

支持参数：无方法专用参数。
仅登记规格；中介/匹配不证明可忽略性；SEM 须论证识别与测量模型；功效不能用观察到的 p 值倒推研究质量。

## moderation · 调节效应（交互项回归）

```json
{
  "method": "moderation",
  "params": {},
  "x": [
    "x",
    "x2"
  ],
  "y": "y"
}
```

支持参数：无方法专用参数。
仅登记规格；中介/匹配不证明可忽略性；SEM 须论证识别与测量模型；功效不能用观察到的 p 值倒推研究质量。

## mediation · 单中介 Bootstrap 间接效应

```json
{
  "method": "mediation",
  "params": {
    "bootstrap": 100
  },
  "x": [
    "x"
  ],
  "y": "y",
  "endogenous": [
    "endog"
  ]
}
```

支持参数：bootstrap。
仅登记规格；中介/匹配不证明可忽略性；SEM 须论证识别与测量模型；功效不能用观察到的 p 值倒推研究质量。

## parallel_mediation · 并行中介 Bootstrap

```json
{
  "method": "parallel_mediation",
  "params": {
    "bootstrap": 100
  },
  "x": [
    "x"
  ],
  "y": "y",
  "endogenous": [
    "endog",
    "positive"
  ]
}
```

支持参数：bootstrap。
仅登记规格；中介/匹配不证明可忽略性；SEM 须论证识别与测量模型；功效不能用观察到的 p 值倒推研究质量。

## chain_mediation · 链式中介 Bootstrap

```json
{
  "method": "chain_mediation",
  "params": {
    "bootstrap": 100
  },
  "x": [
    "x"
  ],
  "y": "y",
  "endogenous": [
    "endog",
    "positive"
  ]
}
```

支持参数：bootstrap。
仅登记规格；中介/匹配不证明可忽略性；SEM 须论证识别与测量模型；功效不能用观察到的 p 值倒推研究质量。

## psm · 倾向得分最近邻匹配 ATT

```json
{
  "method": "psm",
  "params": {},
  "x": [
    "x",
    "x2"
  ],
  "y": "y",
  "group": "binary"
}
```

支持参数：caliper。
仅登记规格；中介/匹配不证明可忽略性；SEM 须论证识别与测量模型；功效不能用观察到的 p 值倒推研究质量。

## sem · 结构方程/路径/CFA（MLW）

```json
{
  "method": "sem",
  "params": {
    "model": "F =~ item0 + item1 + item2 + item3 + item4"
  },
  "x": [
    "item0",
    "item1",
    "item2",
    "item3",
    "item4"
  ]
}
```

支持参数：model。
仅登记规格；中介/匹配不证明可忽略性；SEM 须论证识别与测量模型；功效不能用观察到的 p 值倒推研究质量。

## apriori · Apriori 关联规则

```json
{
  "method": "apriori",
  "params": {},
  "x": [
    "binary",
    "binary2",
    "binary3"
  ]
}
```

支持参数：min_support, min_confidence。
仅登记规格；中介/匹配不证明可忽略性；SEM 须论证识别与测量模型；功效不能用观察到的 p 值倒推研究质量。

## power_ttest · 独立两样本 t 检验功效/样本量

```json
{
  "method": "power_ttest",
  "params": {}
}
```

支持参数：effect_size, power, nobs, ratio。
仅登记规格；中介/匹配不证明可忽略性；SEM 须论证识别与测量模型；功效不能用观察到的 p 值倒推研究质量。
