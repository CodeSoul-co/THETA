# 04 · 时间序列与 iAUC 插图

**中文** | [English](README.en.md)

[← 返回总目录](../../README.md)

![Preview](preview.png)

## 数据来源

**mixed** — 均值和误差人工估读；iAUC 散点为固定种子的示例数据

原始实验数据未提供；论文/DOI 尚未核实。这里的值仅用于图式复用，不是原始研究数据，也不构成像素完全一致的复现。

## 通用数据要求

**这些模板不限于生信数据。** 商业、工程、教育、社会调查等领域的数据，只要符合目标图的 CSV 字段名、类型、表结构和数值约束，就可以复用。字段表里的生物学名称与单位描述的是当前示例，可映射为自己的指标含义；CSV 列名仍需保留代码要求的名称。

优先阅读“文件与字段”，按格式整理数据，并同步修改 `style.json` 中的分类、标签、单位和坐标范围；分组或面板数量变化时，按需调整 `plot.py` 的固定布局。下文原始数据与预处理说明仅用于理解原图研究背景，复用图式不要求具备这些生信原始数据，也不要求执行对应生信分析。各图的数学约束仍然适用，例如对数轴取值必须为正、误差非负、树结构不能有环。

## 原图研究背景：原始数据

- 每只动物在每个 day 的体重、基线体重、处理组及个体 ID；保留纵向配对关系。
- 体重变化百分比公式、Exposure/Cessation 的时间窗与 iAUC 积分及基线定义。
- 每个个体的 iAUC、组均值、指定的 SD/SEM/CI 及原始统计检验。

## 原图研究背景：预处理示例

时序 CSV 直接接收 mean/error；iAUC 必须按个体轨迹另行计算后写入辅助表，代码不会从均值曲线求个体 iAUC。

## 文件与字段

`plot.py` 是本图实际绘图代码；`style.json` 控制画布、标签、颜色和分类顺序；`data/` 保存可替换 CSV。`provenance.json` 逐文件记录来源类别、行数、SHA-256、字段含义与单位。

### [figure04.csv](data/figure04.csv)

`mixed` · 24 行。空值/NaN/Inf 不受支持；字段使用 UTF-8。

| 字段 | 类型 | 单位 | 含义 |
|---|---|---|---|
| `group` | string | label | 实验组或细胞群；与样式一致 |
| `day` | number | day | 相对实验起点的天数 |
| `mean` | number | percent weight change | 组均值；相同组每行重复同一数值 |
| `error` | number ≥ 0 | percent weight change | 指定的误差幅度；需注明 SD/SEM/CI |

```csv
group,day,mean,error
"Non-
SMK",0,0,0
"Non-
SMK",7,7.4,0.7
```

### [figure04_insets.csv](data/figure04_insets.csv)

`synthetic` · 400 行。空值/NaN/Inf 不受支持；字段使用 UTF-8。

| 字段 | 类型 | 单位 | 含义 |
|---|---|---|---|
| `phase` | string | label | Exposure 或 Cessation；两个插图阶段 |
| `group` | string | label | 实验组或细胞群；与样式一致 |
| `value` | number | iAUC in the documented integration convention | 单个观测值；单位及归一化见本图原始数据要求 |

```csv
phase,group,value
Exposure,"Non-
SMK",260.8027
Exposure,"Non-
SMK",173.563
```

## 运行与替换数据

```bash
python -m figures.figure04.plot --format png svg
cp -R figures/figure04/data my_data
# 修改 my_data 内所有对应 CSV，再运行：
python render.py --figure 4 --data-dir my_data --out my_output --format png svg pdf
```

命令均从仓库根目录执行。修改组名、基因名、面板数、样本量标签或量纲时，也要修改 `style.json` 与必要的 `plot.py` 布局；固定画布不会自动容纳任意数量的分组。示例原图统计标注在 CSV 改动后默认停用。完整统计模式说明见[总 README](../../README.md)。
