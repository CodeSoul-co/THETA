# 08 · PARP1 变体柱状图

**中文** | [English](README.en.md)

[← 返回总目录](../../README.md)

![Preview](preview.png)

## 数据来源

**screenshot_estimate** — 柱高和误差人工估读；示例散点由均值及误差构造，不是真实重复

原始实验数据未提供；论文/DOI 尚未核实。这里的值仅用于图式复用，不是原始研究数据，也不构成像素完全一致的复现。

## 通用数据要求

**这些模板不限于生信数据。** 商业、工程、教育、社会调查等领域的数据，只要符合目标图的 CSV 字段名、类型、表结构和数值约束，就可以复用。字段表里的生物学名称与单位描述的是当前示例，可映射为自己的指标含义；CSV 列名仍需保留代码要求的名称。

优先阅读“文件与字段”，按格式整理数据，并同步修改 `style.json` 中的分类、标签、单位和坐标范围；分组或面板数量变化时，按需调整 `plot.py` 的固定布局。下文原始数据与预处理说明仅用于理解原图研究背景，复用图式不要求具备这些生信原始数据，也不要求执行对应生信分析。各图的数学约束仍然适用，例如对数轴取值必须为正、误差非负、树结构不能有环。

## 原图研究背景：原始数据

- 每个 PARP1 变体、药物条件和生物学重复的 GFP 读数与样本 ID。
- 背景扣除、−DOX/DMSO 对照、FC 的归一化与配对规则。
- 真实重复值、汇总误差类型及检验结果；不要用 mean ± error 充当重复。

## 原图研究背景：预处理示例

先完成 FC 归一化并提供实际重复值。截图示例中的点来自估计汇总，不可用于统计推断。

## 文件与字段

`plot.py` 是本图实际绘图代码；`style.json` 控制画布、标签、颜色和分类顺序；`data/` 保存可替换 CSV。`provenance.json` 逐文件记录来源类别、行数、SHA-256、字段含义与单位。

### [figure08.csv](data/figure08.csv)

`screenshot_estimate` · 84 行。空值/NaN/Inf 不受支持；字段使用 UTF-8。

| 字段 | 类型 | 单位 | 含义 |
|---|---|---|---|
| `category` | string | label | 分类/条件；与样式顺序一致 |
| `group` | string | label | 实验组或细胞群；与样式一致 |
| `sample` | string | identifier | 示例重复编号，真实数据应保留真实样本 ID |
| `value` | number | normalized expression / fold change | 单个观测值；单位及归一化见本图原始数据要求 |
| `mean` | number | normalized expression / fold change | 组均值；相同组每行重复同一数值 |
| `error` | number ≥ 0 | normalized expression / fold change | 指定的误差幅度；需注明 SD/SEM/CI |

```csv
category,group,sample,value,mean,error
WT,−DOX,1,-0.02,0,0.02
WT,−DOX,2,0,0,0.02
```

## 运行与替换数据

```bash
python -m figures.figure08.plot --format png svg
cp -R figures/figure08/data my_data
# 修改 my_data 内所有对应 CSV，再运行：
python render.py --figure 8 --data-dir my_data --out my_output --format png svg pdf
```

命令均从仓库根目录执行。修改组名、基因名、面板数、样本量标签或量纲时，也要修改 `style.json` 与必要的 `plot.py` 布局；固定画布不会自动容纳任意数量的分组。示例原图统计标注在 CSV 改动后默认停用。完整统计模式说明见[总 README](../../README.md)。
