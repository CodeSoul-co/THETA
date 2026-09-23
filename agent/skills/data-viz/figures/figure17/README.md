# 17 · 区域年代分布 雨云图

**中文** | [English](README.en.md)

[← 返回总目录](../../README.md)

![Preview](preview.png)

## 数据来源

**mixed** — 截图分布轮廓和箱线图估读；rug 为模拟示例

原始实验数据未提供；论文/DOI 尚未核实。这里的值仅用于图式复用，不是原始研究数据，也不构成像素完全一致的复现。

## 通用数据要求

**这些模板不限于生信数据。** 商业、工程、教育、社会调查等领域的数据，只要符合目标图的 CSV 字段名、类型、表结构和数值约束，就可以复用。字段表里的生物学名称与单位描述的是当前示例，可映射为自己的指标含义；CSV 列名仍需保留代码要求的名称。

优先阅读“文件与字段”，按格式整理数据，并同步修改 `style.json` 中的分类、标签、单位和坐标范围；分组或面板数量变化时，按需调整 `plot.py` 的固定布局。下文原始数据与预处理说明仅用于理解原图研究背景，复用图式不要求具备这些生信原始数据，也不要求执行对应生信分析。各图的数学约束仍然适用，例如对数轴取值必须为正、误差非负、树结构不能有环。

## 原图研究背景：原始数据

- 每个 dated sample 的 ID、地区和 kyr BP 年代；保留年代不确定性或校准后概率分布。
- BP 参考年份、年代校准、样本权重、分布汇总方式和 KDE 带宽。
- 同一批数据的五数概括和 rug 标记/权重；三份绘图输入必须同步。

## 原图研究背景：预处理示例

密度表、五数表和 rug 表必须基于同一数据集更新。density 与 weight 是行距单位的展示高度；它们不是年代误差。

## 文件与字段

`plot.py` 是本图实际绘图代码；`style.json` 控制画布、标签、颜色和分类顺序；`data/` 保存可替换 CSV。`provenance.json` 逐文件记录来源类别、行数、SHA-256、字段含义与单位。

### [figure17.csv](data/figure17.csv)

`mixed` · 5010 行。空值/NaN/Inf 不受支持；字段使用 UTF-8。

| 字段 | 类型 | 单位 | 含义 |
|---|---|---|---|
| `region` | string | label | 地区名称；与样式中的分组匹配 |
| `age` | number | kyr BP | 距今千年；需要说明 BP 参考年份 |
| `density` | number ≥ 0 | row spacing | 分布的展示高度，不是样本量 |

```csv
region,age,density
Western Europe,15.0,0.0
Western Europe,14.98501,0.0
```

### [figure17_boxes.csv](data/figure17_boxes.csv)

`mixed` · 5 行。空值/NaN/Inf 不受支持；字段使用 UTF-8。

| 字段 | 类型 | 单位 | 含义 |
|---|---|---|---|
| `region` | string | label | 地区名称；与样式中的分组匹配 |
| `low` | number | same as observations | 下 whisker 或最小值，需明确约定 |
| `q1` | number | same as observations | 第25百分位 |
| `median` | number | same as observations | 第50百分位 |
| `q3` | number | same as observations | 第75百分位 |
| `high` | number | same as observations | 上 whisker 或最大值，需明确约定 |

```csv
region,low,q1,median,q3,high
Western Europe,0.9,1.2,2.7,5.4,10.3
Central/Eastern Europe,0.25,1.9,4,5.7,10.8
```

### [figure17_rug.csv](data/figure17_rug.csv)

`synthetic` · 650 行。空值/NaN/Inf 不受支持；字段使用 UTF-8。

| 字段 | 类型 | 单位 | 含义 |
|---|---|---|---|
| `region` | string | label | 地区名称；与样式中的分组匹配 |
| `age` | number | kyr BP | 距今千年；需要说明 BP 参考年份 |
| `weight` | number ≥ 0 | row spacing | rug 竖线展示高度；不等同于年代误差 |

```csv
region,age,weight
Western Europe,5.5856,0.0913
Western Europe,2.8353,0.0827
```

## 运行与替换数据

```bash
python -m figures.figure17.plot --format png svg
cp -R figures/figure17/data my_data
# 修改 my_data 内所有对应 CSV，再运行：
python render.py --figure 17 --data-dir my_data --out my_output --format png svg pdf
```

命令均从仓库根目录执行。修改组名、基因名、面板数、样本量标签或量纲时，也要修改 `style.json` 与必要的 `plot.py` 布局；固定画布不会自动容纳任意数量的分组。示例原图统计标注在 CSV 改动后默认停用。完整统计模式说明见[总 README](../../README.md)。
