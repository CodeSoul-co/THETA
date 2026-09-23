# 18 · 进化年龄分面箱线图

**中文** | [English](README.en.md)

[← 返回总目录](../../README.md)

![Preview](preview.png)

## 数据来源

**screenshot_estimate** — 截图五数概括估读；统计标注为原图展示值，未重新做检验

原始实验数据未提供；论文/DOI 尚未核实。这里的值仅用于图式复用，不是原始研究数据，也不构成像素完全一致的复现。

## 通用数据要求

**这些模板不限于生信数据。** 商业、工程、教育、社会调查等领域的数据，只要符合目标图的 CSV 字段名、类型、表结构和数值约束，就可以复用。字段表里的生物学名称与单位描述的是当前示例，可映射为自己的指标含义；CSV 列名仍需保留代码要求的名称。

优先阅读“文件与字段”，按格式整理数据，并同步修改 `style.json` 中的分类、标签、单位和坐标范围；分组或面板数量变化时，按需调整 `plot.py` 的固定布局。下文原始数据与预处理说明仅用于理解原图研究背景，复用图式不要求具备这些生信原始数据，也不要求执行对应生信分析。各图的数学约束仍然适用，例如对数轴取值必须为正、误差非负、树结构不能有环。

## 原图研究背景：原始数据

- 每个旁系同源基因对的 ID、dN、dS 或 dN/dS，以及基因年龄和 Reference/diapause 分组。
- 序列比对/替代率估计与过滤方法、dS=0 处理、年龄区间定义。
- 从实际样本计算的箱线概括、whisker 规则、比较检验和校正 P 值。

## 原图研究背景：预处理示例

先计算 low/q1/median/q3/high 并明确 low/high 是极值还是 whisker。代码按输入绘制，不从序列估计 dN/dS 或重新检验 P 值。

## 文件与字段

`plot.py` 是本图实际绘图代码；`style.json` 控制画布、标签、颜色和分类顺序；`data/` 保存可替换 CSV。`provenance.json` 逐文件记录来源类别、行数、SHA-256、字段含义与单位。

### [figure18.csv](data/figure18.csv)

`screenshot_estimate` · 8 行。空值/NaN/Inf 不受支持；字段使用 UTF-8。

| 字段 | 类型 | 单位 | 含义 |
|---|---|---|---|
| `panel` | string or integer | label | 分面编号或名称；与样式定义一致 |
| `group` | string | label | 实验组或细胞群；与样式一致 |
| `low` | number | same as observations | 下 whisker 或最小值，需明确约定 |
| `q1` | number | same as observations | 第25百分位 |
| `median` | number | same as observations | 第50百分位 |
| `q3` | number | same as observations | 第75百分位 |
| `high` | number | same as observations | 上 whisker 或最大值，需明确约定 |

```csv
panel,group,low,q1,median,q3,high
"Genome-Wide
(all ages)",Reference,0,0.097,0.169,0.274,0.5
"Genome-Wide
(all ages)","Specialized
for diapause",0,0.09,0.157,0.243,0.5
```

## 运行与替换数据

```bash
python -m figures.figure18.plot --format png svg
cp -R figures/figure18/data my_data
# 修改 my_data 内所有对应 CSV，再运行：
python render.py --figure 18 --data-dir my_data --out my_output --format png svg pdf
```

命令均从仓库根目录执行。修改组名、基因名、面板数、样本量标签或量纲时，也要修改 `style.json` 与必要的 `plot.py` 布局；固定画布不会自动容纳任意数量的分组。示例原图统计标注在 CSV 改动后默认停用。完整统计模式说明见[总 README](../../README.md)。
