# 12 · 分组基因气泡矩阵

**中文** | [English](README.en.md)

[← 返回总目录](../../README.md)

![Preview](preview.png)

## 数据来源

**digitized** — 按截图网格估读气泡大小和颜色，非原始表达矩阵

原始实验数据未提供；论文/DOI 尚未核实。这里的值仅用于图式复用，不是原始研究数据，也不构成像素完全一致的复现。

## 通用数据要求

**这些模板不限于生信数据。** 商业、工程、教育、社会调查等领域的数据，只要符合目标图的 CSV 字段名、类型、表结构和数值约束，就可以复用。字段表里的生物学名称与单位描述的是当前示例，可映射为自己的指标含义；CSV 列名仍需保留代码要求的名称。

优先阅读“文件与字段”，按格式整理数据，并同步修改 `style.json` 中的分类、标签、单位和坐标范围；分组或面板数量变化时，按需调整 `plot.py` 的固定布局。下文原始数据与预处理说明仅用于理解原图研究背景，复用图式不要求具备这些生信原始数据，也不要求执行对应生信分析。各图的数学约束仍然适用，例如对数轴取值必须为正、误差非负、树结构不能有环。

## 原图研究背景：原始数据

- 基因 × 细胞/细胞核表达矩阵，细胞类型标注，以及基因所属 Early/Intermediate/Late 分组。
- 归一化/对数/缩放方法，表达大于 0 的阈值、每组分母和缺失值处理。
- 对每个 gene × cell_type 计算平均表达及阳性百分比；当前脚本接收这两项汇总。

## 原图研究背景：预处理示例

先从表达矩阵计算 gene × cell_type 汇总，expression 为指定归一化后的均值，percent 为 0–100 的阳性百分比。代码不负责单细胞归一化。

## 文件与字段

`plot.py` 是本图实际绘图代码；`style.json` 控制画布、标签、颜色和分类顺序；`data/` 保存可替换 CSV。`provenance.json` 逐文件记录来源类别、行数、SHA-256、字段含义与单位。

### [figure12.csv](data/figure12.csv)

`digitized` · 378 行。空值/NaN/Inf 不受支持；字段使用 UTF-8。

| 字段 | 类型 | 单位 | 含义 |
|---|---|---|---|
| `section` | string | label | Early/Intermediate/Late 分区 |
| `gene` | string | identifier | 基因名称；与样式中的基因顺序一致 |
| `cell_type` | string | label | 细胞类型；与样式顺序一致 |
| `expression` | number | normalized expression | 指定归一化方式下的平均表达；非原始计数 |
| `percent` | number [0, 100] | percent | 阳性细胞百分比；控制气泡面积 |

```csv
section,gene,cell_type,expression,percent
Early,CRISPLD2,Activated fibroblast,0.0,30.25
Early,CRISPLD2,Fibroblast,0.67,56.25
```

## 运行与替换数据

```bash
python -m figures.figure12.plot --format png svg
cp -R figures/figure12/data my_data
# 修改 my_data 内所有对应 CSV，再运行：
python render.py --figure 12 --data-dir my_data --out my_output --format png svg pdf
```

命令均从仓库根目录执行。修改组名、基因名、面板数、样本量标签或量纲时，也要修改 `style.json` 与必要的 `plot.py` 布局；固定画布不会自动容纳任意数量的分组。示例原图统计标注在 CSV 改动后默认停用。完整统计模式说明见[总 README](../../README.md)。
