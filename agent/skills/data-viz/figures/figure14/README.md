# 14 · 三组分子动力学山脊图

**中文** | [English](README.en.md)

[← 返回总目录](../../README.md)

![Preview](preview.png)

## 数据来源

**synthetic_fitted_curve** — 按截图峰位 宽度 高度近似拟合的混合高斯轮廓；高度为绘图单位，非概率密度原值

原始实验数据未提供；论文/DOI 尚未核实。这里的值仅用于图式复用，不是原始研究数据，也不构成像素完全一致的复现。

## 通用数据要求

**这些模板不限于生信数据。** 商业、工程、教育、社会调查等领域的数据，只要符合目标图的 CSV 字段名、类型、表结构和数值约束，就可以复用。字段表里的生物学名称与单位描述的是当前示例，可映射为自己的指标含义；CSV 列名仍需保留代码要求的名称。

优先阅读“文件与字段”，按格式整理数据，并同步修改 `style.json` 中的分类、标签、单位和坐标范围；分组或面板数量变化时，按需调整 `plot.py` 的固定布局。下文原始数据与预处理说明仅用于理解原图研究背景，复用图式不要求具备这些生信原始数据，也不要求执行对应生信分析。各图的数学约束仍然适用，例如对数轴取值必须为正、误差非负、树结构不能有环。

## 原图研究背景：原始数据

- 分子动力学轨迹的 frame/time/replicate ID，以及每个帧的 Core RMSD、A-loop RMSD 和 Distance 1 Δ。
- 参考结构、原子选择、结构对齐方法、距离定义与 Å 单位。
- 0/8/12/16/20/24 ns 分组对应的时间窗与采样策略；不要把自相关帧当独立重复。

## 原图研究背景：预处理示例

可直接提供 panel/group/value 样本长表，由程序估计 KDE；也可提供 x/density 曲线。峰高按展示高度归一化，不表示样本量。

## 文件与字段

`plot.py` 是本图实际绘图代码；`style.json` 控制画布、标签、颜色和分类顺序；`data/` 保存可替换 CSV。`provenance.json` 逐文件记录来源类别、行数、SHA-256、字段含义与单位。

### [figure14.csv](data/figure14.csv)

`synthetic_fitted_curve` · 16290 行。空值/NaN/Inf 不受支持；字段使用 UTF-8。

| 字段 | 类型 | 单位 | 含义 |
|---|---|---|---|
| `panel` | string or integer | label | 分面编号或名称；与样式定义一致 |
| `group` | string | label | 实验组或细胞群；与样式一致 |
| `x` | number | Å | 已转换的横轴坐标；不是原始特征矩阵 |
| `density` | number ≥ 0 | row spacing | 分布的展示高度，不是样本量 |

```csv
panel,group,x,density
0,0,2.05,0.0
0,0,2.05387,0.0
```

## 运行与替换数据

```bash
python -m figures.figure14.plot --format png svg
cp -R figures/figure14/data my_data
# 修改 my_data 内所有对应 CSV，再运行：
python render.py --figure 14 --data-dir my_data --out my_output --format png svg pdf
```

命令均从仓库根目录执行。修改组名、基因名、面板数、样本量标签或量纲时，也要修改 `style.json` 与必要的 `plot.py` 布局；固定画布不会自动容纳任意数量的分组。示例原图统计标注在 CSV 改动后默认停用。完整统计模式说明见[总 README](../../README.md)。
