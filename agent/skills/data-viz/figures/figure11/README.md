# 11 · 环形系统发育树与热图

**中文** | [English](README.en.md)

[← 返回总目录](../../README.md)

![Preview](preview.png)

## 数据来源

**synthetic** — 模拟树拓扑 分支长度及叶节点注释；仅复现图式，不能用于系统发育推断

原始实验数据未提供；论文/DOI 尚未核实。这里的值仅用于图式复用，不是原始研究数据，也不构成像素完全一致的复现。

## 通用数据要求

**这些模板不限于生信数据。** 商业、工程、教育、社会调查等领域的数据，只要符合目标图的 CSV 字段名、类型、表结构和数值约束，就可以复用。字段表里的生物学名称与单位描述的是当前示例，可映射为自己的指标含义；CSV 列名仍需保留代码要求的名称。

优先阅读“文件与字段”，按格式整理数据，并同步修改 `style.json` 中的分类、标签、单位和坐标范围；分组或面板数量变化时，按需调整 `plot.py` 的固定布局。下文原始数据与预处理说明仅用于理解原图研究背景，复用图式不要求具备这些生信原始数据，也不要求执行对应生信分析。各图的数学约束仍然适用，例如对数轴取值必须为正、误差非负、树结构不能有环。

## 原图研究背景：原始数据

- 原始带枝长的系统发育树、根、叶节点 ID 与属/门分类。
- 每个叶节点的 BBAA 检出分子/分母及对应比例，bsh 存在/缺失信息。
- BBAA、bsh 的检测方法与阈值，属区块边界和颜色刻度范围。

## 原图研究背景：预处理示例

将真实树转为边表并合并 tip 注释。此圆形布局把终端节点对齐到固定半径，不能从图上读出真实末端枝长；genus_labels 需按新树重新设置。

## 文件与字段

`plot.py` 是本图实际绘图代码；`style.json` 控制画布、标签、颜色和分类顺序；`data/` 保存可替换 CSV。`provenance.json` 逐文件记录来源类别、行数、SHA-256、字段含义与单位。

### [figure11.csv](data/figure11.csv)

`synthetic` · 258 行。空值/NaN/Inf 不受支持；字段使用 UTF-8。

| 字段 | 类型 | 单位 | 含义 |
|---|---|---|---|
| `node` | string | identifier | 唯一节点 ID；不能命名为 root |
| `parent` | string | identifier | 父节点 ID 或隐含根 root |
| `length` | number ≥ 0 | tree-specific | 从父节点到该节点的枝长；必须指定单位 |
| `phylum` | string | taxonomy | 门分类；与 style.phyla 中名称一致 |

```csv
node,parent,length,phylum
n0,root,0.11828764720890793,Fusobacteria
n1,n0,0.17776591392450247,Fusobacteria
```

### [figure11_leaves.csv](data/figure11_leaves.csv)

`synthetic` · 133 行。空值/NaN/Inf 不受支持；字段使用 UTF-8。

| 字段 | 类型 | 单位 | 含义 |
|---|---|---|---|
| `node` | string | identifier | 唯一节点 ID；不能命名为 root |
| `order` | unique number | ordering | 圆周上叶节点顺序，不可重复 |
| `phylum` | string | taxonomy | 门分类；与 style.phyla 中名称一致 |
| `type` | WGS / MAG / SAG | category | 基因组类型；图 10 使用 |
| `presence` | 0 / 1 / 2 | category | 图10的缺失/至少1/至少10样本检出类别 |
| `fraction` | number [0, 1] | fraction | 图11检出比例；颜色范围默认 0–0.6 |
| `bsh` | 0 / 1 | boolean | 图11的 bsh 缺失/存在 |

```csv
node,order,phylum,type,presence,fraction,bsh
n2,0,Fusobacteria,MAG,1,0.028,1
n4,1,Fusobacteria,WGS,2,0.4377,0
```

## 运行与替换数据

```bash
python -m figures.figure11.plot --format png svg
cp -R figures/figure11/data my_data
# 修改 my_data 内所有对应 CSV，再运行：
python render.py --figure 11 --data-dir my_data --out my_output --format png svg pdf
```

命令均从仓库根目录执行。修改组名、基因名、面板数、样本量标签或量纲时，也要修改 `style.json` 与必要的 `plot.py` 布局；固定画布不会自动容纳任意数量的分组。示例原图统计标注在 CSV 改动后默认停用。完整统计模式说明见[总 README](../../README.md)。
