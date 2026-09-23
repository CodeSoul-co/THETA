# 新增与维护图表

**中文** | [English](CONTRIBUTING.en.md)

## 新图的最小结构

运行 `python -m scripts.new_figure --id 23 --title "新图" --title-en "New figure"`。每个 `figures/figureNN/` 需要：

- `plot.py`：实现 `draw(context, rows)`；从仓库根目录用 `python -m figures.figureNN.plot` 运行。
- `style.json`：`id` 与文件夹编号一致，包含画布配置与主 CSV 文件名。
- `data/`：UTF-8 CSV；只放明确可以公开的示例数据。
- `provenance.json`：双语原始数据需求、来源类别、逐文件行数/字段/单位/SHA-256。
- `example_hashes.json`：公开示例的 SHA-256 映射，放在数据目录之外。
- 中英文 README，写清实际数据来自哪里、缺少什么、预处理做了什么。
- `preview.png`：由当前公开示例生成。

`context.s` 是样式，`context.fig` 是 Matplotlib Figure，`context.ax([左,上,宽,高])` 以画布像素建立坐标轴，`context.load("辅助.csv")` 从当前数据目录读取辅助表。`context.annotations` 控制参考标注，`context.rng` 是可复现的随机排点来源。

## 记录原始数据

写清观测单位、样本 ID、物理单位、分组/配对关系、归一化、缺失值、排除规则及上游分析方法。明确区分：

1. 原始科研输入，如 FCS、Ct、基因表达矩阵、轨迹或 Newick 树。
2. 分析后的绘图输入，如 PCoA 坐标、KDE 曲线、均值/SEM、相关系数或 KS。
3. 模拟、截图估读和数字化提取的演示数据。

如果原始数据未获得，将 `original_raw_data_available` 写为 false；没有核实论文来源就保持 `source_publication: null`。不要把保留的图例样本量或 P 值当成重新测量/计算的结果。

## 两张树图的输入约定

边表使用 `node,parent,length,phylum`。`root` 是隐含根节点名称，`node` ID 唯一，枝长非负，所有节点必须连通且无环。叶表通过 `node` 连接，每个终端节点恰好一行；`order` 唯一，决定圆周顺序。`phylum` 的名称与 style.phyla 一致。

图 10 的 `type` 为 WGS/MAG/SAG，`presence` 为 0/1/2。图 11 使用 `fraction`（0–1）和 `bsh`（0/1），默认热图范围为 0–0.6。叶表为兼容两个案例保留全部列，但不是每张图都消费每一列。真实树文件需要在上游转成边表；本项目不做树推断，也没有 Newick 解析器。图 11 的固定终端半径会改变末端枝长的视觉比例。

## 修改已公布的示例

修改自己的私有 CSV 时不要改仓库指纹。只有在有意发布一个新的示例版本时，才同步更新 `provenance.json` 的行数、字段与 SHA-256，以及 `example_hashes.json`。可以用标准库计算：

```python
from pathlib import Path
import hashlib
p = Path("figures/figure23/data/figure20.csv")
print(hashlib.sha256(p.read_bytes()).hexdigest())
```

当前案例的初次生成过程使用 NumPy 随机种子 190913，但跨图共享的随机数流以及手工估读不可能只凭种子重建。**提交的 CSV 快照是当前示例的确切版本**。新增加的模拟器应把生成代码和独立种子放在对应图目录，并说明参数与分布假设。

## 发布前检查

```bash
python -m figures.figure23.plot --format png svg
cp output/figure23.png figures/figure23/preview.png
python -m scripts.build_gallery
python check_reuse.py
```

新图如果没有通用 `value` 列，在 `check_reuse.py` 的 `CHANGES` 中登记一个确实影响绘图的数值字段。检查器会验证改值后图像改变。不要上传真实敏感样本、凭据、原始参考截图或机器本地路径。原始研究资源的使用权需由其提供方说明。
