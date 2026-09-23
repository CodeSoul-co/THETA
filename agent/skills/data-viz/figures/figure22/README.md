# 22 · 文档—词语关联与分布

[中文](README.md) | [English](README.en.md)

[← 返回主 README](../../README.md)

![Preview](preview.png)

## 数据来源与复刻范围

文档/词组标签依据图片转录，可能存在拼写估读误差；带状位置与 10 组箱线图五数概括为视觉估计。300 条文档—词组关联流线由固定种子 202622 模拟，不是从图片恢复的真实关联权重或文本分析结果。蓝色小方点是路径控制位置的视觉标记，不表示额外观测。

原始数值数据未提供，具体论文/DOI 未核实，不能声称像素完全一致；本图可独立由矢量代码绘制，不依赖参考截图。

## 通用数据要求

- 文档或文档组的 ID 和短标签；主题、关键词或词组的 ID、标签与类别颜色。
- 文档—词组关联表：如共现次数、TF-IDF、主题权重或人工编码强度；必须说明自己的权重含义。
- 可选：每个词组的某项数值指标分布，或同一指标的有序五数概括。Dissemination 在原图中的定义未核实，可替换为自己有定义的覆盖率或其他指标。

预先整理关联表与词组表；把每条流线的起止位置写成所属带内的 0–1 布局比例。颜色取目标词组，线宽=weight×flow_width，弯曲路径通过组内中心聚束。路径不是时间顺序，也不是守恒 Sankey 流量。箱线图读取给定摘要，不从流线权重推导。

## 保留布局并增删元素

优先保留三列布局，替换标签与关联数据。删去无数据关联直接删流线行；文档与词组表中的位置可以保留。缺少右侧分布时设置 show_dissemination=false，保留左侧布局且不读取摘要文件；摘要只覆盖部分词组也可，其余不画箱体。蓝色装饰节点可通过 show_nodes=false 关闭。

## 文件与字段

### [figure22.csv](data/figure22.csv)

`synthetic` · 300 行

| 字段 | 类型 | 单位 | 含义 |
|---|---|---|---|
| `link_id` | string | ID | 流线唯一标识 |
| `document_id` | string | ID | 文档或文档组标识 |
| `topic_id` | string | ID | 词组或主题标识 |
| `weight` | number | relative weight | 正值关联权重；乘以 flow_width 得到线宽 |
| `source_position` | number | fraction [0,1] | 流线起点在文档带内的位置；布局参数 |
| `target_position` | number | fraction [0,1] | 流线终点在词组带内的位置；布局参数 |

```csv
link_id,document_id,topic_id,weight,source_position,target_position
l000,d0,t8,0.603,0.01389,0.93396
l001,d0,t4,0.651,0.04167,0.12188
```

### [figure22_documents.csv](data/figure22_documents.csv)

`transcribed_estimated_and_synthetic` · 8 行

| 字段 | 类型 | 单位 | 含义 |
|---|---|---|---|
| `document_id` | string | ID | 文档或文档组标识 |
| `label` | string | text | 显示标签，以 &#124; 换行 |
| `y0` | number | canvas pixels | 带状区域上边界 |
| `y1` | number | canvas pixels | 带状区域下边界，必须大于 y0 |

```csv
document_id,label,y0,y1
d0,Knotted protein|Bioinformatics|Computational biology,16,60
d1,BioUML|K-mer|Intl. Soc. for Comp. Biol.,60,101
```

### [figure22_summary.csv](data/figure22_summary.csv)

`transcribed_estimated_and_synthetic` · 10 行

| 字段 | 类型 | 单位 | 含义 |
|---|---|---|---|
| `topic_id` | string | ID | 词组或主题标识 |
| `minimum` | number | metric units | 显示的下须值；需注明使用最小值或其他须规则 |
| `q1` | number | metric units | 第一四分位数 |
| `median` | number | metric units | 中位数 |
| `q3` | number | metric units | 第三四分位数 |
| `maximum` | number | metric units | 显示的上须值；需注明所用须规则 |

```csv
topic_id,minimum,q1,median,q3,maximum
t0,0.19,0.34,0.55,0.71,1.0
t1,0.11,0.32,0.48,0.71,1.0
```

### [figure22_topics.csv](data/figure22_topics.csv)

`transcribed_estimated_and_synthetic` · 10 行

| 字段 | 类型 | 单位 | 含义 |
|---|---|---|---|
| `topic_id` | string | ID | 词组或主题标识 |
| `label` | string | text | 显示标签，以 &#124; 换行 |
| `y0` | number | canvas pixels | 带状区域上边界 |
| `y1` | number | canvas pixels | 带状区域下边界，必须大于 y0 |
| `color` | string | Matplotlib color | 该词组关联流线的颜色 |

```csv
topic_id,label,y0,y1,color
t0,"linear, beam, models,|air, center",16,53,#8fb1ca
t1,"formula, electric, spin,|frequency, wave",53,83,#adc36b
```

## 运行与替换数据

```bash
python -m figures.figure22.plot --format png svg pdf
cp -R figures/figure22/data my_data
cp figures/figure22/style.json my_style.json
python render.py --figure 22 --data-dir my_data --style my_style.json --out my_output --format png svg pdf --annotations none
```

命令从工程根目录执行。数值须有限；必需数据不可留空或填伪造零值。删减元素后同步更新关联 ID、图例和相关文件依赖。显示中文时，选择本机已安装且支持中文的字体，并适当调节字号。

## 模拟数据复现

```bash
python figures/figure22/simulate_links.py --out simulated_links.csv --seed 202622
```

仅重建关联 CSV，其他标签与箱线图摘要是独立的估读快照。现有输出文件不会覆盖。
