# 20 · 词语空间与分区网格

[中文](README.md) | [English](README.en.md)

[← 返回主 README](../../README.md)

![Preview](preview.png)

## 数据来源与复刻范围

427 个词语/短语根据低分辨率图片中可辨认内容整理，词语与位置的对应为近似估计，不能保证逐字恢复；网格由颜色线条检测并合并为矢量线段。没有恢复原始语义向量、距离或真实分区。

原始数值数据未提供，具体论文/DOI 未核实，不能声称像素完全一致；本图可独立由矢量代码绘制，不依赖参考截图。

## 通用数据要求

- 需要显示的词语或短语、唯一 ID、二维位置、分组与高亮方式。
- 如果位置具有语义含义，需由自己的词/句向量及降维流程生成并记录方法；若只是复用排版，可直接替换标签、保留展示坐标。
- 可选分区网格需与同一坐标系匹配；本例网格不能当作新语料的语义边界。

代码直接绘制文字与折线，不计算词向量、UMAP、t-SNE、聚类或相似度。坐标可先归一化到样式的 xlim/ylim，注意当前 y 轴向下。

## 保留布局并增删元素

优先保留坐标、字号、红色强调与网格布局，替换所需文字。删词直接删 CSV 行，少量新词加入空余位置。若重新计算二维位置，同时更新网格；没有可信网格时设置 show_mesh=false，代码不会读取网格文件。长文本宜用短标签，或局部调整字号/位置。

## 文件与字段

### [figure20.csv](data/figure20.csv)

`transcribed_estimated_and_digitized` · 427 行

| 字段 | 类型 | 单位 | 含义 |
|---|---|---|---|
| `word_id` | string | ID | 词语的唯一标识 |
| `text` | string | text | 显示文字，可为词语或短语 |
| `x` | number | canvas pixels | 横坐标；本例从左向右 |
| `y` | number | canvas pixels | 纵坐标；本例从上向下 |
| `role` | string | enum | word 或 highlight，决定文字颜色 |
| `font_size` | number | points | 正值文字字号 |
| `group` | string | label | 文字所属类别；元数据，不自动生成聚类 |

```csv
word_id,text,x,y,role,font_size,group
h000,Kindness,258.0,11.0,highlight,3.9,Kindness
h001,Deception,263.0,25.0,highlight,3.9,Deception
```

### [figure20_mesh.csv](data/figure20_mesh.csv)

`transcribed_estimated_and_digitized` · 496 行

| 字段 | 类型 | 单位 | 含义 |
|---|---|---|---|
| `path_id` | string | ID | 一条网格折线的标识 |
| `order` | number | ordinal | 同一折线的点序，或叶节点左右顺序；须唯一 |
| `x` | number | canvas pixels | 横坐标；本例从左向右 |
| `y` | number | canvas pixels | 纵坐标；本例从上向下 |

```csv
path_id,order,x,y
p000,0,169.22,312.33
p000,1,40.03,236.66
```

## 运行与替换数据

```bash
python -m figures.figure20.plot --format png svg pdf
cp -R figures/figure20/data my_data
cp figures/figure20/style.json my_style.json
python render.py --figure 20 --data-dir my_data --style my_style.json --out my_output --format png svg pdf --annotations none
```

命令从工程根目录执行。数值须有限；必需数据不可留空或填伪造零值。删减元素后同步更新关联 ID、图例和相关文件依赖。显示中文时，选择本机已安装且支持中文的字体，并适当调节字号。
