# 21 · 词组层次聚类树

[中文](README.md) | [English](README.en.md)

[← 返回主 README](../../README.md)

![Preview](preview.png)

## 数据来源与复刻范围

9 组共 36 个可见英文词已转录；8 次合并关系与高度按图估读。这里的高度只是展示估计，不是从原神经元响应或词向量计算出的距离。

原始数值数据未提供，具体论文/DOI 未核实，不能声称像素完全一致；本图可独立由矢量代码绘制，不依赖参考截图。

## 通用数据要求

- 每个叶节点的 ID、显示词组和左右顺序。
- 一棵完整二叉层次树：每次合并的节点 ID、左右子节点 ID 和合并高度。可来自文本聚类、主题层次或其他领域的层次关系。
- 若要解释为真实文本聚类，需记录文本表示、距离函数、聚类方法；树必须由该分析实际生成。

代码从明确的合并表绘制树，不执行神经数据分析或文本聚类。叶节点在原绘图区等距排列，内部横坐标取两个子节点的中点。高度需单调且在 height_limit 内。

## 保留布局并增删元素

直接替换叶节点 words，保留树布局；新语料同时修改 heading，避免保留神经元含义。删掉叶节点时必须同步剪枝并合并只有一个子节点的分叉；不能只删标签行。增加词组时提供对应新合并，优先保持画布与分叉颜色，必要时局部调整字号。

## 文件与字段

### [figure21.csv](data/figure21.csv)

`transcribed_and_screenshot_estimate` · 8 行

| 字段 | 类型 | 单位 | 含义 |
|---|---|---|---|
| `node` | string | ID | 节点唯一标识 |
| `left` | string | ID | 左子节点 ID |
| `right` | string | ID | 右子节点 ID |
| `height` | number | merge-height units | 非负合并高度，不能低于子节点 |

```csv
node,left,right,height
m0,w4,w5,48
m1,w3,m0,61
```

### [figure21_labels.csv](data/figure21_labels.csv)

`transcribed_and_screenshot_estimate` · 9 行

| 字段 | 类型 | 单位 | 含义 |
|---|---|---|---|
| `node` | string | ID | 节点唯一标识 |
| `order` | number | ordinal | 同一折线的点序，或叶节点左右顺序；须唯一 |
| `words` | string | text | 以 &#124; 分隔的堆叠词语；每个叶节点一组 |

```csv
node,order,words
w0,0,feelings|asleep|enjoys|happy
w1,1,hours|inside|nearby|times
```

## 运行与替换数据

```bash
python -m figures.figure21.plot --format png svg pdf
cp -R figures/figure21/data my_data
cp figures/figure21/style.json my_style.json
python render.py --figure 21 --data-dir my_data --style my_style.json --out my_output --format png svg pdf --annotations none
```

命令从工程根目录执行。数值须有限；必需数据不可留空或填伪造零值。删减元素后同步更新关联 ID、图例和相关文件依赖。显示中文时，选择本机已安装且支持中文的字体，并适当调节字号。
