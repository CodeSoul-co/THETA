# 论文插图与可复核可视化

[English](publication-visualization.md) | **中文**

所有模型的原生绘图入口共享 `src/models/visualization/publication.py` 的字体、导出和配色配置。实际绘图仍由现有 `VisualizationGenerator`、`TopicVisualizer` 及 STM/DTM 专用函数完成，不另行实现训练算法。

## 期刊风格

默认采用参考 Nature 图表规范的简洁版式：白底、细轴线、向外刻度、无背景网格、统一主题颜色、黑色文字和无边框图例。多面板按 a、b、c 编号；主题占比使用排序点线图，词权重使用窄条形图。普通文字在最终画布上保持约 7–11 pt，指标摘要的大数值单独使用 20 pt，英文使用可用的 Arial/Helvetica，中文文字独立选择 CJK 字体，避免旧版 Matplotlib 的混排缺字。

画布按最大 183 mm 宽、170 mm 高等比例缩放并重新排版；紧边界导出会改变最终外框，具体期刊尺寸仍需投稿前核对。此处是风格参考，不代表 Nature/Science 的官方模板或投稿合规认证。

参考：[Nature 图表规范](https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/) 与 [Nature 面板与尺寸说明](https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/)。

## 选择清晰度与格式

```bash
# 在仓库根目录，用装有引擎依赖的 Python 执行；只读取已有结果，不训练。
python src/models/visualization/run_visualization.py \
  --baseline --model lda --dataset OPC --num_topics 8 \
  --result_dir /path/to/single/experiment \
  --workspace_dir /path/to/its/preprocessing/workspace \
  --output_dir result/OPC-publication/zh \
  --language zh --dpi 600 --formats png pdf svg
```

- `--dpi 150` 用于预览；`300` 是默认；`600` 用于高清交付。支持 72–1200 的整数。
- `--formats png` 只出位图；`--formats pdf svg` 只出矢量文件。默认同时输出三种。
- PDF 嵌入 TrueType 字体；SVG 保留可编辑文字，接收方需要安装相应字体，跨机器交付优先使用 PDF。高密度散点与词云包含栅格图层，分辨率随 DPI 提升；不能把这两类图称为全矢量。
- 中英文标签共用本机真实可用的中文字体。未安装中文字体时会明确报错；不会用空白方框冒充完成。建议在绘图环境准备 Noto Sans CJK SC。
- 多主题表格分页，完整数据保留在 CSV；画面尺寸、字号和期刊单栏/双栏规范仍需按具体投稿要求核对。
- `index.html` 是本地图表目录，可按名称筛选、放大预览、下载 PNG/PDF/SVG。`publication-manifest.json` 记录实际文件及导出设置，`chart-status.json` 记录主绘图函数的生成、跳过或失败原因；`additional-chart-status.json` 记录共用补充图及 STM/DTM 专用图。实际绘图异常会使命令失败，并保留已经完成的图和诊断清单。

THETA 使用同一组 `--dpi` 和 `--formats` 参数：

```bash
python src/models/visualization/run_visualization.py \
  --result_dir /path/to/results --dataset OPC --mode zero_shot \
  --model_size 0.6B --output_dir result/OPC-theta-figures \
  --language zh --dpi 600 --formats png pdf svg
```

## 接入原始时间与来源标签

只有样本量相等不足以证明矩阵与原文对应。要补充时间/来源信息，需要逐行对照训练时的实际输入：

```bash
# 在上述单次可视化命令后添加：
--source_file data/OPCdata.xlsx \
--training_data /path/to/recorded/training/data.csv \
--text_column 正文 --time_column 发布时间 --group_column 公众号名称
```

`training_data` 为训练时保存的规范化 CSV，其文本列名是 `text`。如果存在实际的 `source_rows.npy`，先应用行映射，再验证原文。对应不上时拒绝拼接标签。该功能写入新可视化目录，不修改原数据、theta、beta 或训练目录。

时序图只使用有效日期，其他全局图仍用全部模型文档；`temporal-scope.json` 记录分母。来源按文档数量保留前 8 类，其余明确合并为“other sources”；分组趋势图展示其中样本量最大的 4 类，完整数值表提供实际展示范围。不同年份样本量不同，最后一年可能未覆盖全年，不能从原始数量增长直接推出业务增长。

## 准确性规则与模型适用范围

| 模型 | 图表语义与额外要求 |
| --- | --- |
| LDA、HDP、BTM、ETM、CTM、GSM、ProdLDA、THETA | 共用主题词、分布、相关、降维、时序和分组图；每类图以实际存在且对齐的证据为前提 |
| STM | 上述图及实际协变量/系数图；ANOVA 仅为探索性组间差异，按 BH-FDR 校正，不作因果解释 |
| DTM | 共用图及真实 `beta_over_time` 词权重图；静态 beta 图代表最后时间片，不冒充全期词分布 |
| NVDM | 潜在坐标不是主题概率。仅使用适用的词权重、坐标投影、训练记录等图，不输出概率占比或 pyLDAvis |
| BERTopic | 使用真实 `document_topics.npy`、自身词表和相应 BOW。占比图明确限定非离群且有权重的文档；原始软权重与 -1 标签保留。词权重来自 c-TF-IDF，非生成模型词概率 |

正文图与解释应遵循这些规则：

- UMAP 抽样固定随机种子，并导出样本矩阵行号与坐标。散点距离是投影结果，不能当作原空间精确距离；投影后的 DBSCAN 是另一个探索性诊断，不是训练模型的离群标签。
- 相关网络标注 Pearson 阈值和正负边；文档主题权重受总和约束，相关不等于因果或独立性检验。
- 时间走势连接实际观测均值，不使用可能产生负占比的三次样条；缺失时间段用缺口，不补零。
- 不同量纲指标分别显示原始值，不把 UMass/NPMI 等冒充统一 0–1 评分。
- 主题词条形图显示真实导出权重，不用固定倍数伪造总体或主题内词频。主题均值不再称作统计“显著性”。
- 没有训练历史、多 K 实验、分时 beta、词义轨迹或跨期实体对应时，相应图明确跳过。不能为凑齐图片而生成随机曲线、补造迁移流或另行训练。
- pyLDAvis 使用实际 BOW 计数，保留主题顺序。零词文档只在此交互视图排除，排除范围写入相邻 `.scope.json` 和 HTML；不生成均匀分布来修补空数据。HTML 使用 pyLDAvis 包内的脚本和样式，可离线打开。

BERTopic 新训练导出文档分配、词表与词频。历史缺失的聚类分配不能从 soft theta 唯一恢复，相关旧任务仍拒绝误导性绘图。

## 图表布局

使用 Matplotlib ≥3.7 的约束布局和外部图例，继续复用上述原生入口。绘图环境最低依赖版本已同步更新。

- 散点图保留所有抽样点；存在远端点时增加局部图，边界为各轴 Q1−1.5×IQR 到 Q3+1.5×IQR，受实际数据范围限制。局部可见点数明确标注；没有裁切时只画单个全景。此规则只控制视口，不重算、移动或删除坐标。
- 主题趋势与原始词频拆为小面板；共享纵轴上限包含全部已观测值。单主题趋势下方另列真实样本量，缺失年份保留断点。
- 年度与来源热图将长标签置于行，使用百分比单元格和连续色阶；相关矩阵使用完整正负色阶。网络保持真实 Pearson 阈值与边权。
- 词条形图增加排名和精确数值列；词云采用紧凑椭圆布局、较大的主词字号和同色系深浅层次；可用时选择已安装的中等字重字体。独立、旧版和合并词云共用原生 WordCloud 布局。词云字号是排版结果，精确权重以条形图与原始导出值为准。
- STM 协变量图改为更紧凑的横条、矩阵与系数图；DTM 使用两列分面和独立图例，真实分时词权重保持不变。



## 相关网络与桑基图

相关网络采用固定主题顺序的弧线布局，不使用弹簧或力导向布局。所有节点保留，边仍来自原始文档主题矩阵的 Pearson 相关（`|r| > 0.3`）；实线/虚线区分正负，线宽表达强度，节点附代表词。节点间距与弧线位置不表示统计距离。`topic_network_edges.csv` 导出所有实际展示的边。

原生桑基图入口输出两类可核实的分配图：`year_topic_sankey`（年份→主题）、`source_topic_sankey`（来源→主题）。每条带是该组文档主题权重之和，使用双精度汇总，保留全部正权重流量与组别；不把相邻年份均值拼成迁移。年份图仅排除缺失日期的行，来源图使用全部适用文档。沿用调用方已有的来源分组规则，不另行裁剪。输出包含 PNG/PDF/SVG、完整权重 CSV 和自包含的离线 Plotly HTML；环境需安装 `plotly>=5.0`。

此功能共用于适用的模型绘图入口；NVDM 的有符号潜在坐标不生成这种非负分配图。BERTopic 继续沿用有真实分配且非离群的适用文档范围，具体范围由图册说明。桑基带宽表示权重合计，不直接等同于硬分配文档计数。

`publication-manifest.json` 同时收录原生绘图状态；图册直接展示未生成的图表及原因，并按类别汇总逐主题缺项。训练曲线、多 K 评估、分时词矩阵、词义轨迹与推断检验仍需要各自的真实证据。真正的跨期迁移桑基图需要实体对应，不能从这两类分配图推断。


## 参考风格：矩形词云拼图与环形网络

原生全量绘图现在同时输出原有样式和新增样式，按图册中的文件选择：

- `topic_wordcloud_grid_1.*` 等：矩形词云，每页最多 6 个主题、3 列；超过 6 个主题自动分页。使用实际 beta 与词表中最多 80 个正权重词，不重复填词。标题与词云分别占用独立区域；英文、中文及长标题均经过约束布局。原有独立椭圆词云保留。
- `topic_network_circular.*`：固定圆周排列的彩色节点。代表词、相关系数位于独立说明区，避免交叉连线遮挡文字。边多时说明区显示绝对相关最强的 10 条，完整边仍全部绘制并导出 CSV；超过 10 个主题时，主题说明使用已有主题表。固定几何位置不表示主题距离；不生成没有检验依据的显著性星号。

直接调用原生方法时可选择 `generate_topic_network(layout='arc'|'circular', threshold=0.3)`，阈值会如实写入标题。词云拼图使用 `visualize_wordcloud_grid(topic_words, num_words=80, columns=3, topics_per_page=6)`；可以选择 1–3 列、每页 1–6 主题。图片继续支持导出 DPI 与 PNG/PDF/SVG 选项。词云图片本体为栅格，标题在 PDF/SVG 中保留矢量文字。

回归检查包含：分页不遗漏主题、标题与词云/总标题边界分离、环形与弧线网络的相同边数据，以及网络说明区与图形区域分离。模型训练与矩阵计算逻辑没有改变。

## 中英文组图示例

[查看完整组图与下载](examples/figure-atlas/README_zh.md)：每种语言各 188 张图，按全量 OPC、LDA/DTM/STM 补训样本和可选网络阈值分区。包含 7680 × 19328 像素总览、轻量预览、六图细节示例与编号索引。英文标签保留原始中文语料关键词。组图示例不附原始文本、矩阵或训练权重。
