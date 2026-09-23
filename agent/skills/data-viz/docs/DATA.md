# 数据格式与原图背景索引

[中文](DATA.md) | [English](DATA.en.md)

模板不限数据领域：符合各图 README 的字段、类型、表结构和数值约束即可复用，并需同步修改标签与样式。下表保留原图研究的数据背景，不是其他领域复用时的前置要求。

来源和原始数据是否可得，以每张图的 provenance.json 为准。示例值与科研原始数据不是同一层次；截图估读和模拟数据不能冒充实验观测。

| ID | 图式与字段说明 | 当前来源 | 原图研究背景（非通用要求） |
|---|---|---|---|
| 01 | [空心柱状图与散点](../figures/figure01/README.md) | `screenshot_estimate` | 样本/动物编号、处理组及每个重复的 PCNA+ 细胞计数。 计数区域与长度、300 μm 归一化方法、重复单位及排除规则。 原始组间比较方法、误差类型和多重检验设置。 |
| 02 | [PCoA 与边缘箱线图](../figures/figure02/README.md) | `digitized` | 每个样本的丰度/特征矩阵或样本间距离矩阵，以及样本 ID 到组别的映射。 距离度量、变换方法、PCoA 实现、PCoA1/2 坐标和各轴解释比例。 PERMANOVA 的模型、置换次数、分层信息，以及椭圆计算约定。 |
| 03 | [多队列 PCoA 组合图](../figures/figure03/README.md) | `digitized` | 各队列的特征表或统一距离矩阵；每个样本的 study、control/IBD 组别和 ID。 跨队列归一化/批次处理方法、PCoA 坐标及轴解释比例。 原始 PERMANOVA 设计，包括队列效应、协变量和置换约束。 |
| 04 | [时间序列与 iAUC 插图](../figures/figure04/README.md) | `mixed` | 每只动物在每个 day 的体重、基线体重、处理组及个体 ID；保留纵向配对关系。 体重变化百分比公式、Exposure/Cessation 的时间窗与 iAUC 积分及基线定义。 每个个体的 iAUC、组均值、指定的 SD/SEM/CI 及原始统计检验。 |
| 05 | [半小提琴 雨云图](../figures/figure05/README.md) | `synthetic_from_estimated_distribution` | 每个 genome 的唯一编号、物种/类别和 BGC 数量。 基因组筛选规则、BGC 检测方法/版本，以及一条记录代表的独立观测单位。 KDE 带宽、箱线图 whisker 规则，以及是否包含异常值。 |
| 06 | [断轴分组柱状图](../figures/figure06/README.md) | `screenshot_estimate` | Control/SA 条件下，各基因型的独立样本编号和 PR1a 表达测量。 表达归一化公式与参考基因；若使用 qPCR，保留目标/参考 Ct 和校准样本。 原始误差类型、统计检验，以及新数据应采用的断轴上下范围。 |
| 07 | [基因表达分组柱状图](../figures/figure07/README.md) | `screenshot_estimate` | 每个基因、基因型、Day15 样本的 mRNA 测量和生物学重复 ID。 原始 Ct/表达量、内参和归一化方法；保留技术重复与生物学重复的对应关系。 组间比较与多重检验、误差定义；P 值不能从图片恢复计算过程。 |
| 08 | [PARP1 变体柱状图](../figures/figure08/README.md) | `screenshot_estimate` | 每个 PARP1 变体、药物条件和生物学重复的 GFP 读数与样本 ID。 背景扣除、−DOX/DMSO 对照、FC 的归一化与配对规则。 真实重复值、汇总误差类型及检验结果；不要用 mean ± error 充当重复。 |
| 09 | [双轴嵌套柱状图](../figures/figure09/README.md) | `mixed` | 每个菌株/构建体和独立重复的 Indigoidine OD600 与 Daptomycin 浓度（μg/ml）。 检测、标准曲线、稀释系数及两项测量的配对关系。 两套组均值和误差类型，以及 0.72/115 基准线的实际来源。 |
| 10 | [环形系统发育树与分类注释](../figures/figure10/README.md) | `synthetic` | 原始 Newick/Nexus 系统发育树：真实拓扑、枝长、根和叶节点唯一编号。 每个叶节点的 phylum、WGS/MAG/SAG 类型和样本检出情况。 检出类别的阈值和样本范围；树的末端顺序与图例计数。 |
| 11 | [环形系统发育树与热图](../figures/figure11/README.md) | `synthetic` | 原始带枝长的系统发育树、根、叶节点 ID 与属/门分类。 每个叶节点的 BBAA 检出分子/分母及对应比例，bsh 存在/缺失信息。 BBAA、bsh 的检测方法与阈值，属区块边界和颜色刻度范围。 |
| 12 | [分组基因气泡矩阵](../figures/figure12/README.md) | `digitized` | 基因 × 细胞/细胞核表达矩阵，细胞类型标注，以及基因所属 Early/Intermediate/Late 分组。 归一化/对数/缩放方法，表达大于 0 的阈值、每组分母和缺失值处理。 对每个 gene × cell_type 计算平均表达及阳性百分比；当前脚本接收这两项汇总。 |
| 13 | [双对数散点与边缘直方图](../figures/figure13/README.md) | `synthetic` | 每个 genome 的 ID、SAG/MAG/WGS 类别、总长度 bp 和 CDS 数量。 原始长度/计数，以及 corrected 的校正方法和质量筛选规则。 真实样本量、拟合方法、残差尺度，以及用于组间残差检验的设计。 |
| 14 | [三组分子动力学山脊图](../figures/figure14/README.md) | `synthetic_fitted_curve` | 分子动力学轨迹的 frame/time/replicate ID，以及每个帧的 Core RMSD、A-loop RMSD 和 Distance 1 Δ。 参考结构、原子选择、结构对齐方法、距离定义与 Å 单位。 0/8/12/16/20/24 ns 分组对应的时间窗与采样策略；不要把自相关帧当独立重复。 |
| 15 | [流式细胞术山脊矩阵](../figures/figure15/README.md) | `synthetic_fitted_curve` | 原始 FCS 文件或每个 event 的 marker 强度、样本 ID、细胞群和刺激 −/+ 条件。 补偿矩阵、门控层级、活细胞/双细胞排除及 marker 对应通道。 对数/logicle/arcsinh 变换及参数、阴性对照和密度归一化方法。 |
| 16 | [细胞群表达山脊矩阵](../figures/figure16/README.md) | `synthetic_fitted_curve` | 每个细胞/event 的 CD161、NKG2A、CD31、CD8、CD57、CD28 表达，以及样本与 c1–c8 群标签。 Vγ9Vδ2/Vδ1 群的定义、预处理变换、门控/聚类方法和 All cells 的组成。 原始样本权重及中位数的计算规则；不能从峰高推断细胞数量。 |
| 17 | [区域年代分布 雨云图](../figures/figure17/README.md) | `mixed` | 每个 dated sample 的 ID、地区和 kyr BP 年代；保留年代不确定性或校准后概率分布。 BP 参考年份、年代校准、样本权重、分布汇总方式和 KDE 带宽。 同一批数据的五数概括和 rug 标记/权重；三份绘图输入必须同步。 |
| 18 | [进化年龄分面箱线图](../figures/figure18/README.md) | `screenshot_estimate` | 每个旁系同源基因对的 ID、dN、dS 或 dN/dS，以及基因年龄和 Reference/diapause 分组。 序列比对/替代率估计与过滤方法、dS=0 处理、年龄区间定义。 从实际样本计算的箱线概括、whisker 规则、比较检验和校正 P 值。 |
| 19 | [相关性 哑铃 气泡 与条形组合图](../figures/figure19/README.md) | `screenshot_estimate_and_digitized` | 单细胞表达矩阵、细胞类型、WT/KO 条件、样本 ID，以及细胞空间坐标或邻近度量。 TGFβRII signature、P14 CD8 T cell 参考群、邻近/距离定义与相关方法。 每种细胞类型的 WT/KO 相关系数、指定基因的平均表达/阳性率、两样本 KS 值及 Further/Closer/Similar 判定规则。 |
| 20 | [词语空间与分区网格](../figures/figure20/README.md) | `transcribed_estimated_and_digitized` | 需要显示的词语或短语、唯一 ID、二维位置、分组与高亮方式。 如果位置具有语义含义，需由自己的词/句向量及降维流程生成并记录方法；若只是复用排版，可直接替换标签、保留展示坐标。 可选分区网格需与同一坐标系匹配；本例网格不能当作新语料的语义边界。 |
| 21 | [词组层次聚类树](../figures/figure21/README.md) | `transcribed_and_screenshot_estimate` | 每个叶节点的 ID、显示词组和左右顺序。 一棵完整二叉层次树：每次合并的节点 ID、左右子节点 ID 和合并高度。可来自文本聚类、主题层次或其他领域的层次关系。 若要解释为真实文本聚类，需记录文本表示、距离函数、聚类方法；树必须由该分析实际生成。 |
| 22 | [文档—词语关联与分布](../figures/figure22/README.md) | `transcribed_estimated_and_synthetic` | 文档或文档组的 ID 和短标签；主题、关键词或词组的 ID、标签与类别颜色。 文档—词组关联表：如共现次数、TF-IDF、主题权重或人工编码强度；必须说明自己的权重含义。 可选：每个词组的某项数值指标分布，或同一指标的有序五数概括。Dissemination 在原图中的定义未核实，可替换为自己有定义的覆盖率或其他指标。 |
