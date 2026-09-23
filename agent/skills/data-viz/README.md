# Data Viz Skill

```bash
git clone --depth 1 https://github.com/AdamsukS/data-viz-skill.git "${CODEX_HOME:-$HOME/.codex}/skills/data-viz"
```

**中文** | [English](README.en.md)

上面的命令可由 Agent 在终端直接执行，将完整 Skill 下载到 Codex 的技能目录；需要 Git，目标目录已存在时会停止，避免覆盖已有修改。安装后在下一轮对话中使用 `$data-viz`。安装只下载代码与素材，Python 依赖在实际绘图工作目录中安装。

**让 AI 直接复用已有绘图代码，尽量保留原图布局。** 这个 Skill 包含 22 套可运行的 Python 图式，以及独立数据、配色、版式、字段说明和预览。核心能力是把用户数据接入现有代码，再做必要的局部修改：缺少数据时删去对应元素，少量新增信息优先在原图上扩展。只有大量新变量或关系无法通过局部修改清晰表达，或用户明确要求新设计时，才创造新的构图。数据不限于生信，适用于业务、工程、调查及其他领域。

## 如何使用

可以把下面这段话直接复制给 Agent，仓库地址也已包含在指令中：

> 请使用 https://github.com/AdamsukS/data-viz-skill 中的 Data Viz Skill。如果尚未安装，请按仓库 README 的命令下载，并阅读 SKILL.md；如果已安装，直接使用 $data-viz。根据我提供的数据和可视化要求，优先复制并直接修改最合适模板的 Python 代码，非必要不改变布局、配色和信息密度。缺少部分数据时，在原图中移除不支持的元素；少量新增数据优先局部扩展。只有大量新增信息无法容纳，或我明确要求时，才创造新图。请交付图片、可复用的 Python 代码、数据格式说明和运行命令。

具体任务示例：

> 使用 https://github.com/AdamsukS/data-viz-skill 提供的 $data-viz 分析这份销售数据，选择适合展示地区差异和月度趋势的图式，输出图片与可替换数据的 Python 代码。

> 使用 https://github.com/AdamsukS/data-viz-skill 提供的 $data-viz，直接复用最接近这些设备指标的多面板模板代码。缺失的指标面板可以移除，保留其他面板布局；新增少量指标优先局部扩展，不必重新设计整张图。

Agent 会按 **替换数据和标签 → 局部修改现有代码 → 在原布局增删元素 → 必要时新建构图** 的顺序工作，在独立工作目录中渲染，并对照模板预览检查是否发生不必要的改版。交付时说明复用了哪个模块、增删了哪些元素，以及布局调整的具体原因。安装目录中的模板保持不变，原案例统计值不会套用到新数据。

缺失数据不等于零：只有均值时可保留柱形并去掉散点；没有可信误差就移除误差条；缺少插图数据时同时移除插图及其 CSV 读取逻辑。现有渲染器不会自动处理所有缺列情况，Agent 需要在复制的代码中调整数据依赖，不能填入虚构值来维持外观。

- [Skill 指令入口](SKILL.md)：数据检查、选图、绘制、验证与交付流程。
- [自然语言数据指南](docs/TEXT_DATA.md)：词语空间图、词组层次树和文档—词语关联图。
- [选图指南](docs/TEMPLATE_SELECTION.md)：按数据结构和表达目标选择模板。
- [风格指南](docs/STYLE_GUIDE.md)：配色、视觉层次、信息密度与新图设计。
- [数据格式与原图背景](docs/DATA.md) · [新增模板](CONTRIBUTING.md)。

支持 `SKILL.md` 的其他 Agent 也可使用这个自包含目录。例如，使用 `git clone --depth 1 https://github.com/AdamsukS/data-viz-skill.git ./data-viz` 下载，然后让 Agent 阅读其中的 `SKILL.md`。具体技能发现路径以目标 Agent 的配置为准。

## 创建绘图工作目录

以下命令从安装的 Skill 复制第 7 张图及运行依赖代码到新目录，不会修改安装包，也不会覆盖已有目录：

```bash
python3 "${CODEX_HOME:-$HOME/.codex}/skills/data-viz/scripts/prepare_workspace.py" --out ./visualization --figures 7
cd visualization
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python render.py --figure 7 --format png svg pdf --annotations none
```

可一次选择多个模板，如 `--figures 4 12 19`；默认应选择已有模板；仅在确认需要新构图后，才省略 `--figures` 创建空白工程并使用 `scripts/new_figure.py`。上面的渲染使用附带演示数据；真实使用时由 Agent 按字段格式准备自己的数据并更新标签、单位和范围。

## 模板库与手动运行

这些初始案例是从位图参考近似重绘的，**不是像素完全一致的复刻**。当前没有取得原始实验数据；CSV 包括截图估读、数字化提取、模拟观测和拟合曲线，早期环形树的拓扑为模拟，新增词组树的合并关系为按图估读。逐图元数据保留来源说明，示例值不能用于原研究的统计结论。

## 快速开始

需要 Python 3.10+。以下命令在仓库根目录执行：

```bash
git clone https://github.com/AdamsukS/data-viz-skill.git
cd data-viz-skill
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python render.py --list
python render.py --figure all --format png svg
python render.py --figure 7 --format png svg pdf
```

Windows 可用 `py -m venv .venv`，PowerShell 激活命令为 `.venv\Scripts\Activate.ps1`。命令行 `cp` 示例在 Windows 中可用文件管理器复制目录代替。

输出默认写入 `output/`；该目录不进入版本控制。PNG 默认保持参考画布的宽高，`--scale 3` 导出三倍像素尺寸。SVG 保留可编辑文字，PDF 嵌入 TrueType 字体。优先使用已安装的 Arial，否则回退到 DejaVu Sans；跨平台字体与抗锯齿可能不同。

## 每张图都可以单独运行

```bash
python -m figures.figure07.plot --format png svg
python -m figures.figure12.plot --out my_output --format pdf
```

`figures/figure07/plot.py` 中是第 7 张图实际的 `draw(context, rows)` 绘图实现，不是调用一个包含所有图的中央大函数。共用的画布、CSV、箱线图、KDE 等小工具放在 `vizlib/common.py`；批量入口 `render.py` 自动发现图目录。

```text
SKILL.md                 # Agent 技能入口
agents/openai.yaml       # Agent 展示信息
docs/TEMPLATE_SELECTION.md # 按数据与表达目标选图
docs/STYLE_GUIDE.md       # 新图的配色、层次与信息密度
figures/
  figure07/
    plot.py               # 这一张图的实际绘图代码
    style.json            # 尺寸、字体、颜色、标签、分类顺序
    data/
      figure07.csv        # 已注明来源的可替换示例数据
    provenance.json       # 数据类别、字段、单位、原始数据需求、文件校验值
    example_hashes.json   # 已发布示例的指纹，用于识别旧统计标注
    README.md             # 中文说明
    README.en.md          # English documentation
    preview.png           # 本图示例预览
vizlib/common.py          # 共用绘图小工具
render.py                 # 自动发现与批量导出
check_reuse.py            # 数据替换、导出、数据说明和扩展检查
scripts/prepare_workspace.py # 将模板复制到独立用户工程
scripts/new_figure.py     # 创建新图目录
scripts/build_gallery.py  # 更新中英文目录
```

## 替换成自己的数据

**不限定生信或任何特定学科，只要数据符合图式要求就可以绘制。** 先阅读目标图 README 的“通用数据要求”和“文件与字段”，将自己的数据整理为相同的 CSV 列名、类型、结构与数值范围。原图中的基因、细胞群等可以换成业务指标、产品类别、设备或调查分组；同步修改显示标签、单位和样式即可。

| 图式 | 可使用的通用数据 |
|---|---|
| 柱状图、箱线图、雨云图 | 分组观测值，或该图要求的均值、误差、分位数等汇总值 |
| 散点图、边缘分布图 | 两个数值变量及分组标签；不要求来自 PCoA |
| 气泡图、热图 | 类别之间的数值矩阵，以及控制颜色、大小的指标 |
| 山脊图 | 按组排列的观测值或密度曲线，按对应图支持的输入格式提供 |
| 环形树与注释图 | 符合节点、父子关系及注释表格式的层级数据；不要求是系统发育数据 |

各图文档中的“原图研究背景”保留原案例的数据来源和分析步骤，供理解原图时参考，**不是复用模板的前置条件**。`provenance.json` 中的 `raw_data_required` 和 `preprocessing` 也描述原图场景，不是对其他领域数据的限制。复用时保留列名，将其含义映射到自己的领域；并遵守数学约束，例如对数轴数值为正、误差非负、树结构无环。

```bash
cp -R figures/figure07/data my_data
# 修改 my_data/figure07.csv
python render.py --figure 7 --data-dir my_data --out my_output --format png svg pdf
```

`--data-dir` 可以是单张图的数据目录；批量替换时也可以是包含 `figure01/`、`figure02/` 等 CSV 子目录的父目录。组合图的辅助 CSV 必须一起更新，例如图 17 的密度、五数概括和 rug 数据必须来自同一批样本。

CSV 使用 UTF-8，保留代码要求的列名；分类值可替换，但须与样式配置一致。数字列不接受空值、NaN 或 Infinity。修改组名、指标名、单位、面板数或图例样本量时，需要同步修改样式。固定版式不会自动容纳任意数量的分类；复杂面板位置和点大小在该图的 `plot.py` 中调整。

```bash
cp figures/figure07/style.json my_style.json
python render.py --figure 7 --style my_style.json --data-dir my_data --out my_output
```

### 均值和误差

图 1、6、7、8 使用 `category,group,sample,value,mean,error` 长表。`value` 是样本值；`mean/error` 是重复填写的同组汇总。参考图中遮挡点无法完整恢复，所以估读的散点均值可能不同于估读的柱高。

| 模式 | 行为 |
|---|---|
| `--summary auto`（默认） | 未改动的示例用给定 `mean/error`；CSV 改动后从 `value` 计算均值和 SEM |
| `--summary samples` | 总是从样本值计算均值和 SEM，SEM = 样本标准差 / √n |
| `--summary provided` | 使用输入的 `mean/error`；需自行注明误差是 SD、SEM 还是 CI |

给定误差必须非负，同一组的重复汇总字段必须一致。一个样本的 SEM 按 0 展示，这不代表没有实验不确定性。图 4、9 接收预计算汇总值；图 18 接收五数概括，不自动从原始实验记录计算。

### 统计标注

代码**不会从截图或新数据自动推断显著性**。未改动的示例可以显示原图 P 值和星号作为外观参考，但这些不是重新计算的检验结果。

- `--annotations auto`（默认）：任一相关 CSV 改动后，停用原图统计标注。
- `--annotations none`：始终不显示参考统计标注。
- `--annotations reference`：强制显示参考标注，只适合外观比较，不适合报告新实验结果。

每张图的 `example_hashes.json` 位于数据目录外；替换数据目录中的同名文件不会被当作可信指纹。不要修改示例指纹来让真实数据显示旧结论。图 9 的三点为构造的参考外观符号，新数据默认隐藏。每次运行会写出 `render_log.json`，记录实际模式，日志不包含机器绝对路径。

### 山脊图的原始样本入口

图 14–16 接收两种形式：已经整理好的 `x,density` 曲线，或将这两列替换为 `value` 的样本长表。分组列保持不变，例如图 14：

```csv
panel,group,value
0,0,2.71
0,0,2.83
0,0,2.90
```

真实文件应包含样式要求的全部分组。程序按组计算高斯核密度估计，默认把峰归一化到 1.4 个行距；可增加 `height` 列控制展示高度。该高度不表示样本数量或组间可比较的绝对概率密度。如果使用流式细胞术数据，需先完成适用的补偿、门控和 log/logicle/arcsinh 变换；其他领域按自己的数据含义进行预处理，本项目不执行这些上游分析。

## 图表索引

每个链接都包含：通用数据要求、逐文件字段/单位、CSV 示例行、单图运行方法，以及原图背景与示例数据来源。

<!-- FIGURES:START -->

| ID | 图式 | 数据来源 |
|---|---|---|
| 01 | [空心柱状图与散点](figures/figure01/README.md) | `screenshot_estimate` |
| 02 | [PCoA 与边缘箱线图](figures/figure02/README.md) | `digitized` |
| 03 | [多队列 PCoA 组合图](figures/figure03/README.md) | `digitized` |
| 04 | [时间序列与 iAUC 插图](figures/figure04/README.md) | `mixed` |
| 05 | [半小提琴 雨云图](figures/figure05/README.md) | `synthetic_from_estimated_distribution` |
| 06 | [断轴分组柱状图](figures/figure06/README.md) | `screenshot_estimate` |
| 07 | [基因表达分组柱状图](figures/figure07/README.md) | `screenshot_estimate` |
| 08 | [PARP1 变体柱状图](figures/figure08/README.md) | `screenshot_estimate` |
| 09 | [双轴嵌套柱状图](figures/figure09/README.md) | `mixed` |
| 10 | [环形系统发育树与分类注释](figures/figure10/README.md) | `synthetic` |
| 11 | [环形系统发育树与热图](figures/figure11/README.md) | `synthetic` |
| 12 | [分组基因气泡矩阵](figures/figure12/README.md) | `digitized` |
| 13 | [双对数散点与边缘直方图](figures/figure13/README.md) | `synthetic` |
| 14 | [三组分子动力学山脊图](figures/figure14/README.md) | `synthetic_fitted_curve` |
| 15 | [流式细胞术山脊矩阵](figures/figure15/README.md) | `synthetic_fitted_curve` |
| 16 | [细胞群表达山脊矩阵](figures/figure16/README.md) | `synthetic_fitted_curve` |
| 17 | [区域年代分布 雨云图](figures/figure17/README.md) | `mixed` |
| 18 | [进化年龄分面箱线图](figures/figure18/README.md) | `screenshot_estimate` |
| 19 | [相关性 哑铃 气泡 与条形组合图](figures/figure19/README.md) | `screenshot_estimate_and_digitized` |
| 20 | [词语空间与分区网格](figures/figure20/README.md) | `transcribed_estimated_and_digitized` |
| 21 | [词组层次聚类树](figures/figure21/README.md) | `transcribed_and_screenshot_estimate` |
| 22 | [文档—词语关联与分布](figures/figure22/README.md) | `transcribed_estimated_and_synthetic` |

## 代码绘制效果

以下图片均由各图的 Python 代码与随附示例数据生成。

### 01 · 空心柱状图与散点

![空心柱状图与散点](figures/figure01/preview.png)

[代码与数据说明](figures/figure01/README.md)

### 02 · PCoA 与边缘箱线图

![PCoA 与边缘箱线图](figures/figure02/preview.png)

[代码与数据说明](figures/figure02/README.md)

### 03 · 多队列 PCoA 组合图

![多队列 PCoA 组合图](figures/figure03/preview.png)

[代码与数据说明](figures/figure03/README.md)

### 04 · 时间序列与 iAUC 插图

![时间序列与 iAUC 插图](figures/figure04/preview.png)

[代码与数据说明](figures/figure04/README.md)

### 05 · 半小提琴 雨云图

![半小提琴 雨云图](figures/figure05/preview.png)

[代码与数据说明](figures/figure05/README.md)

### 06 · 断轴分组柱状图

![断轴分组柱状图](figures/figure06/preview.png)

[代码与数据说明](figures/figure06/README.md)

### 07 · 基因表达分组柱状图

![基因表达分组柱状图](figures/figure07/preview.png)

[代码与数据说明](figures/figure07/README.md)

### 08 · PARP1 变体柱状图

![PARP1 变体柱状图](figures/figure08/preview.png)

[代码与数据说明](figures/figure08/README.md)

### 09 · 双轴嵌套柱状图

![双轴嵌套柱状图](figures/figure09/preview.png)

[代码与数据说明](figures/figure09/README.md)

### 10 · 环形系统发育树与分类注释

![环形系统发育树与分类注释](figures/figure10/preview.png)

[代码与数据说明](figures/figure10/README.md)

### 11 · 环形系统发育树与热图

![环形系统发育树与热图](figures/figure11/preview.png)

[代码与数据说明](figures/figure11/README.md)

### 12 · 分组基因气泡矩阵

![分组基因气泡矩阵](figures/figure12/preview.png)

[代码与数据说明](figures/figure12/README.md)

### 13 · 双对数散点与边缘直方图

![双对数散点与边缘直方图](figures/figure13/preview.png)

[代码与数据说明](figures/figure13/README.md)

### 14 · 三组分子动力学山脊图

![三组分子动力学山脊图](figures/figure14/preview.png)

[代码与数据说明](figures/figure14/README.md)

### 15 · 流式细胞术山脊矩阵

![流式细胞术山脊矩阵](figures/figure15/preview.png)

[代码与数据说明](figures/figure15/README.md)

### 16 · 细胞群表达山脊矩阵

![细胞群表达山脊矩阵](figures/figure16/preview.png)

[代码与数据说明](figures/figure16/README.md)

### 17 · 区域年代分布 雨云图

![区域年代分布 雨云图](figures/figure17/preview.png)

[代码与数据说明](figures/figure17/README.md)

### 18 · 进化年龄分面箱线图

![进化年龄分面箱线图](figures/figure18/preview.png)

[代码与数据说明](figures/figure18/README.md)

### 19 · 相关性 哑铃 气泡 与条形组合图

![相关性 哑铃 气泡 与条形组合图](figures/figure19/preview.png)

[代码与数据说明](figures/figure19/README.md)

### 20 · 词语空间与分区网格

![词语空间与分区网格](figures/figure20/preview.png)

[代码与数据说明](figures/figure20/README.md)

### 21 · 词组层次聚类树

![词组层次聚类树](figures/figure21/preview.png)

[代码与数据说明](figures/figure21/README.md)

### 22 · 文档—词语关联与分布

![文档—词语关联与分布](figures/figure22/preview.png)

[代码与数据说明](figures/figure22/README.md)

<!-- FIGURES:END -->

## 新增第 23 张图

```bash
python -m scripts.new_figure --id 23 --title "新的图表" --title-en "New chart"
python -m figures.figure23.plot --format png svg
```

创建器会生成一个可运行的两组模拟数据案例，且不会覆盖已有图。将它替换为需要的图式：编辑 `plot.py`、`style.json`、`data/`、`provenance.json` 和两份 README，然后更新预览与目录：

```bash
cp output/figure23.png figures/figure23/preview.png
python -m scripts.build_gallery
python check_reuse.py
```

`render.py --figure all` 自动包含新图，不需要中央注册项。如何记录真实来源、导入树和更新指纹见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 验证与范围

```bash
python check_reuse.py
python check_skill.py
python check_text_figures.py
```

`check_text_figures.py` 验证词组树结构、模拟关联表可复现性、缺失网格/分布的局部关闭，以及三张文本图的独立运行。

`check_skill.py` 另外在隔离目录中接入服务延迟数据，验证独立模板导出 PNG/SVG/PDF、空白工程新建图表，以及已有目录不会被覆盖。

检查会逐图渲染原始示例及修改后的 CSV，验证输出尺寸、SVG 未嵌入位图、数据变化确实改变图像、旧统计标注停用，以及文档、字段、指纹匹配。还会在临时项目中新建并运行一张图，验证新增流程。GitHub Actions 在推送和 PR 时执行相同检查。

19 个初始模块拆分后，在相同本机字体/依赖环境下，输出与拆分前预览逐像素一致。**这是对重构的回归检查，不表示与文档原图逐像素一致。** 科学计算方法、数据来源真实性和最终科研结论仍需由实际数据与原分析流程确认。

原始 Word、参考截图、本地对照网页、虚拟环境和临时文件不在公开仓库中。预览是代码重绘结果。已公布的模拟/估读 CSV 足以运行全部模板，不依赖参考图片。原始论文/DOI 尚未核实，`source_publication` 明确为 null。

## 许可证

[MIT](LICENSE) 适用于本项目代码和项目撰写的说明。模拟/估读数据的性质逐文件记载；本仓库不声称拥有或授予参考论文、原始研究数据或第三方图片的权利。

---

图片复刻来源于 **Nature**。以上展示为本项目代码生成的复现图；具体论文及 DOI 尚待补充。
