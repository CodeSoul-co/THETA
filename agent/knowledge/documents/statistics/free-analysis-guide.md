# THETA 自由分析模式

默认仍为主题分析：发现主题、解释结果，再围绕主题做挖掘。自由分析从研究问题出发，主题模型不再是前置步骤；数值统计、预测、生存、计量、数学优化和文本分析共用现有 Agent 对话界面、研究记录及确认卡。

Python worker 登记了 138 个可执行方法标识；[逐项方法与示例](methods.md)记录参数和边界。可用方法以 `statistics_methods` 与 `statistics_inspect` 返回的当前规格为准；使用前仍需核查数据条件与方法假设。

## 使用方式

网页：进入现有 `/workbench` → 保持“对话” → 在输入框底部的分析方式选项中选择“自由分析” → 上传 CSV/TSV/XLSX/JSON/JSONL → 描述研究问题。模式保存在当前会话，恢复历史不会凭空变回主题模式。执行中、等待确认或待完成解读时不可切换。

CLI：`/mode free`；恢复默认用 `/mode topic`。新会话默认主题模式。

示例提问：

- “我是初学者。y 是连续结果，x 是主要解释变量，x2 是控制变量。先描述数据，再用 HC3 标准误回归，解释效应量、区间、样本排除和局限，给我论文三线表与复现脚本。”
- “预测 binary，比较 Logistic、SVM、随机森林和 XGBoost。x、x2 是输入，文本 text 也可能有信息。保留独立测试集，调参只能在训练集里做，不要挑测试集上最好的一次随机种子。”
- “研究 duration 到 event 的时间；event=1 发生事件，0 是右删失。比较 KM 曲线，再做 Cox，并检查比例风险条件。”
- “entity 是个体，wave 是时间，估计 y 关于 x 和 x2 的双向固定效应，按个体聚类标准误。解释哪些识别条件还没有证据。”
- “目标最小化 -x1-2x2，约束 x1+x2≤5、变量非负。给最优解、可行性检查和参数设定。”

Agent 会读取数据 → 查方法规格 → 保存可证伪假设、分析计划、敏感性检查与停止条件 → 展示具体确认卡。确认后只执行该批次。拒绝不会计算；修改会使旧批准失效，新方案需要新卡。每批最多 6 项计算，最长 120 秒。当前受控 worker 输入上限为 20,000 行、200 列、50 MiB；不静默抽样。更大的分析需要明确分片/计算后端方案。

## 只有数据时怎样开始

可以直接说“我只有刚上传的文件，还没确定研究问题”。Agent 会检查实际字段、缺失与重复、观测单位、代表性原文，提出两三个问题及推荐起点。预览诊断已有的数值应标明来自预览，正式估计仍经过保存计划与确认卡。

有实质评论、访谈或开放题文本时，自由模式会建议主题模型，并说明如何结合评分、组别或时间做后续分析；编号和类别标签不当作语料。用户明确说“不要主题、只做回归/描述”等，应尊重限制。模拟数据和模板化文本仅用于练习，不能据此宣称发现真实总体结构。

## 异步任务与恢复

网页消息和确认操作先保存任务，再返回 HTTP 202 回执。前端通过短轮询及事件流同步状态；离开页面或刷新不会取消已受理的任务。输入框上方显示任务阶段、最近保存时间；统计 worker 每完成一个步骤原子保存步骤结果和进度。任务、计划、批准及研究笔记保存在 Agent home 的 SQLite 中，最终报告和复现材料保存在 `statistics/<analysisId>/`。

“停止生成”明确请求服务端暂停；关闭网页只断开查看。服务必须继续运行，关闭电脑/服务不能保证计算继续。服务重启会将已失去宿主进程的活动任务标记为“中断”，保留记录并释放该任务自己的锁，不自动重放。点“核查记录并继续”，Agent 先使用 `statistics_status` 核查；完整且哈希吻合的结果可以重新登记并继续解读，部分步骤只作进度证据，未完成批次需新计划/新确认，不能冒充完整报告。

相同 requestId 与内容只返回原回执，变更内容复用同一标识会被拒绝；同一会话同时只能处理一个任务。这里是本地持久化任务机制，不是无限重试或跨机器调度器。统计单批仍受 120 秒和 6 步限制；长分析应预先拆分批次、保存检查点，既有主题训练沿用其 job/monitor 生命周期。

## 独立 worker 安装

在仓库根目录执行（Python 3.11–3.13）：

```sh
python3 -m venv agent/.local/runtimes/statistics
agent/.local/runtimes/statistics/bin/python -m pip install -r agent/workers/statistics/requirements-lock.txt
cd agent
npm run build
npm run test:statistics
```

`requirements.txt` 是受限版本依赖范围；`requirements-lock.txt` 是锁定依赖的完整版本快照。跨平台安装需重新验证；计算由独立 Python 环境执行。不要使用 `--system-site-packages`。启动时默认选择上述 venv；也可以由宿主设置 `THETA_WORKER_STATISTICS_PYTHON` 的绝对路径与 `THETA_WORKER_STATISTICS_REVISION`。缺失环境不会回退到主题 worker，也不自动安装依赖或下载模型。

每次批准绑定数据 SHA-256、完整计划、研究键、Python 路径、解释器版本、依赖指纹、统计实现源码指纹及环境版本。Python 端复核宿主 SQLite 批准并一次性消费；重放同一个凭据不会重算。数值线程限制为 1。venv 是依赖隔离，不是操作系统沙箱；注册工具只接受白名单数据/参数，不接受任意 Python 或公式执行。代码型 `analysis_execute` 仍要求原有宿主显式授权的 Docker workspace，不能因为切换自由模式就获得任意主机代码执行能力。

## 工具协议

| 工具 | 作用 |
|---|---|
| `statistics_methods` | 分页查询已实现方法与适用边界，可按 family/query 筛选 |
| `statistics_inspect` | 查看确切输入、参数范围、示例及限制 |
| `statistics_plan` | 验证数据/规格并保存实证计划；不估计、不批准 |
| `statistics_request_approval` | 创建现有对话中的确认卡；不计算 |
| `statistics_status` | 核查当前计算进度、进程及保存的结果；恢复前验证完整产物哈希，不重用旧批准计算 |
| `statistics_synthesize` | 根据已交付批次重新写论文解读，隔离旧助手文字并保存修订文件；不重新计算或申请批准 |
| `statistics_results` | 读取该研究已交付的指标、系数、完整表名与分页行 |
| `analysis_checkpoint` / `analysis_history` | 保存研究笔记，核查原始回执；笔记不是事实或授权 |
| 既有 `models_*` / `plan_propose` / `training_*` | 需要文本主题结构时仍可自主选择使用 |

`statistics_plan` 示例（datasetRef 替换为当前附件回执）：

```json
{
  "datasetRef": "dataset-<sha256>",
  "plan": {
    "question": "x 与 y 的条件关联有多大？",
    "hypotheses": ["控制 x2 后 x 与 y 正相关；这是探索性假设"],
    "assumptions": ["观测独立", "线性条件均值", "不存在足以颠覆解释的遗漏变量尚需论证"],
    "steps": [
      {"method":"describe","x":["x","x2","y"]},
      {"method":"ols","name":"Model 1","x":["x","x2"],"y":"y","params":{"covariance":"HC3"},"missing":"drop","alpha":0.05,"seed":42}
    ],
    "sensitivity": ["检查残差与影响点，必要时另批稳健估计"],
    "stoppingRule": "完成预设模型和诊断即报告，不以显著性为停止标准"
  }
}
```

纯线性/整数/凸二次规划和先验功效计算可省略 datasetRef，直接使用参数，无需制造或上传虚构数据。

批次不支持把上一步的派生 CSV 自动当作下一步数据：先交付、检查行标识并显式附加派生文件。原始文件始终保留。主题权重与元数据联结必须使用已验证的行来源，不能凭长度一致就按行拼接。纯文本也可使用预测工具的 `params.text_column`：字符 2–3 gram TF-IDF 在训练折内拟合，数值 `x` 与文本特征结合；不是 SBERT 的替代实现。SBERT/主题模型仍走仓库既有模型路径。

## 常用分析方法

| 常见工作 | THETA 方法 |
| --- | --- |
| 描述/频数 | describe / frequency |
| 相关 | correlation，params.correlation 指定秩相关 |
| t 检验 | ttest_one / ttest_ind / ttest_paired |
| 卡方/精确检验 | chi_square / fisher / mcnemar |
| 线性回归 | ols；HC1、HC3、cluster、HAC 显式选择 |
| 二分类/有序/多项 | logit / probit / ordinal_logit / multinomial_logit |
| 分位数/稳健回归 | quantile / robust_linear（估计细节不同） |
| 计数 | poisson / negative_binomial |
| 面板 | panel_fe / panel_twfe / panel_re |
| 工具变量 | iv_2sls / iv_gmm（线性 IV） |
| 生存 | kaplan_meier / cox / AFT 系列 |
| SEM | sem（连续数据 MLW，受限关系语法） |
| 表格 | 自动 esttab.csv / .tex / .rtf |

方法、参数和估计结果以当前登记规格为准。OLS 明确包含截距；分类自变量的统计回归须显式编码；Logit 因变量用 0/1。机器学习工具自动对分类输入编码，但统计推断回归不会偷偷猜参照组。OLS/WLS 显式使用 Student t 推断（聚类时按估计器的推断自由度）；其他模型在 metrics.inference_distribution 标明 t 或渐近 z。

## 论文输出与解释

每批交付 HTML/Markdown 报告、完整 CSV 表、原始精度 JSON、esttab 风格并列系数表（CSV、LaTeX、RTF）、适用方法的 PDF/SVG/300 dpi PNG 图、计划 JSON、`reproduce.py` 和文件哈希清单。TeX 是 `booktabs` 表格片段，主文档加载 `booktabs`；中文标签建议 XeLaTeX/ctex。RTF 可使用兼容的文档编辑器打开。星号定义为 `* p<0.05, ** p<0.01, *** p<0.001`，表注明确，未做全批多重检验校正。空白是未估计，不能写成零。

复现时从仓库 `agent/` 目录执行 `.local/runtimes/statistics/bin/python -s /绝对路径/reproduce.py`。输入版本会再次校验；结果写到旁边的 `reproduced/`，记录当前运行环境及 `reproduction.originalRuntime`、`sameSource`，不会将变更后的代码冒充为原版本。保留原仓库提交与锁定依赖可重建相同环境。

HTML/Markdown 的数值段落是确定性证据报告；Agent 根据实际回执另作研究解释并保存为 interpretation.md，不能把自动表注当作完成了实质论文讨论。统计解读隔离历史助手文字与研究笔记，只读取当前获批批次及参考资料，禁止在解读阶段调用计划/执行工具。提示词要求每张图/表解释分析对象、分母/排除、尺度/参照组、方向/效应量、置信区间、准确 p 值、估计与标准误类型、实际意义、假设、反例和外推边界。模型不收敛、不可识别或推断量非有限时拒绝交付显著性系数表。

特别限制：Cox/AFT 目前仅右删失；Fine–Gray 尚未实现。DID 仅两组两期，RDD 为预设带宽 uniform kernel 的 sharp 局部线性估计。PSM 为有放回最近邻、logit 倾向分数卡尺，提供匹配 ATT 与 SMD，没有匹配推断 SE。SEM 不含 WLSMV、测量不变性、多层 SEM。重复测量 ANOVA 仅平衡被试内设计，没有球形性校正。ML 默认独立随机拆分，分组/时间数据必须显式选择；禁止用测试集调参。参数网格最多 12 组、CV 最多 5 折。`tune` 省略、`{}` 或 `false` 表示不调参；`cv=0` 关闭交叉验证，启用调参需 `cv=2..5`。非法网格/CV 在创建计划时检查。

## 方法来源

实现依据：[statsmodels](https://www.statsmodels.org/stable/index.html)、[scikit-learn](https://scikit-learn.org/stable/common_pitfalls.html)、[lifelines](https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html)、[linearmodels](https://bashtage.github.io/linearmodels/)、[SciPy optimize](https://docs.scipy.org/doc/scipy/reference/optimize.html)、[semopy](https://semopy.com/)。

Agent 知识库复用了 [K-Dense statistical-analysis skill](https://github.com/K-Dense-AI/scientific-agent-skills/blob/0b2afe68a5f9379097ad815e028af664f1e222b7/skills/statistical-analysis/SKILL.md) 的 MIT 参考文本，保留许可证、固定 commit 与导入 SHA。它只提供研究设计/报告参考，不会安装插件、执行其脚本、绕过确认或赋予工具不存在的能力。
