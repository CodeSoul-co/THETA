# THETA 模型、参数与适用场景完整清单


本清单是静态代码核对与选型说明，场景与调参方向是根据实现和原始资料作出的建议，不代表本仓库已经做过效果对比。

## 1. 统计与参数口径

- 统一训练入口 src/models/run_pipeline.py 支持 12 个模型，包含 50 个独立 CLI 选项。连字符/下划线别名计为一个选项。
- 以下默认值优先指直接调用 run_pipeline.py 时的 argparse 默认值。不是第三方库默认值，也不是 config/default.yaml 的默认值。
- main.py 是 THETA 的另一入口；prepare_data.py 负责数据准备，参数不能直接混用。
- “CLI”表示统一入口可设置；“main”表示 THETA main.py 的专用参数；“API”表示必须直接调用 Python 类/训练器，统一入口没有暴露；“配置”表示 PipelineConfig/ModelConfig 字段，需要核对调用链。
- 所有模型都依赖正确的文本清洗、词表和数据行对齐。语言参数不等于重新训练/更换一个适合该语言的嵌入模型。
- 不计不在 ALL_MODELS 中的 neural_lda；它虽然在底层文件中存在，但统一训练入口不接受。旧 registry.py 部分模块路径仍是 model.baselines，实际目录是 model/baseline，且 etm 名称与主模型混用，不能拿旧注册表作为支持列表。

## 2. 支持的 12 个模型与适用场景

| ID | 本项目实现/输入 | 建议适用场景 | 主题数与主要限制 |
|---|---|---|---|
| theta | 文档嵌入 + BOW + 项目自有 ETM；支持云嵌入或本地 Qwen | 需要语义信息的主题分析；zero_shot 可用于先建立基线；有标签/无标签领域适配是另外两种模式的设计目标 | 固定 K；zero_shot 仍训练主题模型，只是不微调嵌入模型；双阶段实现有需核验的问题，见第 11 节 |
| lda | sklearn LDA，BOW | 新闻、政策、论文摘要等词频信号较充分的语料；传统可解释基线；不准备嵌入时 | 固定 K；缺少上下文语义，极短文本共现稀疏 |
| hdp | gensim HdpModel，BOW | 不清楚主题数、希望先探索主题结构，再筛选有效主题 | max_topics 是截断上限；当前结果直接保留 get_topics() 的维度，不自动剔除低权重主题，不能把输出维度当成精炼后的主题数 |
| stm | 优先 R stm/rpy2，否则 Python logistic-normal + 协变量回归近似 | 比较不同地区、年份、平台、机构等元数据与主题占比的关联 | 固定 K；本项目必须有协变量；接入的是 prevalence，不是完整 content covariate 模型；关联不等于因果 |
| btm | 自实现 Gibbs 采样，语料级词对 | 评论、微博、标题、简短问答等短文本 | 固定 K；词对越多越慢；本实现从 BOW 重建词序，window_size 不是原始文本上的真实相邻窗口 |
| etm | OriginalETM，BOW + 可选 Word2Vec | 希望利用词级语义、但不使用大语言模型文档嵌入；ETM 方法对照 | 固定 K；预训练词向量缺失时可训练 Word2Vec或随机初始化 |
| ctm | 上下文化主题模型，BOW + SBERT | 同义表达较多、短文本语义分析；具备跨语言对齐嵌入时探索跨语言主题迁移 | 固定 K；zeroshot/combined 是推断网络输入方式，不是免训练；必须有适合语料的嵌入 |
| dtm | 项目自实现神经动态主题模型；BOW + 时间索引 + SBERT，缺 SBERT 时可用 BOW/SVD | 研究主题和主题词在年份/时间段之间如何变化 | 固定 K 与时间片；需要每片足够文档；不是原始 Blei DTM 的直接封装 |
| nvdm | BOW + 高斯潜变量 VAE | 神经文档表示/重构的研究对照、检验不用预训练嵌入的效果 | 固定潜维数 K；潜变量没有 GSM 那样的 softmax 主题概率约束，比较 theta 时需核对语义 |
| gsm | BOW + Gaussian Softmax VAE | 希望获得概率化主题比例的轻量神经基线；与 NVDM 做结构对照 | 固定 K；没有预训练语义信息 |
| prodlda | BOW + logistic-normal/product-of-experts 风格 VAE | 检验神经主题模型是否能改善词表主题表示；与 LDA/NVDM/GSM 比较 | 固定 K；不能保证主题一定更清晰，仍需一致性、排他性与人工检查 |
| bertopic | 文档嵌入 + UMAP + HDBSCAN + 主题表示 | 语义聚类、短评/反馈主题发现、允许存在离群文档的探索性分析 | 算法支持自动聚类；本项目默认 num_topics=20 会传给 nr_topics 作为合并目标；用 --num_topics 0 取消合并目标，并不保证目标为20时恰好输出20个主题 |

依据：[LDA 官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html)、[HDP 官方文档](https://radimrehurek.com/gensim/models/hdpmodel.html)、[STM 官方介绍](https://www.structuraltopicmodel.com/)、[BTM 原论文](https://jiafengguo.github.io/2013/2013-A%20Biterm%20Topic%20Model%20for%20Short%20Texts.pdf)、[ETM 原论文](https://doi.org/10.1162/tacl_a_00325)、[CTM 跨语言论文](https://aclanthology.org/2021.eacl-main.143/)、[NVDM 原论文](https://proceedings.mlr.press/v48/miao16.html)、[GSM 相关原论文](https://proceedings.mlr.press/v70/miao17a.html)、[ProdLDA 原论文](https://arxiv.org/abs/1703.01488)、[BERTopic 官方调参指南](https://maartengr.github.io/BERTopic/getting_started/parameter%20tuning/parametertuning.html)。本项目实现是否与论文一致，以本地代码为准。

## 3. 每个模型实际收到的统一入口训练参数

不重复列 dataset、workspace、GPU、评估/可视化开关等流程参数；它们见第 5 节。

| 模型 | 实际可由 run_pipeline.py 调整的模型/训练参数及默认值 |
|---|---|
| theta | mode=zero_shot；model_size=0.6B；num_topics=20；hidden_dim=512；epochs=100；batch_size=64；learning_rate=0.002；kl_start=0；kl_end=1；kl_warmup=50；patience=10；no_early_stopping=False。epochs/learning_rate/早停主要作用于 zero_shot；双阶段见第6节 |
| lda | num_topics=20；max_iter=100 |
| hdp | max_topics=150；alpha=1.0。num_topics 不控制 HDP |
| stm | num_topics=20；max_iter=100；协变量由预处理文件载入 |
| btm | num_topics=20；n_iter=100；alpha=1.0；beta=0.01 |
| etm | num_topics=20；epochs=100；batch_size=64；learning_rate=0.002；hidden_dim=512；embedding_dim=300；dropout=0.2；patience=10 |
| ctm | num_topics=20；epochs=100；batch_size=64；learning_rate=0.002；hidden_dim=512；num_layers=2；inference_type=zeroshot；patience=10。实际 hidden_sizes=(512,512) |
| dtm | num_topics=20；epochs=100；batch_size=64；learning_rate=0.002；hidden_dim=512；embedding_dim=300；时间片从文件载入 |
| nvdm | num_topics=20；epochs=100；batch_size=64；learning_rate=0.002；hidden_dim=512 |
| gsm | num_topics=20；epochs=100；batch_size=64；learning_rate=0.002；hidden_dim=512 |
| prodlda | num_topics=20；epochs=100；batch_size=64；learning_rate=0.002；hidden_dim=512 |
| bertopic | num_topics=20；n_neighbors=15；n_components=5；min_cluster_size=10；min_samples=None→与min_cluster_size相同；top_n_words=10；random_state=42；language由--language转换 |

关键代码：[统一分发](/Users/erwin/Downloads/codespace/THETA/src/models/run_pipeline.py:648)、[THETA参数转发](/Users/erwin/Downloads/codespace/THETA/src/models/run_pipeline.py:510)。

## 4. 通用训练参数：含义和适用场景

这些是调参方向，不是代码验证过的取值区间。帮助文本中的范围多数没有做强校验。

| 参数 | 统一入口默认值 | 适用模型 | 参数用途/适用场景 |
|---|---|---|---|
| --num_topics | 20 | 除HDP外，BERTopic特殊 | K小便于总览，K大便于细分，但更可能出现重复或弱主题。固定K模型可比较10/20/30/50；这只是探索起点。BERTopic设0取消合并目标 |
| --vocab_size | 5000 | 数据准备/BOW模型 | 减少词表可降低内存和稀疏噪声，增加可保留领域术语。训练复用已有BOW时，不能靠此参数重建词表；预处理--prepare还有转发问题，见第11节 |
| --epochs | 100 | THETA zero_shot、ETM/CTM/DTM/NVDM/GSM/ProdLDA | 神经训练最大轮数；损失仍下降才考虑增加。无效于LDA/HDP/STM/BTM/BERTopic；THETA双阶段另用stage1_epochs/stage2_epochs |
| --batch_size | 64 | 上述神经训练、嵌入预处理 | 显存不足时减小；资源允许时适度增大提高吞吐；不同批量可能需重调学习率。不是LDA内部online batch_size |
| --hidden_dim | 512 | THETA、ETM、CTM、DTM、NVDM、GSM、ProdLDA | 控制网络容量；小语料/低显存可减小，复杂语料可尝试增大，同时检查过拟合。不是嵌入向量输出维数 |
| --learning_rate | 0.002 | 同上；THETA限单阶段 | 损失震荡/数值不稳时降低，收敛慢时谨慎增加。双阶段THETA使用stage1_lr/stage2_lr |
| --dropout | 0.2 | 统一入口只转发给ETM | 过拟合时可增加丢弃比例，欠拟合时减小。其他神经模型虽然有底层dropout，该CLI未接入 |
| --num_layers | 2 | 统一入口只转发给CTM | CTM的hidden_sizes=[hidden_dim]重复num_layers次。加深可增加非线性容量和开销；小数据优先少层 |
| --embedding_dim | 300 | ETM、DTM | 词/解码嵌入维数；维数越高参数和内存越多。ETM载入预训练矩阵时以矩阵实际维数为准；不是云嵌入参数 |
| --patience | 10 | THETA zero_shot、ETM、CTM | 连续未改善多少轮后停止。THETA使用验证损失与min_delta；ETM/CTM当前监控训练损失。DTM/NVDM/GSM/ProdLDA没有收到这个值 |
| --no_early_stopping | False | THETA单阶段 | 需要按固定轮数训练时启用；未传给ETM/CTM，不会关闭它们的早停 |
| --kl_start | 0.0 | THETA；双阶段第二阶段也使用 | KL退火初始权重；初期先学习重构时用较小值 |
| --kl_end | 1.0 | THETA；双阶段第二阶段也使用 | 最终先验正则强度；调大通常加强约束，可能损害重构；调小可能弱化先验约束 |
| --kl_warmup | 50 | THETA；双阶段第二阶段也使用 | KL权重从起点升到终点的轮数；初期训练不稳/潜变量退化时可尝试更慢退火；需正数。ETM自己的退火不是此参数控制 |
| --max_iter | 100 | LDA、STM | 批量变分/EM最大迭代数；复杂语料未收敛时增加，原型验证时减少 |
| --max_topics | 150 | HDP | 语料级截断容量；大主题空间可增加但耗时增加；不是固定有效主题数 |
| --n_iter | 100 | BTM | Gibbs采样轮数；增加通常给采样更多稳定机会，成本随词对数增加 |
| --alpha | 1.0 | HDP、BTM | HDP是文档层集中参数；BTM是语料级主题先验。较小鼓励更集中的分配，较大平滑更多主题。统一CLI不传给LDA |
| --beta | 0.01 | BTM | 主题词分布平滑；小值更尖锐，大值对少见词更平滑。不是模型输出矩阵beta |
| --inference_type | zeroshot | CTM | zeroshot编码只看上下文嵌入；combined编码结合BOW和嵌入。词表稳定、词面线索重要可试combined；跨语言迁移要有对齐多语言嵌入 |
| --n_neighbors | 15 | BERTopic | UMAP邻域规模；小值偏局部细粒度，大值偏整体结构；须考虑语料样本数 |
| --n_components | 5 | BERTopic | UMAP降维后的维数；它用于聚类，不必等于展示用2D。增加可保留更多结构，也可能增加密度聚类难度 |
| --min_cluster_size | 10 | BERTopic | 期望认可的最小簇规模；提高通常减少碎片小主题，降低可发现小众主题 |
| --min_samples | None→10 | BERTopic | HDBSCAN保守程度；提高通常更多点成为噪声，降低常减少离群点。若None会随min_cluster_size变化 |
| --top_n_words | 10 | BERTopic | 每个主题保留的代表词数；展示/解释可调整，不等于聚类数。非BERTopic不会收到此值 |
| --random_state | 42 | BERTopic | 控制其UMAP随机性；固定便于比较参数。不是全模型统一种子 |

BERTopic参数方向依据[官方调参说明](https://maartengr.github.io/BERTopic/getting_started/parameter%20tuning/parametertuning.html)，CTM推断方式与跨语言条件参考[原论文](https://aclanthology.org/2021.eacl-main.143/)及本地类实现。

## 5. 其余统一入口选项：补齐全部50项

本节24项与上节26项合计50个独立CLI选项。

| 参数 | 默认/可选值 | 适用场景与当前行为 |
|---|---|---|
| --dataset | 必填字符串 | 选择数据集及其工作目录；应与预处理输出名称一致 |
| --models | 必填，逗号分隔 | 12个ID，可依次训练多个模型。共享CLI值不代表每个模型都有该参数 |
| --mode | zero_shot；另supervised/unsupervised | THETA；前者固定嵌入训练主题模型，后两者启用项目双阶段适配；不作用于其他模型 |
| --model_size | 0.6B；另4B/8B | 本地Qwen模型规模/实验路径标识。小模型适合试跑和资源有限，大模型需要更多资源，效果需验证；云模型实际由embedding-model控制 |
| --embedding-provider | None，运行时按配置/环境解析；zero_shot默认cloud | THETA预处理。cloud/local/qwen，或openai/dashscope/siliconflow/zhipu/volcengine/openai_compatible。supervised/unsupervised要求本地提供方 |
| --embedding-cloud-provider | None；cloud默认预设openai | 选择云服务预设；支持openai/dashscope/siliconflow/zhipu/volcengine/openai_compatible |
| --embedding-model | None | 指定云嵌入模型名称；决定语义模型，不是12个主题模型的ID |
| --embedding-api-base | None | 使用特定OpenAI兼容端点/代理或私有服务时设置 |
| --embedding-api-key-env | None | 保存API密钥的环境变量名称，例如某个服务的KEY变量；传变量名，不是密钥内容 |
| --embedding-dimensions | None | 服务支持时指定输出维数，权衡存储/计算与语义保留；不改变主题数 |
| --gpu | None | 显式物理GPU编号；覆盖CUDA_VISIBLE_DEVICES；省略继承环境。不应把负数当作CPU开关 |
| --language | zh；en/zh/chinese/english | THETA图表语言只可靠接收en/zh；BERTopic也借此选择english/multilingual；普通基线图表由--lang控制 |
| --lang | en；en/cn/both | 基线图表输出语言；both输出双语版本。THETA主流程不使用这个值 |
| --skip-train | False | 基线复用现有实验，仅评估/画图；THETA分支会跳过整个main.py pipeline，当前不能据此完成“只评估” |
| --skip-eval | False | 跳过评估以节省时间；可能减少可视化可用的指标 |
| --skip-viz | False | 跳过画图，先关注模型/指标或批量实验 |
| --check-only | False | 检查数据文件后返回，不训练；检查通过不代表整个训练依赖都经过验证 |
| --prepare | False | 只调用数据准备后返回，不接着训练；嵌入provider系列参数在这个分支转发 |
| --data_exp | None | THETA选择预处理实验；基线实际主要使用workspace_dir，旧data_exp不在run_baseline中消费 |
| --exp_name | None | 实验名标签；THETA特定分支还用其复制按K保存的结果；基线作为任务名候选 |
| --user_id | default_user | 基线工作区/结果隔离；THETA路径体系未完全统一，只在部分复制分支使用 |
| --workspace_dir | None | 基线指定已有BOW/嵌入工作区，覆盖自动寻找；THETA不消费 |
| --force | False | 统一入口虽声明但未使用/转发；需要覆盖矩阵时应使用prepare_data.py自己的--force |
| --task_name | None→exp_时间戳 | 基线结果任务名，适合命名对照实验；THETA训练分支不使用它 |

所有embedding选项也接受对应下划线别名。后附自动提取清单用于逐项核对。

## 6. THETA：专用入口、双阶段参数与配置

入口：src/models/main.py 的 pipeline/train/evaluate/visualize 等子命令，具体以 create_parser 为准。这些专用选项未全部由 run_pipeline.py 暴露或转发。

### 6.1 模式

| mode | 设计目标 | 当前训练路径 |
|---|---|---|
| zero_shot | 不微调嵌入，先获取通用语义主题结果 | 预先生成文档嵌入，再训练主题网络；使用epochs/learning_rate与KL退火、早停 |
| supervised | 利用标签适配嵌入空间 | 本地模型+标签；stage1为带分类损失的LoRA阶段，stage2为主题阶段；当前实现细节见第11节 |
| unsupervised | 无标签领域适配 | 本地模型；stage1采用项目当前的对比目标，stage2为主题阶段；不能简单等同于经过验证的标准SimCSE流程 |

云嵌入只适用于zero_shot。模型大小与文档向量维度可随模型变化，维度应以实际矩阵为准。

### 6.2 main.py额外可设置的参数

| 参数 | main解析器默认值 | 适用场景/作用 |
|---|---|---|
| --config | None | 读取嵌套PipelineConfig JSON，不是config/default.yaml；部分字段随后被config_from_args覆盖 |
| --dev | False | 排查流程/模型日志；不是缩小数据集的通用开关 |
| --stage1_epochs | 10 | 双阶段LoRA训练轮数；只在supervised/unsupervised使用 |
| --stage2_epochs | 100 | 双阶段主题训练轮数；--epochs不代替它 |
| --lora_r | 8 | 低秩适配容量；领域差异较大可尝试增大，参数/显存随之增加 |
| --lora_alpha | 16 | LoRA缩放强度；应结合r调节，当前普通LoRA缩放与alpha/r相关 |
| --lora_dropout | 0.1 | LoRA分支正则；小数据过拟合时可增加。配置解析使用or回退，显式0会回落为0.1 |
| --train_word_embeddings | True | 从随机初始化学习词嵌入；此开关默认已开启 |
| --no_train_word_embeddings | False | 使用预训练词嵌入并冻结；需存在与词表对齐的词向量。也用于与随机学习词向量做对照 |
| --enable_temporal | False | 请求提取/保存时间戳供后续演化分析 |
| --timestamp_column | None | 指定THETA数据中的时间列；配合enable_temporal |
| --train_exp | 空字符串 | 指定训练实验，复用已有数据/整理结果 |
| --output_base_dir | 空字符串 | 覆盖THETA输出根目录 |
| --num_workers | 4 | 数据加载进程数；CPU/I/O瓶颈时尝试增加，调试/受限环境可用0 |
| --no_pin_memory | False | 关闭锁页内存；无CUDA或主机内存紧张时有用 |
| --no_persistent_workers | False | 不常驻数据加载worker；避免某些进程问题或降低常驻资源 |
| --label_col | label | 指定监督标签列；类别应与文档严格对齐 |
| --timestamp | None | evaluate/visualize子命令选择旧命名模型时间戳；不是训练时间分桶 |
| --no_wordcloud | False | visualize子命令声明；主可视化委托链未发现消费，不应当作已有效的禁用词云开关 |
| --local_rank | -1 | main追加的分布式运行参数，由torchrun/分布式启动环境使用 |
| --world_size | 1 | 分布式GPU进程数；仅设置数值不能替代正确的分布式启动 |

main.py另外接受与统一入口重复的dataset/mode/model_size/num_topics/vocab_size/hidden_dim/epochs/batch_size/learning_rate/KL/patience/GPU/embedding提供方等参数。main的skip开关拼写为--skip_eval、--skip_viz。main的hidden_dim解析器默认1024、epochs解析器默认50，直接调用时与统一入口512/100不同；环境与配置解析还可能改变结果。

LoRA含义依据[PEFT官方文档](https://huggingface.co/docs/peft/package_reference/lora)。以下底层字段以本仓库默认值为准。

### 6.3 ModelConfig全部字段与适用场景

“字段默认”来自数据类本身；实际入口覆盖时以前面入口表为准。

| 配置字段 | 字段默认 | 场景与接线状态 |
|---|---|---|
| num_topics | 20 | 主题粒度；入口会覆盖 |
| hidden_dim | 1024 | 网络容量；统一入口512 |
| doc_embedding_dim | 1024 | 文档矩阵列数；训练前会按实际输入调整 |
| word_embedding_dim | 1024 | 解码词向量维数；与词矩阵对齐，不能当成K |
| encoder_dropout | 0.2 | 主题编码器正则；main会传入ETM，统一--dropout不控制它 |
| encoder_activation | relu | 编码器激活函数；底层支持，但main构造ETM未转发此字段，仍用底层默认 |
| train_word_embeddings | True | 随机学习/冻结预训练词向量；受main开关覆盖 |
| epochs | 100 | 单阶段轮数；受main/统一入口覆盖 |
| batch_size | 64 | 批量/内存 |
| learning_rate | 0.002 | 单阶段Adam学习率 |
| weight_decay | 0.0001 | 权重衰减；减轻过拟合，main单/双阶段优化器使用 |
| stage1_epochs | 10 | 双阶段第一阶段轮数；入口解析覆盖 |
| stage2_epochs | 100 | 双阶段第二阶段轮数；入口解析覆盖 |
| stage1_lr | 0.0001 | LoRA阶段学习率；没有同名CLI，配置/API调整 |
| stage2_lr | 0.002 | 第二阶段学习率；没有同名CLI，配置/API调整 |
| lora_r | 8 | LoRA低秩维度 |
| lora_alpha | 16 | LoRA缩放 |
| lora_dropout | 0.1 | LoRA正则 |
| contrastive_temp | 0.07 | 无监督第一阶段对比温度；越低相似度分布越尖锐，数值更敏感；需先核验当前对比目标 |
| kl_start | 0 | KL初始强度 |
| kl_end | 1 | KL最终强度；解析器用or回退，0不能按原意保留 |
| kl_warmup_epochs | 30 | 数据类默认30，但统一CLI/main通常覆盖为50；需大于0 |
| early_stopping | True | 单阶段早停；main按no_early_stopping重设 |
| patience | 15 | 数据类15，入口通常10 |
| min_delta | 0.001 | 单阶段验证损失最小改善量，降低对微小抖动的敏感性 |
| use_scheduler | True | 单阶段验证平台期降低学习率 |
| scheduler_patience | 5 | 平台期等待轮数；太短可能过早减速 |
| scheduler_factor | 0.5 | 学习率乘数；小于1时降低学习率 |
| train_ratio | 0.8 | 训练集占比，训练和验证应有足够文档 |
| val_ratio | 0.1 | 验证集占比，用于调参与早停 |
| test_ratio | 0.1 | 字段存在，但main实际用剩余n_total-n_train-n_val作为test；不能独立控制 |
| num_workers | 4 | 加载进程数 |
| pin_memory | True | CUDA传输优化 |
| persistent_workers | True | 保持worker进程存活，减少每轮启动开销 |
| prefetch_factor | 2 | 每worker预取批数，增大可能缓解I/O等待也增加内存 |

### 6.4 THETA底层ETM构造参数补充

完整签名见自动提取附录。上表已解释的维数/容量字段不再重复。底层train_word_embeddings默认False，与主流程True不同。

| 参数 | 底层默认 | 含义/场景与状态 |
|---|---|---|
| vocab_size | 必填 | BOW列数/词表长度，必须一致 |
| word_embeddings | None | 与词表对齐的预训练词向量矩阵 |
| kl_weight | 0.5 | 不显式传forward KL权重时的默认值；主单阶段使用退火权重 |
| num_classes | 0 | 分类分支类别数；有标签时由数据计算，不必等于主题数 |
| contrastive_weight | 0.1 | 底层组合目标中的对比损失权重；主双阶段不直接按这个参数构造完整联合目标 |
| contrastive_temp | 0.07 | 底层对比温度，与ModelConfig第一阶段温度是两个传递位置，不宜混为一个CLI |
| dev_mode | False | 额外诊断信息 |

## 7. 各基线额外Python API参数

这些不是统一入口已开放的CLI。位置为src/models/model/baseline/*.py及baseline_trainer.py。参数默认值为直接实例化/调用时的默认值。

共同参数：vocab_size通常为必填BOW列数；num_topics=20为主题/潜变量数；hidden_dim控制隐藏层容量；dropout控制正则。所有矩阵维数应从数据推导。**kwargs在多个类中被忽略，不能据此认为第三方库任意参数都可透传。

### LDA

| API参数 | 默认 | 适用场景 |
|---|---|---|
| alpha | None→1/K | 文档—主题先验；小值偏少数主题，大值更平滑。不是“自动学习alpha”，当前是固定1/K |
| eta | None→1/K | 主题—词先验；小值词分布更尖锐，大值更平滑 |
| learning_method | batch | batch适合常规批量语料；online可用于更大语料的增量式批量更新。BaselineTrainer也暴露此参数，统一CLI没有 |
| random_state | 42 | 初始化复现，需在API设置；统一--random_state不传给LDA |
| n_jobs | 1 | sklearn并行工作数，CPU资源允许时调整 |
| dev_mode | False | 日志；训练器传True |
| doc_embedding_dim/word_embedding_dim/word_embeddings/train_word_embeddings | None/None/None/False | 仅兼容接口，LDA不用它们 |

max_iter/num_topics见通用表。sklearn另有learning_decay等参数，但本项目包装类没有暴露，不能直接当作本项目支持参数。[官方参数说明](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html)。

### HDP

| API参数 | 默认 | 适用场景 |
|---|---|---|
| gamma | 1.0 | 语料层集中参数，影响全局主题分布；不是topic-word prior。训练器暴露，统一CLI未暴露 |
| kappa | 1.0 | 在线更新衰减指数；用于研究学习速度/稳定性 |
| tau | 64.0 | 降低早期更新影响，数据流顺序影响明显时可研究 |
| K | 15，实际min(K,max_topics) | 文档层截断，限制文档局部主题表示容量；不同于统一num_topics |
| T | 150 | 签名保留但被max_topics覆盖；单独传T不生效 |
| random_state | 42 | 可复现训练 |

底层HdpModel还固定chunksize=256、eta=0.01、scale=1、var_converge=0.0001、max_chunks/max_time=None；包装器未把它们开放为独立参数。[gensim官方文档](https://radimrehurek.com/gensim/models/hdpmodel.html)。

### STM

| API参数/输入 | 默认 | 场景 |
|---|---|---|
| random_state | 42 | R seed或Python初始化种子 |
| covariates | None，但本项目要求实际提供 | 每文档一行的元数据设计矩阵；地区/平台类别应合理编码，连续变量关注尺度 |
| covariate_names | None | 协变量名称，解释系数并构造R prevalence公式 |
| vocab | None | 主题词解释，必须与BOW列对应 |
| dataset | None | 数据集标识/报错上下文，不改变模型统计含义 |

backend自动检测R stm，不提供backend选择CLI。R路径公式为协变量相加，未开放content、交互项或完整R stm参数。Python是近似实现；不能预设两套后端结果可直接互换。

### BTM

| API参数 | 默认 | 场景 |
|---|---|---|
| window_size | 15 | 词对提取窗口，扩大增加词对与计算；本实现作用于从BOW重建的词列表，不是原始语序 |
| max_doc_words | 50 | 每文档最多抽样词数，避免长文本词对爆炸；增大保留更多词但更慢 |
| random_state | 42 | 词抽样与Gibbs随机性 |
| fit.verbose | True | 采样进度日志 |
| fit.vocab | None | 词表输入用于解释 |

### ETM

| API参数 | 默认 | 场景 |
|---|---|---|
| hidden_dim | 800 | 类/训练器默认800，统一入口512；网络容量 |
| dropout | 0.5 | 类/训练器默认0.5，统一入口0.2；过拟合时试较强正则 |
| activation | softplus | 编码器激活函数，网络结构对照时调整；需使用实现接受的名称 |
| word_embeddings | None | 预训练Word2Vec/GloVe式矩阵，与词表对齐 |
| train_embeddings | True | 允许更新词向量；固定预训练语义作对照时设False。无预训练矩阵时随机rho仍会requires_grad=True |
| use_pretrained_embeddings | True（训练器） | 是否先查找/训练预训练词向量；控制随机初始化对照 |
| early_stopping_patience | 10（训练器） | 对应统一--patience |
| kl_weight | 1.0 | 类默认KL权重；训练器自算recon+退火*KL，不能靠修改这个默认替代训练器退火 |
| dev_mode | False | 诊断日志 |
| doc_embedding_dim/word_embedding_dim | None/None | 兼容接口，不控制OriginalETM的输入；使用embedding_dim/矩阵实际列数 |

训练Word2Vec辅助函数还支持embedding_dim=300、window=5、min_count=1、workers=4：分别控制词向量维数、上下文范围、低频过滤、CPU线程。ETM训练器自动调用时只显式传入embedding_dim，其余默认。

### CTM

| API参数 | 默认 | 场景 |
|---|---|---|
| doc_embedding_dim | 1024 | SBERT维数；训练器按矩阵实际列数传入 |
| hidden_sizes | (100,100) | 每层宽度，API可设置不等宽网络；统一入口构造成等宽hidden_dim重复num_layers次 |
| activation | softplus | 网络非线性 |
| dropout | 0.2 | 编码器正则；统一--dropout不传入 |
| model_type | prodLDA，可LDA | 解码结构对照；训练器也支持，但统一CLI未暴露 |
| learn_priors | True | 学习先验参数；与固定先验做研究对照 |
| kl_weight | 1.0 | 调整重构与KL约束的相对强度；与THETA退火不是一套CLI |
| early_stopping_patience | 10（训练器） | 训练损失连续不改善时停止 |
| dev_mode | False | 诊断日志；训练器True |
| word_embedding_dim/word_embeddings/train_word_embeddings | None/None/False | 兼容接口，CTM不用预训练词嵌入矩阵 |

inference_type与训练参数见前表。底层内部编码/解码网络的input_size、bert_size、n_components分别对应词表维度、文档嵌入维度、主题数，不是额外用户模型。

### DTM

| API参数 | 默认 | 场景/状态 |
|---|---|---|
| time_slices | 10 | 时间片数量，训练器从预处理文件读取；不是主题数 |
| doc_embedding_dim | 1024 | 文档向量维数；训练器按SBERT或SVD推导 |
| word_embedding_dim | 1024 | 类默认；统一入口embedding_dim=300传入此字段 |
| hidden_dim | 512（类）；256（训练器） | 统一入口传512，控制编码器容量 |
| encoder_dropout | 0.2 | 网络正则，统一CLI未暴露 |
| word_embeddings | None | 自定义解码词向量初始化 |
| train_word_embeddings | False | 签名存在，但未用于控制DTMDecoder的冻结/训练，不能依靠此开关冻结 |
| kl_weight | 0.5 | KL先验正则强度，统一KL参数不传入 |
| evolution_weight | 0.1 | 邻接时间片主题向量平滑强度；高值偏平稳演化，低值允许更大变化 |
| dev_mode | False | 诊断日志 |

训练器固定90%/10%训练验证划分、种子42、梯度裁剪1.0、ReduceLROnPlateau factor=0.5/patience=10；这些不是当前统一CLI参数。它保存验证最优状态，但没有patience早停分支。

### NVDM

类参数完整为vocab_size、num_topics=20、hidden_dim=256、dropout=0.2、**kwargs。统一入口将hidden_dim改为512；dropout固定使用类默认。适合以BOW VAE为对照，调参先看K、hidden_dim、learning_rate与epochs；不要把其未经softmax约束的潜表示直接解释成GSM式主题概率。

训练器额外接收epochs=100、batch_size=64、learning_rate=0.002、hidden_dim=256；统一入口覆盖为前表值。没有已接入的早停、KL退火或层数参数。

### GSM

类参数完整为vocab_size、num_topics=20、hidden_dim=256、dropout=0.2、**kwargs。与NVDM共用训练器；同样只有K/轮数/批量/学习率/隐藏宽度在统一入口可调。适用于需要softmax主题比例的BOW神经基线。

### ProdLDA

类参数完整为vocab_size、num_topics=20、hidden_dim=256、dropout=0.2、variance=0.995、**kwargs。

variance为logistic-normal潜变量先验方差：调整潜空间先验约束，用于先验敏感性研究；统一入口未暴露。其余与NVDM/GSM共用训练器。不要把alpha/beta CLI应用到ProdLDA，它们只传给HDP/BTM。

### BERTopic

| API参数 | 默认 | 场景/与统一入口差异 |
|---|---|---|
| num_topics | None | 包装类转成BERTopic.nr_topics，控制聚类后的合并目标；训练器默认带入统一num_topics=20 |
| vocab_size | None | 仅接口占位，不决定BERTopic的CountVectorizer词表 |
| embedding_model | all-MiniLM-L6-v2 | 自定义SentenceTransformer模型；训练器优先SBERT_MODEL_PATH，预计算嵌入存在时不加载该模型 |
| min_samples | 10（类） | 训练器/API默认None再解析为min_cluster_size |
| calculate_probabilities | True | 需要文档主题概率时开启，语料大时概率矩阵增加内存/计算；统一CLI未暴露 |
| verbose | True | 训练日志 |
| language | english（类/训练器） | 传给BERTopic；主入口zh/chinese映射multilingual，否则english。更改此值不替换已提供的嵌入 |
| fit.texts | 必需 | 原始/清洗文本用于主题表示；预计算embeddings不能完全代替文本 |
| fit.embeddings | None | 使用已有文档嵌入，避免重复编码；行顺序必须与texts一致 |

其余UMAP/HDBSCAN/词数/种子参数见通用表。当前UMAP min_dist=0、metric=cosine；HDBSCAN metric=euclidean、cluster_selection_method=eom、prediction_data=True为固定实现设置。第三方BERTopic的n_gram_range、representation_model等未被本项目包装器开放。

## 8. 数据准备参数：模型场景不可缺少的部分

入口src/models/prepare_data.py。同名参数可能与训练入口默认值不同。

| 参数 | 默认/选项 | 适用场景 |
|---|---|---|
| --dataset | 必填 | 数据集名 |
| --model | 必填：theta/baseline/dtm | 选择准备哪类矩阵；这里不是12个模型ID列表 |
| --model_size | 0.6B，可4B/8B | THETA本地模型/实验目录 |
| --mode | zero_shot，可supervised/unsupervised | THETA嵌入准备模式 |
| --vocab_size | 5000 | 控制新生成BOW词表，改变需重新准备矩阵 |
| --batch_size | 32 | 嵌入生成批量；与训练默认64不同 |
| --max_length | 512 | 嵌入文本处理长度；长文档需考虑窗口与聚合成本，不等于词表大小 |
| --bow-only | False | 仅词袋流程，适用于LDA/HDP/BTM或先检查词表 |
| --skip-sbert | False | 不准备SBERT；LDA等无需上下文嵌入，CTM后续会缺数据或自动补算 |
| --with-time | False | 请求基线时间片输出，用于动态分析 |
| --check-only | False | 只检查已有文件 |
| --gpu | None | 嵌入生成GPU编号 |
| --clean | False | 先运行清洗生成数据 |
| --raw-input | None | clean使用的原始输入路径 |
| --language | None；english/chinese/german/spanish/multi | 已弃用且忽略，语言自动检测；不能据此强制某种文本处理语言 |
| --time_column | year | DTM时间列；年份/时间切片设计应与研究问题一致 |
| --time_slices | None→按不同时间值推导 | 当前只是覆盖声明的片数，没有重新合并时间索引；不要用它直接要求“按月改按年/合成5片” |
| --covariate_columns | None；一个或多个列名 | STM元数据，例province/year/platform；显式选择有解释意义的协变量 |
| --label_col | label | THETA监督标签列 |
| --exp_name | None | 数据实验名标签 |
| --user_id | default_user | 工作区隔离 |
| --output_dir | None | 指定准备矩阵的输出目录 |
| --force | False | 强制覆盖已有矩阵；与run_pipeline未接入的同名选项不同 |

另有6个embedding-provider/cloud-provider/model/api-base/api-key-env/dimensions选项，与第5节相同，含下划线别名。完整29项由附录自动提取；模型维数与词表必须保持匹配。

## 8.1 嵌入与运行配置补充（不是统一训练CLI）

| 配置/环境项 | 默认/来源 | 使用场景 |
|---|---|---|
| SBERT_MODEL_PATH | 环境/项目路径配置；BERTopic缺省回退all-MiniLM-L6-v2 | CTM/BERTopic及可选DTM文档嵌入模型；多语言语料应选择合适且对齐的嵌入，不能只改图表language |
| embedding.model_path | 本地Qwen路径配置 | 本地嵌入/LoRA，必须与模型规模和权重匹配 |
| embedding.embedding_dim | 字段默认1024 | 嵌入维数描述；以实际输出矩阵为准 |
| embedding.batch_size / max_length | 64 / 512（字段） | 嵌入吞吐、文本长度；数据准备CLI批量默认32，入口默认可能覆盖 |
| embedding.normalize / EMBEDDING_NORMALIZE | True | 嵌入向量归一化；不同表示做对比时保持一致，避免把尺度变化当语义变化 |
| EMBEDDING_PROVIDER / EMBEDDING_CLOUD_PROVIDER | cloud / openai（默认工厂） | 未显式传CLI时的服务选择，微调模式要求本地 |
| EMBEDDING_MODEL / EMBEDDING_API_BASE / EMBEDDING_API_KEY_ENV / EMBEDDING_DIMENSIONS | 环境或CLI解析 | 云模型名称、端点、密钥环境变量名、可选维数；只说明变量，不读取/展示密钥 |
| PipelineConfig.seed | 42 | THETA数据划分和相关随机性；没有统一--seed，--random_state只给BERTopic |
| BaselineTrainer.device | auto | Python API选择设备；统一入口使用GPU环境配置 |
| BaselineTrainer.data_dir/result_dir/output_dir/data_exp_dir/workspace_dir/user_id | None或default_user | 输入与结果隔离，不是模型质量超参数 |

## 9. 默认值差异：为什么不能直接照YAML填写

| 参数/模型 | run_pipeline.py直接调用 | config/default.yaml | API默认 |
|---|---|---|---|
| ETM hidden_dim/dropout | 512 / 0.2 | 800 / 0.5 | 800 / 0.5 |
| CTM hidden_sizes | (512,512) | hidden_dim=100,num_layers=2 | (100,100) |
| NVDM/GSM/ProdLDA hidden_dim | 512 | 256 | 256 |
| THETA hidden_dim | 512 | 512 | ModelConfig=1024，底层ETM=512 |
| THETA patience | 10 | 10 | ModelConfig=15 |
| THETA kl_warmup | 50 | 50 | ModelConfig=30 |
| BERTopic num_topics | 20，正数传nr_topics | 未为BERTopic明确配置K，shell回退20 | 类默认None |

run_pipeline.py导入ConfigLoader/YAMLConfig，但当前执行路径没有实际用它们读取config/default.yaml。scripts/train_baseline.sh会读取YAML，且只读逗号列表中的第一个模型，再把相同参数传给这一批模型；一次批跑多个模型不等于各自使用各自YAML默认。

为了可复核比较，记录入口、完整CLI、BOW版本、嵌入模型与维数、实际K、随机种子；不能只记录“用了默认配置”。

## 10. 按研究问题选模型和优先调参

以下为建议顺序，不是保证最优的推荐配方。

| 问题/资源条件 | 可先比较 | 优先检查/调节 |
|---|---|---|
| 长文本、无嵌入、需要传统参照 | LDA，必要时HDP | 清洗与词表、K、max_iter；HDP看max_topics及低权重主题 |
| 极短评论/标题，词共现稀疏 | BTM、BERTopic、CTM | BTM的K/n_iter/alpha/beta；BERTopic的簇规模/噪声；CTM的嵌入质量 |
| 想利用语义但先不微调大模型 | THETA zero_shot、CTM、ETM | 嵌入是否适合语言/领域，再比较K、学习率、容量 |
| 比较地区/平台/机构与主题的关联 | STM | 协变量设计/编码、K、max_iter；不能把类别编码数字的大小当作有序含义 |
| 多年政策/新闻的真实词义主题变化 | DTM + 通用主题占比统计 | 时间片文档量、时间索引对齐、K；API可调evolution_weight |
| 完全不知道K、允许噪声文档 | BERTopic自动模式；HDP探索 | BERTopic --num_topics 0，min_cluster_size/min_samples/n_neighbors；HDP另外筛选有效主题 |
| 神经模型方法对照 | NVDM、GSM、ProdLDA | 同一BOW/数据划分、K、容量、学习率与多次重复 |
| 标签引导或领域LoRA适配 | THETA supervised/unsupervised的设计目标 | 先核验双阶段训练实现，再讨论stage1/stage2/LoRA；当前不宜直接当成熟默认流程推荐 |

## 11. 当前代码中影响选型或参数生效的事实

1. num_layers只传给CTM；dropout只传给ETM。帮助文本中的“for neural models”范围大于实际实现。
2. patience只传THETA/ETM/CTM；no_early_stopping只转发THETA。ETM/CTM看训练损失，不是验证损失。
3. kl_start/end/warmup只转发THETA；ETM内部固定warmup=min(20,epochs//3)。DTM类用kl_weight=0.5和evolution_weight=0.1。
4. THETA supervised/unsupervised会直接返回train_two_stage，不使用单阶段epochs/learning_rate/早停循环；应调stage1_epochs/stage2_epochs/stage1_lr/stage2_lr。
5. 更关键：双阶段stage2实际反传loss=output['kl_loss']，该值在ETM.forward内部已经乘以退火权重，所以KL退火确实生效；但没有使用重构/对比组成的总损失，不能把它描述成常规完整主题重构训练。见[main.py](/Users/erwin/Downloads/codespace/THETA/src/models/main.py:988)。
6. 双阶段无监督stage1用同一批嵌入的自相似矩阵及其对角标签，并非标准独立双视图SimCSE实现；有监督分类头又是在优化器创建后动态挂载，需要验证是否实际被优化。这些是代码核对发现，未通过本次训练实验验证效果。
7. BERTopic统一入口默认20传作nr_topics；--num_topics 0才取消合并目标。HDP则输出截断主题矩阵，未清除弱主题。先前把二者简单说成“自动主题数”不够准确。
8. LDA alpha/eta=None实际固定1/K，注释“auto-learn”不准确。统一--alpha不传LDA。HDP YAML把gamma写成topic-word prior，也不准确，gamma是语料层集中参数。
9. main配置读取为JSON；读取之后config_from_args会覆盖多个字段。只改config/default.yaml不会改变直接调用run_pipeline.py的默认值。
10. run_pipeline --prepare后立即return；它不会在准备后自动训练，且vocab_size取DATASET_CONFIGS/5000，忽略传入args.vocab_size。user_id/workspace_dir/force/label_col/时间/协变量也未完整转发，复杂数据准备应直接使用prepare_data.py。
11. 已有BOW载入时，vocab_size常按矩阵列数重设；训练参数不能替代重新建词表。ETM已有词向量维数也以矩阵为准。
12. run_pipeline --force未消费；THETA --skip-train跳过整个pipeline；语言别名chinese/english被外层接受但内层main只接受zh/en；基线图表看--lang。
13. DTM time_slices覆盖不会重新分桶，不合理值可能造成时间索引越界。DTM train_word_embeddings签名存在但没有接线。
14. 旧registry.py与models.yaml是描述性信息，存在旧路径与默认值差异；以ALL_MODELS和实际调用链为准。
15. 对上一轮可视化说明补充纠正：除词语义演化图、词分布变化表外，“主题分布KL散度时序变化”也用静态beta加随机噪声构造年份差异，不能解释为实际跨期训练结果。见[visualization_generator.py](/Users/erwin/Downloads/codespace/THETA/src/models/visualization/visualization_generator.py:1034)。

## 12. 本地证据入口

- [12模型与50个CLI参数](/Users/erwin/Downloads/codespace/THETA/src/models/run_pipeline.py:78)
- [实际模型训练分发](/Users/erwin/Downloads/codespace/THETA/src/models/run_pipeline.py:648)
- [基线训练器](/Users/erwin/Downloads/codespace/THETA/src/models/model/baseline_trainer.py)
- [THETA配置与专用参数](/Users/erwin/Downloads/codespace/THETA/src/models/config.py:662)
- [THETA主训练程序](/Users/erwin/Downloads/codespace/THETA/src/models/main.py)
- [数据准备参数](/Users/erwin/Downloads/codespace/THETA/src/models/prepare_data.py:48)
- [YAML默认](/Users/erwin/Downloads/codespace/THETA/config/default.yaml)
- [Shell读取YAML的逻辑](/Users/erwin/Downloads/codespace/THETA/scripts/train_baseline.sh:213)

## 13. 从源码自动提取的CLI与API签名

本附录直接读取AST，不导入模型或运行训练。默认值是源码字面值；运行时覆盖见正文。内部层组件不计作额外模型。

### src/models/run_pipeline.py：50个参数声明

| 行号 | 名称/别名 | 默认值 | 类型/行为 | 可选值 |
|---|---|---|---|---|
| 185 | `--dataset` | 必填 | str | — |
| 187 | `--models` | 必填 | str | — |
| 189 | `--mode` | 'zero_shot' | str | ['zero_shot', 'supervised', 'unsupervised'] |
| 192 | `--num_topics` | 20 | int | — |
| 193 | `--vocab_size` | 5000 | int | — |
| 194 | `--epochs` | 100 | int | — |
| 195 | `--batch_size` | 64 | int | — |
| 196 | `--hidden_dim` | 512 | int | — |
| 197 | `--learning_rate` | 0.002 | float | — |
| 198 | `--kl_start` | 0.0 | float | — |
| 199 | `--kl_end` | 1.0 | float | — |
| 200 | `--kl_warmup` | 50 | int | — |
| 201 | `--patience` | 10 | int | — |
| 202 | `--no_early_stopping` | False | 'store_true' | — |
| 203 | `--skip-train` | False | 'store_true' | — |
| 204 | `--skip-eval` | False | 'store_true' | — |
| 205 | `--skip-viz` | False | 'store_true' | — |
| 206 | `--gpu` | None | int | — |
| 213 | `--language` | 'zh' | str | ['en', 'zh', 'chinese', 'english'] |
| 216 | `--model_size` | '0.6B' | str | MODEL_SIZES |
| 219 | `--embedding-provider` / `--embedding_provider` | None | str | ['cloud', 'local', 'qwen', 'openai', 'dashscope', 'siliconflow', 'zhipu', 'volcengine', 'openai_compatible'] |
| 224 | `--embedding-cloud-provider` / `--embedding_cloud_provider` | None | str | ['openai', 'dashscope', 'siliconflow', 'zhipu', 'volcengine', 'openai_compatible'] |
| 229 | `--embedding-model` / `--embedding_model` | None | str | — |
| 232 | `--embedding-api-base` / `--embedding_api_base` | None | str | — |
| 235 | `--embedding-api-key-env` / `--embedding_api_key_env` | None | str | — |
| 238 | `--embedding-dimensions` / `--embedding_dimensions` | None | int | — |
| 241 | `--check-only` | False | 'store_true' | — |
| 243 | `--prepare` | False | 'store_true' | — |
| 247 | `--max_iter` | 100 | int | — |
| 248 | `--max_topics` | 150 | int | — |
| 249 | `--n_iter` | 100 | int | — |
| 250 | `--alpha` | 1.0 | float | — |
| 251 | `--beta` | 0.01 | float | — |
| 252 | `--inference_type` | 'zeroshot' | str | ['zeroshot', 'combined'] |
| 254 | `--dropout` | 0.2 | float | — |
| 255 | `--num_layers` | 2 | int | — |
| 256 | `--embedding_dim` | 300 | int | — |
| 258 | `--n_neighbors` | 15 | int | — |
| 259 | `--n_components` | 5 | int | — |
| 260 | `--min_cluster_size` | 10 | int | — |
| 261 | `--min_samples` | None | int | — |
| 262 | `--top_n_words` | 10 | int | — |
| 263 | `--random_state` | 42 | int | — |
| 266 | `--data_exp` | None | str | — |
| 268 | `--exp_name` | None | str | — |
| 272 | `--user_id` | 'default_user' | str | — |
| 274 | `--workspace_dir` | None | str | — |
| 276 | `--force` | False | 'store_true' | — |
| 280 | `--task_name` | None | str | — |
| 282 | `--lang` | 'en' | str | ['en', 'cn', 'both'] |

### src/models/prepare_data.py：29个参数声明

| 行号 | 名称/别名 | 默认值 | 类型/行为 | 可选值 |
|---|---|---|---|---|
| 53 | `--dataset` | 必填 | str | — |
| 54 | `--model` | 必填 | str | ['theta', 'baseline', 'dtm'] |
| 57 | `--model_size` | '0.6B' | str | ['0.6B', '4B', '8B'] |
| 60 | `--embedding-provider` / `--embedding_provider` | None | str | ['cloud', 'local', 'qwen', 'openai', 'dashscope', 'siliconflow', 'zhipu', 'volcengine', 'openai_compatible'] |
| 65 | `--embedding-cloud-provider` / `--embedding_cloud_provider` | None | str | ['openai', 'dashscope', 'siliconflow', 'zhipu', 'volcengine', 'openai_compatible'] |
| 70 | `--embedding-model` / `--embedding_model` | None | str | — |
| 73 | `--embedding-api-base` / `--embedding_api_base` | None | str | — |
| 76 | `--embedding-api-key-env` / `--embedding_api_key_env` | None | str | — |
| 79 | `--embedding-dimensions` / `--embedding_dimensions` | None | int | — |
| 82 | `--mode` | 'zero_shot' | str | ['zero_shot', 'supervised', 'unsupervised'] |
| 85 | `--vocab_size` | 5000 | int | — |
| 86 | `--batch_size` | 32 | int | — |
| 87 | `--max_length` | 512 | int | — |
| 88 | `--bow-only` | False | 'store_true' | — |
| 89 | `--skip-sbert` | False | 'store_true' | — |
| 90 | `--with-time` | False | 'store_true' | — |
| 91 | `--check-only` | False | 'store_true' | — |
| 92 | `--gpu` | None | int | — |
| 98 | `--clean` | False | 'store_true' | — |
| 100 | `--raw-input` | None | str | — |
| 104 | `--language` | None | str | ['english', 'chinese', 'german', 'spanish', 'multi', None] |
| 108 | `--time_column` | 'year' | str | — |
| 110 | `--time_slices` | None | int | — |
| 113 | `--covariate_columns` | None | str | — |
| 116 | `--label_col` | 'label' | str | — |
| 118 | `--exp_name` | None | str | — |
| 122 | `--user_id` | 'default_user' | str | — |
| 124 | `--output_dir` | None | str | — |
| 126 | `--force` | False | 'store_true' | — |

### src/models/config.py：48个参数声明

| 行号 | 名称/别名 | 默认值 | 类型/行为 | 可选值 |
|---|---|---|---|---|
| 1022 | `--timestamp` | None | str | — |
| 1027 | `--timestamp` | None | str | — |
| 1028 | `--no_wordcloud` | False | 'store_true' | — |
| 1037 | `--input` | 必填 | str | — |
| 1038 | `--output` | 必填 | str | — |
| 1039 | `--language` | 'english' | str | ['english', 'chinese', 'german'] |
| 1046 | `--dataset` | 'socialTwitter' | str | — |
| 1048 | `--mode` | 'zero_shot' | str | ['zero_shot', 'supervised', 'unsupervised'] |
| 1051 | `--config` | None | str | — |
| 1053 | `--gpu` | None | int | — |
| 1055 | `--dev` | False | 'store_true' | — |
| 1062 | `--num_topics` | 20 | int | — |
| 1064 | `--vocab_size` | 5000 | int | — |
| 1066 | `--hidden_dim` | 1024 | int | — |
| 1070 | `--epochs` | 50 | int | — |
| 1072 | `--batch_size` | 64 | int | — |
| 1074 | `--learning_rate` | 0.002 | float | — |
| 1078 | `--stage1_epochs` | 10 | int | — |
| 1080 | `--stage2_epochs` | 100 | int | — |
| 1082 | `--lora_r` | 8 | int | — |
| 1084 | `--lora_alpha` | 16 | int | — |
| 1086 | `--lora_dropout` | 0.1 | float | — |
| 1090 | `--kl_start` | 0.0 | float | — |
| 1092 | `--kl_end` | 1.0 | float | — |
| 1094 | `--kl_warmup` | 50 | int | — |
| 1098 | `--no_early_stopping` | False | 'store_true' | — |
| 1100 | `--patience` | 10 | int | — |
| 1104 | `--train_word_embeddings` | True | 'store_true' | — |
| 1106 | `--no_train_word_embeddings` | False | 'store_true' | — |
| 1110 | `--enable_temporal` | False | 'store_true' | — |
| 1112 | `--timestamp_column` | None | str | — |
| 1116 | `--model_size` | '0.6B' | str | ['0.6B', '4B', '8B'] |
| 1119 | `--embedding_provider` / `--embedding-provider` | None | str | ['cloud', 'local', 'qwen', 'openai', 'dashscope', 'siliconflow', 'zhipu', 'volcengine', 'openai_compatible'] |
| 1123 | `--embedding_cloud_provider` / `--embedding-cloud-provider` | None | str | ['openai', 'dashscope', 'siliconflow', 'zhipu', 'volcengine', 'openai_compatible'] |
| 1127 | `--embedding_model` / `--embedding-model` | None | str | — |
| 1130 | `--embedding_api_base` / `--embedding-api-base` | None | str | — |
| 1133 | `--embedding_api_key_env` / `--embedding-api-key-env` | None | str | — |
| 1136 | `--embedding_dimensions` / `--embedding-dimensions` | None | int | — |
| 1141 | `--skip_viz` | False | 'store_true' | — |
| 1143 | `--skip_eval` | False | 'store_true' | — |
| 1147 | `--data_exp` | '' | str | — |
| 1149 | `--train_exp` | '' | str | — |
| 1151 | `--output_base_dir` | '' | str | — |
| 1155 | `--language` | 'en' | str | ['en', 'zh'] |
| 1159 | `--num_workers` | 4 | int | — |
| 1161 | `--no_pin_memory` | False | 'store_true' | — |
| 1163 | `--no_persistent_workers` | False | 'store_true' | — |
| 1167 | `--label_col` | 'label' | str | — |

### src/models/main.py：2个参数声明

| 行号 | 名称/别名 | 默认值 | 类型/行为 | 可选值 |
|---|---|---|---|---|
| 2090 | `--local_rank` | -1 | int | — |
| 2092 | `--world_size` | 1 | int | — |

### src/models/model/theta/etm.py

源码第40行：
```python
ETM.__init__(self, vocab_size: int, num_topics: int, doc_embedding_dim: int=1024, word_embedding_dim: int=1024, hidden_dim: int=512, encoder_dropout: float=0.2, encoder_activation: str='relu', word_embeddings: Optional[torch.Tensor]=None, train_word_embeddings: bool=False, kl_weight: float=0.5, num_classes: int=0, contrastive_weight: float=0.1, contrastive_temp: float=0.07, dev_mode: bool=False)
```

### src/models/model/baseline/bertopic.py

源码第43行：
```python
BERTopicModel.__init__(self, vocab_size: int=None, num_topics: int=None, embedding_model: str='all-MiniLM-L6-v2', n_neighbors: int=15, n_components: int=5, min_cluster_size: int=10, min_samples: int=10, top_n_words: int=10, language: str='english', calculate_probabilities: bool=True, verbose: bool=True, random_state: int=42, **kwargs)
```

源码第154行：
```python
BERTopicModel.fit(self, texts: List[str], embeddings: Optional[np.ndarray]=None, **kwargs)
```

### src/models/model/baseline/btm.py

源码第41行：
```python
BTM.__init__(self, vocab_size: int, num_topics: int=20, alpha: float=1.0, beta: float=0.01, n_iter: int=100, window_size: int=15, max_doc_words: int=50, random_state: int=42, **kwargs)
```

源码第140行：
```python
BTM.fit(self, bow_matrix: np.ndarray, vocab: Optional[List[str]]=None, verbose: bool=True, **kwargs)
```

### src/models/model/baseline/ctm.py

源码第379行：
```python
CTM.__init__(self, vocab_size: int, num_topics: int=20, doc_embedding_dim: int=1024, hidden_sizes: Tuple[int, ...]=(100, 100), activation: str='softplus', dropout: float=0.2, model_type: str='prodLDA', inference_type: str='zeroshot', learn_priors: bool=True, kl_weight: float=1.0, dev_mode: bool=False, word_embedding_dim: int=None, word_embeddings: torch.Tensor=None, train_word_embeddings: bool=False, **kwargs)
```

### src/models/model/baseline/dtm.py

源码第216行：
```python
DTM.__init__(self, vocab_size: int, num_topics: int=20, time_slices: int=10, doc_embedding_dim: int=1024, word_embedding_dim: int=1024, hidden_dim: int=512, encoder_dropout: float=0.2, word_embeddings: Optional[torch.Tensor]=None, train_word_embeddings: bool=False, kl_weight: float=0.5, evolution_weight: float=0.1, dev_mode: bool=False, **kwargs)
```

### src/models/model/baseline/etm.py

源码第35行：
```python
OriginalETM.__init__(self, vocab_size: int, num_topics: int=20, embedding_dim: int=300, hidden_dim: int=800, dropout: float=0.5, activation: str='softplus', word_embeddings: Optional[np.ndarray]=None, train_embeddings: bool=True, kl_weight: float=1.0, dev_mode: bool=False, doc_embedding_dim: int=None, word_embedding_dim: int=None, **kwargs)
```

源码第287行：
```python
train_word2vec_embeddings(texts: List[str], vocab: List[str], embedding_dim: int=300, window: int=5, min_count: int=1, workers: int=4)
```

### src/models/model/baseline/gsm.py

源码第86行：
```python
GSM.__init__(self, vocab_size: int, num_topics: int=20, hidden_dim: int=256, dropout: float=0.2, **kwargs)
```

### src/models/model/baseline/hdp.py

源码第43行：
```python
HDP.__init__(self, vocab_size: int, max_topics: int=150, alpha: float=1.0, gamma: float=1.0, kappa: float=1.0, tau: float=64.0, K: int=15, T: int=150, random_state: int=42, **kwargs)
```

源码第97行：
```python
HDP.fit(self, bow_matrix: np.ndarray, vocab: Optional[List[str]]=None, **kwargs)
```

### src/models/model/baseline/lda.py

源码第37行：
```python
SklearnLDA.__init__(self, vocab_size: int, num_topics: int=20, alpha: float=None, eta: float=None, max_iter: int=100, learning_method: str='batch', random_state: int=42, n_jobs: int=1, dev_mode: bool=False, doc_embedding_dim: int=None, word_embedding_dim: int=None, word_embeddings: Any=None, train_word_embeddings: bool=False, **kwargs)
```

源码第95行：
```python
SklearnLDA.fit(self, bow_matrix: np.ndarray, **kwargs)
```

### src/models/model/baseline/nvdm.py

源码第88行：
```python
NVDM.__init__(self, vocab_size: int, num_topics: int=20, hidden_dim: int=256, dropout: float=0.2, **kwargs)
```

### src/models/model/baseline/prodlda.py

源码第33行：
```python
ProdLDA.__init__(self, vocab_size: int, num_topics: int=20, hidden_dim: int=256, dropout: float=0.2, variance: float=0.995, **kwargs)
```

### src/models/model/baseline/stm.py

源码第93行：
```python
STM.__init__(self, vocab_size: int, num_topics: int=20, max_iter: int=100, random_state: int=42, **kwargs)
```

源码第155行：
```python
STM.fit(self, bow_matrix: np.ndarray, covariates: np.ndarray=None, covariate_names: Optional[List[str]]=None, vocab: Optional[List[str]]=None, dataset: str=None, **kwargs)
```

### src/models/model/baseline_trainer.py

源码第69行：
```python
BaselineTrainer.__init__(self, dataset: str, num_topics: int=20, vocab_size: int=5000, user_id: str='default_user', workspace_dir: str=None, result_dir: str=None, data_dir: str=None, data_exp_dir: str=None, output_dir: str=None, device: str='auto')
```

源码第426行：
```python
BaselineTrainer.train_lda(self, max_iter: int=100, learning_method: str='batch')
```

源码第513行：
```python
BaselineTrainer.train_ctm(self, inference_type: str='zeroshot', model_type: str='prodLDA', hidden_sizes: tuple=(100, 100), epochs: int=100, batch_size: int=64, learning_rate: float=0.002, early_stopping_patience: int=10)
```

源码第704行：
```python
BaselineTrainer.train_etm(self, embedding_dim: int=300, hidden_dim: int=800, dropout: float=0.5, train_embeddings: bool=True, use_pretrained_embeddings: bool=True, epochs: int=100, batch_size: int=64, learning_rate: float=0.002, early_stopping_patience: int=10)
```

源码第934行：
```python
BaselineTrainer.train_dtm(self, epochs: int=100, batch_size: int=64, learning_rate: float=0.002, hidden_dim: int=256, embedding_dim: int=300)
```

源码第1254行：
```python
BaselineTrainer.train_hdp(self, max_topics: int=150, alpha: float=1.0, gamma: float=1.0)
```

源码第1303行：
```python
BaselineTrainer.train_stm(self, max_iter: int=100, covariates: Optional[np.ndarray]=None, covariate_names: Optional[List[str]]=None)
```

源码第1400行：
```python
BaselineTrainer.train_btm(self, n_iter: int=100, alpha: float=1.0, beta: float=0.01)
```

源码第1447行：
```python
BaselineTrainer._train_neural_topic_model(self, model_class, model_name: str, epochs: int=100, batch_size: int=64, learning_rate: float=0.002, hidden_dim: int=256)
```

源码第1547行：
```python
BaselineTrainer.train_nvdm(self, epochs: int=100, batch_size: int=64, **kwargs)
```

源码第1551行：
```python
BaselineTrainer.train_gsm(self, epochs: int=100, batch_size: int=64, **kwargs)
```

源码第1555行：
```python
BaselineTrainer.train_prodlda(self, epochs: int=100, batch_size: int=64, **kwargs)
```

源码第1559行：
```python
BaselineTrainer.train_bertopic(self, n_neighbors: int=15, n_components: int=5, min_cluster_size: int=10, min_samples: int=None, top_n_words: int=10, language: str='english', random_state: int=42)
```

源码第1728行：
```python
BaselineTrainer.train_all(self, models: List[str]=None, **kwargs)
```

