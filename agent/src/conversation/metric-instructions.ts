/** Meaning of the metrics produced by this repository's unified evaluator. */
export const METRIC_INSTRUCTIONS = `
【当前实现的指标口径与证据边界】
TD 是各主题前列词的去重比例。iRBO 是 1 减去主题词排序之间的 Rank-Biased Overlap，反映主题词排序多样性，不是扰动稳定性、重跑稳定性或置信度。
NPMI 是基于词共现的归一化点互信息；UMass 是基于文档共现的非对称对数条件概率一致性，两者量纲、范围不同，不能比较绝对值或说只是正负号相反。
当前 unified_evaluator 的 C_V 使用 topic_metrics.py 中文档共现 NPMI 的简化近似，不应声称等同于标准滑动窗口 C_V。
Exclusivity 表示主题词排他性；PPL 是困惑度，仅在相同数据、词表和计算口径下可比较，不能跨模型任意排好坏。
没有已提供的基线时，不得把单个指标的数值判定为高、中、低、优秀、稳定或收敛。文件名和图像像素不是数值证据。缺失指标不补造；允许引用实际提供的数据文件名，但不要泄露本机绝对路径。
`;
