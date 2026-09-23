const CHART_DATA_RULES: Array<{ chart: RegExp; sources: string[] }> = [
  { chart: /topic[_ ]?table|主题识别|主题表/iu, sources: ["topic_table.csv", "主题表.csv"] },
  { chart: /topic distribution similarity evolution|主题分布相似度演化/iu, sources: ["topic_weights_by_year.csv"] },
  { chart: /topic clustering heatmap|主题聚类热力图/iu, sources: ["topic_correlations.csv"] },
  { chart: /document clustering|document clusters|doc_topic_umap|文档主题聚类|文档聚类|文档主题umap/iu, sources: ["document_projection.csv"] },
  { chart: /network|相关性网络/iu, sources: ["topic_network_edges.csv", "topic_network_circular_edges.csv", "topic_correlations.csv"] },
  { chart: /intertopic distance|主题间距离/iu, sources: ["intertopic_distance.csv"] },
  { chart: /topic similarity|主题相似度/iu, sources: ["topic_beta_similarity.csv"] },
  { chart: /proportion|占比/iu, sources: ["topic_proportions.csv"] },
  { chart: /document volume|文档数量/iu, sources: ["document_counts_by_year.csv"] },
  { chart: /representative topic|strength by year|代表性主题|年度主题|各年度主题/iu, sources: ["topic_weights_by_year.csv"] },
  { chart: /kl|散度/iu, sources: ["temporal_topic_kl.csv"] },
  { chart: /high.?frequency|vocab|高频词/iu, sources: ["word_counts_by_year.csv"] },
  { chart: /dimension|group|source|维度|来源/iu, sources: ["topic_weights_by_group.csv", "group_topic_over_time.csv", "stm_group_topic_means.csv"] },
  { chart: /training|loss|perplexity|训练|损失|困惑度/iu, sources: ["training_curves.csv"] },
  { chart: /coherence|一致性/iu, sources: ["topic_coherence.csv"] },
  { chart: /exclusivity|排他性/iu, sources: ["topic_exclusivity.csv"] },
  { chart: /core metrics|evaluation metrics|核心指标/iu, sources: ["evaluation_metrics.csv"] },
  { chart: /topic number|num evaluation|主题数评估/iu, sources: ["k_evaluation.csv"] },
  { chart: /sankey|桑基/iu, sources: ["year_topic_sankey.csv", "source_topic_sankey.csv"] },
  { chart: /topic_wordcloud_grid/iu, sources: ["topic_word_weights.csv"] },
  { chart: /top salient terms|最显著词汇/iu, sources: ["topic_word_weights.csv"] },
  { chart: /word distribution change|词分布变化/iu, sources: ["word_distribution_by_time.csv", "dtm_word_evolution.csv"] },
  { chart: /word importance|word cloud|word distribution|词重要性|词云|词语分布/iu, sources: ["word_importance.csv"] },
  { chart: /topic evolution|主题演化/iu, sources: ["topic_evolution.csv"] },
]

function fileName(path: string): string {
  return path.split("/").pop() || path
}

function directoryName(path: string): string {
  return path.includes("/") ? path.slice(0, path.lastIndexOf("/")) : ""
}

export function resolveChartDataFiles<T extends { name: string; path: string }>(chart: { name: string; path: string }, allFiles: T[]): T[] {
  const dataFiles = allFiles.filter((file) => /\.(?:csv|json)$/iu.test(file.name))
  const chartDirectory = directoryName(chart.path)
  const directStem = fileName(chart.path).replace(/\.[^.]+$/u, "").toLocaleLowerCase()
  const matchedRule = CHART_DATA_RULES.find((rule) => rule.chart.test(chart.name) || rule.chart.test(chart.path))
  const ruleNames = matchedRule?.sources ?? []
  const requested = new Set([`${directStem}.csv`, `${directStem}.json`, ...ruleNames].map((name) => name.toLocaleLowerCase()))
  const matches = dataFiles.filter((file) => requested.has(fileName(file.path).toLocaleLowerCase()))
  const sameDirectoryMatches = matches.filter((file) => directoryName(file.path) === chartDirectory)
  if (sameDirectoryMatches.length > 0) return sameDirectoryMatches.slice(0, 4)
  if (matches.length > 0) return matches.slice(0, 4)
  return []
}
