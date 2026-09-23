/** Identify a chart independently of its display title. */
export function chartReferenceKey(chart: { referenceId?: string; dataset: string; model: string; chartPath: string; chartName: string }): string {
  return chart.referenceId ?? JSON.stringify([chart.dataset, chart.model, chart.chartPath || chart.chartName])
}

export function mergeChartReferences<T extends Parameters<typeof chartReferenceKey>[0]>(current: T[], incoming: T[]): T[] {
  const next = [...current]
  for (const chart of incoming) {
    const index = next.findIndex(item => chartReferenceKey(item) === chartReferenceKey(chart))
    if (index >= 0) next[index] = chart
    else next.push(chart)
  }
  return next.slice(-4)
}
