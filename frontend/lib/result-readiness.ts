type ResultReadiness = {
  status: string
  reportStatus?: string
  artifacts?: { fileCount: number }
}

/** Training completion and report publication are separate events. GET only; never starts work. */
export function shouldRefreshResultCatalog(results: ResultReadiness[]): boolean {
  return results.some(result => result.status === 'queued' || result.status === 'running' || (
    result.status === 'completed' && !result.artifacts && result.reportStatus !== 'incomplete'
  ))
}

export function resultReadinessNotice(result: ResultReadiness, conversational: boolean): string | undefined {
  if (result.reportStatus === 'incomplete') return '训练已完成，但图表整理不完整。已有文件仍可查看；请返回工作台查看整理日志。'
  if (result.status !== 'completed' || result.artifacts) return undefined
  return conversational
    ? '训练已完成，图表清单尚未就绪。若还未确认“整理结果”，请返回对话确认；若已确认，页面会自动同步生成后的图表，无需重新训练。'
    : '训练已完成，正在等待结果文件同步。页面会自动更新，无需重新训练。'
}
