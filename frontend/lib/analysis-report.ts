/** Conversation and manual results have different same-origin API proxies. */
export function analysisReportEndpoint(kind: 'run' | 'dataset', scope: string, jobId: string): string {
  return kind === 'run'
    ? `/api/v3/runs/${encodeURIComponent(scope)}/analysis-report/${encodeURIComponent(jobId)}`
    : `/api/backend/api/results/${encodeURIComponent(scope)}/analysis-report?job_id=${encodeURIComponent(jobId)}`
}
