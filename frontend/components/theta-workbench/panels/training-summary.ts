import type { WebResultCatalog, WebRunStatus } from '../api/client.ts'

export function trainingSummary(status?: WebRunStatus, catalog?: WebResultCatalog) {
  const jobs = catalog?.results ?? []
  const active = jobs.filter(job => job.status === 'running' || job.status === 'queued')
  const completed = jobs.filter(job => job.status === 'completed').length
  const failed = jobs.filter(job => job.status === 'failed').length
  const cancelled = jobs.filter(job => job.status === 'cancelled').length
  const current = active.at(-1) ?? jobs.at(-1)
  const state = active.length ? (active.some(job => job.status === 'running') ? 'running' : 'queued')
    : failed ? 'failed' : cancelled ? 'cancelled' : current?.status ?? status?.trainingStatus
  const clamp = (value: number) => Number.isFinite(value) ? Math.max(0, Math.min(100, value)) : 0
  const percent = jobs.length
    ? jobs.reduce((sum, job) => sum + (job.status === 'completed' ? 100 : clamp(job.percent ?? 0)), 0) / jobs.length
    : state === 'completed' ? 100 : clamp(status?.trainingPercent ?? 0)
  return {
    state, current, completed, failed, cancelled, total: jobs.length, active: active.length,
    percent: Math.round(percent),
    ready: jobs.some(job => job.status === 'completed' && job.artifacts != null),
    figures: jobs.reduce((sum, job) => sum + (job.artifacts?.figureCount ?? 0), 0),
    tables: jobs.reduce((sum, job) => sum + (job.artifacts?.tableCount ?? 0), 0),
  }
}
