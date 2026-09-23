import { apiFetch, API_BASE } from '@/lib/api/config'
import { getResultCatalog, type WebResultCatalog } from '@/components/theta-workbench/api/client'

export type ResultDestination = 'results' | 'interpretation'
export type ResultSource =
  | { kind: 'run'; runId: string; projectId?: string; projectName: string; datasetName?: string; catalog?: WebResultCatalog }
  | { kind: 'dataset'; project: { id?: string; dbProjectId?: number; name: string; datasetName?: string; mode?: string; models?: string[]; numTopics?: number } }

export const resultScope = (source: ResultSource): string => source.kind === 'run' ? `run:${source.runId}` : `dataset:${source.project.id ?? source.project.datasetName ?? source.project.name}`
export const resultProjectName = (source: ResultSource): string => source.kind === 'run' ? source.projectName : source.project.name
export const resultDatasetName = (source: ResultSource): string | undefined => source.kind === 'run' ? source.datasetName : source.project.datasetName ?? source.project.name

/** Only data transport differs. Both workbenches deliver the same result catalog. */
export function loadResultCatalog(source: ResultSource): Promise<WebResultCatalog> {
  return source.kind === 'run' ? getResultCatalog(source.runId) : apiFetch(API_BASE, `/api/results/${encodeURIComponent(resultDatasetName(source)!)}/catalog`)
}
