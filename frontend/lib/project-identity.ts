interface ProjectIdentity {
  id: string
  dbProjectId?: number
  datasetName?: string
  name: string
  models?: string[]
}

/** Preserve open tabs by database ID; only unambiguous temporary entries may migrate. */
export function reconcileProjectIdentity<T extends ProjectIdentity>(next: T[], previous: ProjectIdentity[]): T[] {
  const byId = new Map(previous.filter(p => p.dbProjectId != null).map(p => [p.dbProjectId, p]))
  const temporary = new Map<string, ProjectIdentity[]>()
  const datasetCounts = new Map<string, number>()
  for (const project of previous) {
    if (project.dbProjectId != null) continue
    const key = project.datasetName || project.name
    temporary.set(key, [...(temporary.get(key) ?? []), project])
  }
  for (const project of next) {
    const key = project.datasetName || project.name
    datasetCounts.set(key, (datasetCounts.get(key) ?? 0) + 1)
  }
  return next.map(project => {
    const key = project.datasetName || project.name
    const candidates = temporary.get(key)
    const old = byId.get(project.dbProjectId) ?? (datasetCounts.get(key) === 1 && candidates?.length === 1 ? candidates[0] : undefined)
    return old ? { ...project, id: old.id, models: old.models?.length ? old.models : project.models } : project
  })
}
