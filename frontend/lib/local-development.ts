/** Desktop production builds require a host-provided secret; ordinary deployments stay closed. */
export function allowLocalSession(requestUrl: string, backendUrl: string, environment: string | undefined, enabled: string | undefined, desktopToken?: string) {
  const desktop = environment === 'production' && enabled === 'true' && (desktopToken?.length ?? 0) >= 32
  if ((!desktop && environment !== 'development') || enabled === 'false') return false
  const loopback = new Set(['127.0.0.1', 'localhost', '[::1]'])
  try {
    const request = new URL(requestUrl)
    const backend = new URL(backendUrl)
    return loopback.has(request.hostname) && loopback.has(backend.hostname)
      && ['http:', 'https:'].includes(backend.protocol)
  } catch { return false }
}

/** Next.js may normalize request.url to localhost; the browser's authority is Host. */
export function isLocalRequestOrigin(protocol: string, host: string | null, origin: string | null, fetchSite: string | null) {
  if (!host || fetchSite === 'cross-site') return false
  try {
    const authority = new URL(`${protocol}//${host}`)
    if (authority.host !== host || !['127.0.0.1', 'localhost', '[::1]'].includes(authority.hostname)) return false
    return !origin || origin === authority.origin
  } catch { return false }
}

/** Local workspace APIs; hosted account and storage services remain unavailable. */
export function isManualWorkspacePath(parts: string[]) {
  if (parts.some(part => !part || part === '.' || part === '..' || /[/\\%]/u.test(part))) return false
  if (parts.length === 1) return parts[0] === 'config' || parts[0] === 'health'
  if (parts[0] !== 'api') return false
  if (parts[1] === 'datasets') return parts.length === 3 || parts.length === 4 && ['preview', 'combine'].includes(parts[3])
  if (parts[1] === 'preprocessing') return parts.length === 4 && parts[2] === 'check'
  return parts[0] === 'api' && ['projects', 'files', 'upload', 'train', 'results', 'data', 'models', 'runtime', 'stopwords'].includes(parts[1])
}
