/** 免登录路由仅限开发服务器、本机请求和本机后端，生产模式始终禁用。 */
export function allowLocalSession(requestUrl: string, backendUrl: string, environment: string | undefined, enabled: string | undefined) {
  if (environment !== 'development' || enabled === 'false') return false
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

/** The open-source manual workspace exposes data APIs without account services. */
export function isManualWorkspacePath(parts: string[]) {
  if (parts.some(part => !part || part === '.' || part === '..' || /[/\\%]/u.test(part))) return false
  if (parts.length === 1) return parts[0] === 'config' || parts[0] === 'health'
  return parts[0] === 'api' && ['projects', 'files', 'upload', 'train', 'results', 'data', 'models', 'runtime', 'stopwords'].includes(parts[1])
}
