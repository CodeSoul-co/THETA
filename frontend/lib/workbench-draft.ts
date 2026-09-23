const PREFIX = 'theta:workbench:v1:'
export function readDraft<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(PREFIX + key)
    if (!raw) return fallback
    const saved = JSON.parse(raw)
    if (saved?.version !== 1 || saved.value == null) return fallback
    // Reject incompatible snapshots instead of breaking the workbench after a reload.
    if (fallback !== null && (typeof saved.value !== typeof fallback || Array.isArray(saved.value) !== Array.isArray(fallback))) return fallback
    return saved.value as T
  } catch { return fallback }
}
export function writeDraft(key: string, value: unknown) {
  try { localStorage.setItem(PREFIX + key, JSON.stringify({ version: 1, value })) } catch { /* Storage unavailable: server tasks remain authoritative. */ }
}
