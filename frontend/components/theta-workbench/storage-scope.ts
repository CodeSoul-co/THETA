import { OPEN_SOURCE_EDITION } from '@/lib/edition'

const storedUserId = (): string | undefined => {
  try {
    const stored = JSON.parse(localStorage.getItem('user') ?? 'null') as { id?: string | number } | null
    if (stored?.id !== undefined && stored.id !== null && String(stored.id).trim()) return String(stored.id)
  } catch {
    // Ignore malformed cached auth state and use the anonymous scope.
  }
  return undefined
}

export const accountStorageScope = (userId?: string | number | null): string => {
  if (OPEN_SOURCE_EDITION) return 'opensource-local'
  const value = userId !== undefined && userId !== null && String(userId).trim()
    ? String(userId)
    : storedUserId()
  return value ? encodeURIComponent(value) : 'anonymous'
}

export const accountStorageKey = (baseKey: string, userId?: string | number | null): string =>
  `${baseKey}.${accountStorageScope(userId)}`
