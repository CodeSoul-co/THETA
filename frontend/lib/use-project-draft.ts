"use client"
import { useEffect, useRef, useState, type SetStateAction } from 'react'
import { readDraft, writeDraft } from './workbench-draft'

/** Drafts contain form values and IDs, never file contents or runtime credentials. */
export function useProjectDraft<T>(key: string, fallback: T): [T, (value: SetStateAction<T>) => void] {
  const [value, setValue] = useState<T>(fallback)
  const current = useRef({ key, value: fallback })
  const activeKey = useRef(key)
  activeKey.current = key
  useEffect(() => { const restored = readDraft(key, fallback); current.current = { key, value: restored }; setValue(restored) }, [key])
  return [value, next => {
    const previous = current.current.key === key ? current.current.value : readDraft(key, fallback)
    const value = typeof next === 'function' ? (next as (old: T) => T)(previous) : next
    writeDraft(key, value)
    // An async response from a project we left must not overwrite the new one.
    if (activeKey.current === key) { current.current = { key, value }; setValue(value) }
  }]
}
