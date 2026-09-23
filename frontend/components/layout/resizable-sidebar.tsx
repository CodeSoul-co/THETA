"use client"

import { useEffect, useRef, useState, type ReactNode } from "react"
import css from "./resizable-sidebar.module.css"

/** Shared pointer/keyboard resize behavior for both workbenches. */
export function ResizableSidebar({ children, storageKey, label, defaultRatio = .3, hidden = false }: {
  children: ReactNode
  storageKey: string
  label: string
  defaultRatio?: number
  hidden?: boolean
}) {
  const panel = useRef<HTMLDivElement>(null)
  const [ratio, setRatio] = useState(() => {
    try {
      const saved = Number(localStorage.getItem(storageKey))
      return saved >= .15 && saved <= .6 ? saved : defaultRatio
    } catch { return defaultRatio }
  })
  const [available, setAvailable] = useState(1200)
  const [resizing, setResizing] = useState(false)
  const drag = useRef<{ x: number; width: number } | null>(null)
  const minWidth = Math.min(300, available * .5)
  const maxWidth = Math.max(minWidth, Math.min(760, available * .6))
  const width = Math.max(minWidth, Math.min(maxWidth, available * ratio))
  const resize = (pixels: number) => setRatio(Math.max(minWidth, Math.min(maxWidth, pixels)) / available)

  useEffect(() => {
    const container = panel.current?.parentElement
    if (!container) return
    const observer = new ResizeObserver(() => {
      if (container.clientWidth > 0) setAvailable(container.clientWidth)
    })
    observer.observe(container)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    if (resizing) return
    try { localStorage.setItem(storageKey, String(ratio)) } catch { /* Layout works without storage. */ }
  }, [ratio, resizing, storageKey])

  useEffect(() => {
    if (!resizing) return
    const { cursor, userSelect } = document.body.style
    document.body.style.cursor = "col-resize"
    document.body.style.userSelect = "none"
    return () => { document.body.style.cursor = cursor; document.body.style.userSelect = userSelect }
  }, [resizing])

  return <div ref={panel} className={css.panel} style={{ width, ...(hidden ? { display: 'none' } : {}) }} data-resizing={resizing || undefined}>
    <div role="separator" tabIndex={0} aria-label={label} aria-orientation="vertical"
      aria-valuemin={Math.round(minWidth)} aria-valuemax={Math.round(maxWidth)} aria-valuenow={Math.round(width)}
      aria-valuetext={`${Math.round(width)} 像素`} title="拖动调整宽度 · 双击复原 · 方向键微调"
      className={css.handle}
      onPointerDown={event => {
        if (event.button !== 0) return
        event.preventDefault()
        event.currentTarget.setPointerCapture(event.pointerId)
        drag.current = { x: event.clientX, width }
        setResizing(true)
      }}
      onPointerMove={event => { if (drag.current) resize(drag.current.width + drag.current.x - event.clientX) }}
      onPointerUp={() => { drag.current = null; setResizing(false) }}
      onPointerCancel={() => { drag.current = null; setResizing(false) }}
      onLostPointerCapture={() => { drag.current = null; setResizing(false) }}
      onDoubleClick={() => setRatio(defaultRatio)}
      onKeyDown={event => {
        if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return
        event.preventDefault()
        resize(event.key === 'Home' ? minWidth : event.key === 'End' ? maxWidth : width + (event.key === 'ArrowLeft' ? 24 : -24))
      }}><span /></div>
    {children}
  </div>
}
