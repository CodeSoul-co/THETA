'use client'

import { droppedFiles } from './dataset-files'
import { useRef, useState, type DragEvent } from 'react'

/** Handle file drags only. Stop bubbling so nested upload areas cannot upload twice. */
export function useFileDrop(onFiles: (files: File[]) => void, disabled = false, onError: (message: string) => void = () => {}) {
  const reading = useRef(false)
  const unavailable = useRef(disabled)
  unavailable.current = disabled
  const depth = useRef(0)
  const [dragging, setDragging] = useState(false)
  const isFile = (event: DragEvent) => Array.from(event.dataTransfer.types).includes('Files')
  const reset = () => { depth.current = 0; setDragging(false) }
  return {
    dragging: dragging && !disabled,
    dropProps: {
      onDragEnter(event: DragEvent) {
        if (!isFile(event)) return
        event.preventDefault(); event.stopPropagation()
        depth.current++
        if (!disabled) setDragging(true)
      },
      onDragOver(event: DragEvent) {
        if (!isFile(event)) return
        event.preventDefault(); event.stopPropagation()
        event.dataTransfer.dropEffect = disabled ? 'none' : 'copy'
      },
      onDragLeave(event: DragEvent) {
        if (!isFile(event)) return
        event.preventDefault(); event.stopPropagation()
        if (--depth.current <= 0) reset()
      },
      onDrop(event: DragEvent) {
        if (!isFile(event)) return
        event.preventDefault(); event.stopPropagation(); reset()
        if (disabled || reading.current) return
        reading.current = true
        void droppedFiles(event.dataTransfer).then(files => { if (!unavailable.current) onFiles(files) })
          .catch(error => onError(error instanceof Error ? error.message : String(error)))
          .finally(() => { reading.current = false })
      },
      onDragEnd: reset,
    },
  }
}
