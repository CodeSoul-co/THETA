"use client"

import React, { useState } from "react"
import { X, FolderOpen } from "lucide-react"
import { ReactNode } from "react"

export type Tab = {
  id: string
  title: string
  closable: boolean
}

interface AppShellProps {
  embedded?: boolean
  tabs: Tab[]
  activeTabId: string
  onTabChange: (tabId: string) => void
  onTabClose: (tabId: string) => void
  onTabsReorder?: (fromIndex: number, toIndex: number) => void
  children: ReactNode
  assistant?: ReactNode

}

export function AppShell({
  embedded = false,
  tabs,
  activeTabId,
  onTabChange,
  onTabClose,
  onTabsReorder,
  children,
  assistant,
}: AppShellProps) {
  const [dragOverIndex, setDragOverIndex] = useState<number | null>(null)

  return (
    <div className={`${embedded ? "h-full" : "h-screen"} w-full min-w-0 flex flex-col bg-slate-50 font-sans antialiased overflow-hidden`}>
      {/* Top Navigation Bar */}
      {!embedded && <header className="h-14 flex-shrink-0 bg-white/90 backdrop-blur-md border-b border-slate-200/60 flex items-center justify-between px-4 sm:px-6 shadow-[0_1px_3px_rgba(0,0,0,0.05)] min-w-0">
        {/* Left: Logo */}
        <div className="flex items-center gap-2 sm:gap-4 min-w-0 flex-1 overflow-hidden">
          <img src="/theta-logo.png" alt="THETA" className="h-9 sm:h-10 w-auto flex-shrink-0" />
          <div className="h-5 w-px bg-gradient-to-b from-transparent via-slate-200 to-transparent hidden sm:block" />
          <span className="text-xs font-medium text-slate-400 hidden sm:block truncate tracking-wide">智能分析平台</span>
        </div>

        {/* Center: Empty */}
        <div className="hidden md:flex items-center gap-1 flex-shrink-0">
        </div>

        <div aria-hidden="true" />
      </header>}

      {/* Main Content Area - min-w-0 so flex children can shrink */}
      <div className="flex-1 flex overflow-hidden min-h-0 min-w-0">
        {/* Center Workspace - Project Hub */}
        <div className="flex-1 flex flex-col overflow-hidden min-w-0 min-h-0">
          {/* Tab Bar */}
          <div className="h-11 bg-white/80 border-b border-slate-200 flex items-center px-2 sm:px-4 gap-1 overflow-x-auto">
            {tabs.map((tab, idx) => {
              const isHub = tab.id === "hub"
              const isActive = activeTabId === tab.id
              const isDragOver = dragOverIndex === idx
              return (
                <div
                  key={tab.id}
                  draggable={!isHub && onTabsReorder !== undefined}
                  onDragStart={(e) => {
                    if (isHub) return
                    e.dataTransfer.effectAllowed = "move"
                    e.dataTransfer.setData("text/plain", String(idx))
                    // Set a delay to prevent visual glitch
                    const el = e.currentTarget
                    requestAnimationFrame(() => el.classList.add("opacity-50"))
                  }}
                  onDragEnd={(e) => {
                    e.currentTarget.classList.remove("opacity-50")
                    setDragOverIndex(null)
                  }}
                  onDragOver={(e) => {
                    if (isHub || !onTabsReorder) return
                    e.preventDefault()
                    e.dataTransfer.dropEffect = "move"
                    setDragOverIndex(idx)
                  }}
                  onDragLeave={(e) => {
                    if (!e.currentTarget.contains(e.relatedTarget as Node)) {
                      setDragOverIndex(null)
                    }
                  }}
                  onDrop={(e) => {
                    if (isHub || !onTabsReorder) return
                    e.preventDefault()
                    const fromIdx = parseInt(e.dataTransfer.getData("text/plain"), 10)
                    setDragOverIndex(null)
                    if (fromIdx !== idx) {
                      onTabsReorder(fromIdx, idx)
                    }
                  }}
                  onClick={() => onTabChange(tab.id)}
                  className={`group relative flex items-center gap-1.5 sm:gap-2 px-3 sm:px-4 h-8 rounded-lg cursor-pointer transition-all duration-200 whitespace-nowrap ${
                    isActive
                      ? isHub
                        ? "bg-blue-50 text-blue-700 ring-1 ring-blue-200/80"
                        : "bg-slate-100 text-slate-800 ring-1 ring-slate-200"
                      : "text-slate-500 hover:bg-slate-50 hover:text-slate-700"
                  } ${isDragOver ? "ring-2 ring-blue-400 ring-offset-1" : ""} ${!isHub && onTabsReorder ? "cursor-grab active:cursor-grabbing" : ""}`}
                >
                  {isHub && <FolderOpen className="h-3.5 w-3.5 shrink-0" />}
                  <span className="text-xs sm:text-sm font-medium">{tab.title}</span>
                  {tab.closable && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        onTabClose(tab.id)
                      }}
                      className="opacity-0 group-hover:opacity-100 p-0.5 rounded hover:bg-slate-200 transition-all"
                    >
                      <X className="h-3 w-3" />
                    </button>
                  )}
                </div>
              )
            })}
          </div>

          {/* Content Viewport */}
          <div className="min-h-0 min-w-0 flex-1 overflow-auto">
            {children}
          </div>
        </div>

        {assistant}
      </div>
    </div>
  )
}
