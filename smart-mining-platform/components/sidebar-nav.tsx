"use client"

import { useState } from "react"
import Image from "next/image"
import {
  Database,
  FileSearch,
  Settings,
  ChevronRight,
  Home,
  Upload,
  BookOpen,
  BarChart3,
  Layers,
} from "lucide-react"
import { cn } from "@/lib/utils"

interface SidebarNavProps {
  activeSection: string
  onSectionChange: (section: string) => void
}

const navItems = [
  { id: "home", label: "概览", icon: Home },
  { id: "data", label: "数据管理", icon: Database },
  { id: "upload", label: "数据上传", icon: Upload },
  { id: "rag", label: "RAG工作台", icon: FileSearch },
  { id: "reader", label: "增强阅读器", icon: BookOpen },
  { id: "analysis", label: "核心分析", icon: BarChart3 },
  { id: "lora", label: "LoRA适配器", icon: Layers },
  { id: "settings", label: "系统设置", icon: Settings },
]

export function SidebarNav({ activeSection, onSectionChange }: SidebarNavProps) {
  const [isCollapsed, setIsCollapsed] = useState(false)

  return (
    <aside
      className={cn(
        "flex flex-col h-full bg-sidebar border-r border-sidebar-border transition-all duration-300",
        isCollapsed ? "w-16" : "w-56",
      )}
    >
      <div className="flex items-center gap-3 px-4 py-5 border-b border-sidebar-border">
        <div className="flex items-center justify-center w-9 h-9 rounded-lg overflow-hidden">
          <Image src="/thetalogo.jpeg" alt="THETA" width={36} height={36} className="object-contain" />
        </div>
        {!isCollapsed && (
          <div className="flex flex-col">
            <span className="text-sm font-semibold text-sidebar-foreground">THETA</span>
          </div>
        )}
      </div>

      <nav className="flex-1 py-4 px-2 space-y-1 overflow-y-auto">
        {navItems.map((item) => (
          <button
            key={item.id}
            onClick={() => onSectionChange(item.id)}
            className={cn(
              "w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all",
              activeSection === item.id
                ? "bg-sidebar-accent text-sidebar-primary"
                : "text-muted-foreground hover:text-sidebar-foreground hover:bg-sidebar-accent/50",
            )}
          >
            <item.icon className="w-4 h-4 flex-shrink-0" />
            {!isCollapsed && <span>{item.label}</span>}
            {!isCollapsed && activeSection === item.id && <ChevronRight className="w-4 h-4 ml-auto" />}
          </button>
        ))}
      </nav>

      <button
        onClick={() => setIsCollapsed(!isCollapsed)}
        className="p-3 border-t border-sidebar-border text-muted-foreground hover:text-sidebar-foreground transition-colors"
      >
        <ChevronRight className={cn("w-4 h-4 mx-auto transition-transform", isCollapsed ? "" : "rotate-180")} />
      </button>
    </aside>
  )
}
