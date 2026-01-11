"use client"

import { useState } from "react"
import { SidebarNav } from "@/components/sidebar-nav"
import { DataGovernance } from "@/components/data-governance"
import { RagWorkbench } from "@/components/rag-workbench"
import { EnhancedReader } from "@/components/enhanced-reader"
import { CoreAnalysis } from "@/components/core-analysis"
import { AgentPanel } from "@/components/agent-panel"
import { DashboardOverview } from "@/components/dashboard-overview"

export default function Theta() {
  const [activeSection, setActiveSection] = useState("home")
  const [triggerDataAgent, setTriggerDataAgent] = useState(false)
  const [triggerAnalysisAgent, setTriggerAnalysisAgent] = useState(false)

  const handleUpload = () => {
    setTriggerDataAgent(true)
  }

  const renderMainContent = () => {
    switch (activeSection) {
      case "home":
        return <DashboardOverview />
      case "data":
      case "upload":
        return <DataGovernance onUpload={handleUpload} />
      case "rag":
        return <RagWorkbench />
      case "reader":
        return <EnhancedReader />
      case "analysis":
      case "lora":
        return <CoreAnalysis />
      default:
        return <DashboardOverview />
    }
  }

  return (
    <div className="flex h-screen overflow-hidden bg-background">
      {/* Left Sidebar Navigation */}
      <SidebarNav activeSection={activeSection} onSectionChange={setActiveSection} />

      {/* Main Content Area */}
      <main className="flex-1 overflow-auto">
        <div className="p-6 max-w-5xl mx-auto">{renderMainContent()}</div>
      </main>

      {/* Right Agent Panel */}
      <AgentPanel
        triggerDataAgent={triggerDataAgent}
        triggerAnalysisAgent={activeSection === "analysis" || activeSection === "lora"}
      />
    </div>
  )
}
