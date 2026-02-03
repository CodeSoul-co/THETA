"use client"

import { useState, useCallback, useEffect } from "react"
import { useRouter } from "next/navigation"
import { AppShell, type Tab } from "@/components/layout/app-shell"
import { ProjectHub, type Project } from "@/components/dashboard/project-hub"
import { NewProjectDialog, type NewProjectData } from "@/components/dashboard/new-project-dialog"
import { AutoPipeline } from "@/components/project/auto-pipeline"
import type { ChatMessage } from "@/components/chat/ai-sidebar"
import { ProtectedRoute } from "@/components/protected-route"
import { ETMAgentAPI, DatasetInfo } from "@/lib/api/etm-agent"

// Helper to generate timestamp
function getTimestamp() {
  return new Date().toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" })
}

// Generate unique ID
function generateId() {
  return `msg-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`
}

// Extended project type with additional fields
interface WorkspaceProject extends Project {
  description?: string
  datasetName?: string
  mode?: "zero_shot" | "unsupervised" | "supervised"
  numTopics?: number
  pipelineStatus?: "running" | "completed" | "error"
}

function DashboardContent() {
  const router = useRouter()
  const [tabs, setTabs] = useState<Tab[]>([
    { id: "hub", title: "项目中心", closable: false },
  ])
  const [activeTabId, setActiveTabId] = useState("hub")
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([])
  const [isNewProjectDialogOpen, setIsNewProjectDialogOpen] = useState(false)
  const [projects, setProjects] = useState<WorkspaceProject[]>([])
  const [isLoading, setIsLoading] = useState(true)

  // Load datasets from backend on mount
  useEffect(() => {
    const loadDatasets = async () => {
      try {
        const datasets = await ETMAgentAPI.getDatasets()
        const projectsFromBackend: WorkspaceProject[] = datasets.map((ds: DatasetInfo) => ({
          id: `proj-${ds.name}`,
          name: ds.name,
          rows: ds.size || ds.file_count || 0,
          createdAt: "已上传",
          status: "completed" as const,
          datasetName: ds.name,
        }))
        setProjects(projectsFromBackend)
      } catch (error) {
        console.error("Failed to load datasets:", error)
      } finally {
        setIsLoading(false)
      }
    }
    loadDatasets()
  }, [])

  // 刷新项目列表
  const refreshProjects = useCallback(async () => {
    setIsLoading(true)
    try {
      const datasets = await ETMAgentAPI.getDatasets()
      const projectsFromBackend: WorkspaceProject[] = datasets.map((ds: DatasetInfo) => ({
        id: `proj-${ds.name}`,
        name: ds.name,
        rows: ds.size || ds.file_count || 0,
        createdAt: "已上传",
        status: "completed" as const,
        datasetName: ds.name,
      }))
      // 合并已有项目（保留正在运行的pipeline项目）
      setProjects(prev => {
        const runningProjects = prev.filter(p => p.pipelineStatus === "running")
        const newProjects = projectsFromBackend.filter(np => 
          !runningProjects.find(rp => rp.id === np.id)
        )
        return [...runningProjects, ...newProjects]
      })
    } catch (error) {
      console.error("Failed to refresh datasets:", error)
    } finally {
      setIsLoading(false)
    }
  }, [])

  const handleOpenProject = (projectId: string) => {
    const existingTab = tabs.find((tab) => tab.id === projectId)
    if (existingTab) {
      setActiveTabId(projectId)
    } else {
      const project = projects.find(p => p.id === projectId)
      const projectName = project?.name || "Project"
      const newTab: Tab = {
        id: projectId,
        title: projectName,
        closable: true,
      }
      setTabs([...tabs, newTab])
      setActiveTabId(projectId)
    }
  }

  // 创建新项目（仅项目名；分析模式、主题数在数据预处理后再配置，此处用默认值）
  const handleCreateProject = useCallback((data: NewProjectData) => {
    const projectId = `proj-${Date.now()}`
    const datasetName = data.name.trim().replace(/\s+/g, "_").replace(/[^\w\u4e00-\u9fa5-]/g, "").toLowerCase() || "dataset"

    const newProject: WorkspaceProject = {
      id: projectId,
      name: data.name,
      datasetName,
      mode: "zero_shot",
      numTopics: 20,
      rows: 0,
      createdAt: "刚刚",
      status: "vectorizing",
      pipelineStatus: "running",
    }

    setProjects(prev => [newProject, ...prev])

    const newTab: Tab = {
      id: projectId,
      title: data.name,
      closable: true,
    }
    setTabs(prev => [...prev, newTab])
    setActiveTabId(projectId)
    setIsNewProjectDialogOpen(false)
  }, [])

  // Pipeline 完成回调（可拿到 result.dataset 更新项目）
  const handlePipelineComplete = useCallback((projectId: string, result?: { dataset?: string } | null) => {
    setProjects(prev => prev.map(p =>
      p.id === projectId
        ? {
            ...p,
            status: "completed" as const,
            pipelineStatus: "completed",
            ...(result?.dataset && { datasetName: result.dataset }),
          }
        : p
    ))
    refreshProjects()
  }, [refreshProjects])

  const handlePipelineError = useCallback((projectId: string) => {
    setProjects(prev => prev.map(p =>
      p.id === projectId ? { ...p, status: "completed" as const, pipelineStatus: "error" } : p
    ))
  }, [])

  const handleTabChange = (tabId: string) => {
    setActiveTabId(tabId)
  }

  const handleTabClose = (tabId: string) => {
    const tab = tabs.find((t) => t.id === tabId)
    if (!tab?.closable) return

    const newTabs = tabs.filter((t) => t.id !== tabId)
    setTabs(newTabs)

    if (activeTabId === tabId) {
      setActiveTabId("hub")
    }
  }

  // 删除项目：后端有对应数据集则调用删除接口，并关闭该项目的标签页
  const handleDeleteProject = useCallback(async (projectId: string) => {
    const project = projects.find((p) => p.id === projectId)
    if (!project) return

    const datasetName = project.datasetName || (projectId.startsWith("proj-") ? projectId.replace(/^proj-/, "") : null)
    if (datasetName) {
      try {
        await ETMAgentAPI.deleteDataset(datasetName)
      } catch (error) {
        const msg = error instanceof Error ? error.message : String(error)
        if (msg.includes("404") || msg.includes("not found")) {
          // 后端无此数据集（例如仅前端新建未上传），仍从本地移除
        } else {
          console.error("删除数据集失败:", error)
          return
        }
      }
    }

    setProjects((prev) => prev.filter((p) => p.id !== projectId))
    const newTabs = tabs.filter((t) => t.id !== projectId)
    setTabs(newTabs)
    if (activeTabId === projectId) {
      setActiveTabId("hub")
    }
  }, [projects, tabs, activeTabId])

  // Chat handlers
  const handleSendMessage = useCallback(async (content: string) => {
    const userMessage: ChatMessage = {
      id: generateId(),
      role: "user",
      content,
      type: "text",
      timestamp: getTimestamp(),
    }
    setChatHistory((prev) => [...prev, userMessage])

    try {
      const response = await ETMAgentAPI.chat(content, {
        current_view_name: "项目中心",
        current_view: activeTabId === "hub" ? "hub" : "workspace",
        app_state: "workspace",
        datasets_count: projects.length,
        datasets: projects.map((p) => ({ name: p.name, fileCount: p.rows })),
        processing_jobs_count: 0,
        selected_dataset: activeTabId !== "hub" ? projects.find((p) => p.id === activeTabId)?.name : undefined,
      })

      const text = response.message ?? (response as { response?: string }).response ?? ""
      const aiMessage: ChatMessage = {
        id: generateId(),
        role: "ai",
        content: text,
        type: "text",
        timestamp: getTimestamp(),
      }
      setChatHistory((prev) => [...prev, aiMessage])
    } catch (error) {
      const aiMessage: ChatMessage = {
        id: generateId(),
        role: "ai",
        content: `无法连接服务或请求失败。请确认后端已启动。`,
        type: "text",
        timestamp: getTimestamp(),
        followUpQuestions: ["如何开始？", "支持哪些格式？"],
      }
      setChatHistory((prev) => [...prev, aiMessage])
    }
  }, [projects, activeTabId])

  const handleDataUploaded = useCallback(async (file: File) => {
    console.log("Data uploaded:", file.name)
  }, [])

  const handleFocusChart = useCallback((chartId: string) => {
    console.log("Focus chart:", chartId)
  }, [])

  const handleClearChat = useCallback(() => {
    setChatHistory([])
  }, [])

  // 渲染内容
  const renderContent = () => {
    if (activeTabId === "hub") {
      return (
        <ProjectHub 
          onProjectSelect={handleOpenProject} 
          onNewProject={() => setIsNewProjectDialogOpen(true)}
          onDeleteProject={handleDeleteProject}
          projects={projects}
          isLoading={isLoading}
        />
      )
    }

    // 查找当前项目
    const currentProject = projects.find(p => p.id === activeTabId)
    
    if (!currentProject) {
      return (
        <div className="p-8 text-center">
          <h2 className="text-xl font-semibold text-slate-900 mb-2">项目未找到</h2>
          <p className="text-slate-500">该项目可能已被删除</p>
        </div>
      )
    }

    // 新建项目：先上传数据，再自动执行流程。出错时也留在本页以便重试
    if (currentProject.pipelineStatus === "running" || currentProject.pipelineStatus === "error") {
      return (
        <AutoPipeline
          projectName={currentProject.name}
          mode={currentProject.mode || "zero_shot"}
          numTopics={currentProject.numTopics || 20}
          onComplete={(result) => handlePipelineComplete(currentProject.id, result)}
          onError={() => handlePipelineError(currentProject.id)}
        />
      )
    }

    // 已完成的项目显示结果概览
    return (
      <ProjectResultView project={currentProject} />
    )
  }

  return (
    <>
      <AppShell
        tabs={tabs}
        activeTabId={activeTabId}
        onTabChange={handleTabChange}
        onTabClose={handleTabClose}
        chatHistory={chatHistory}
        onSendMessage={handleSendMessage}
        onDataUploaded={handleDataUploaded}
        onFocusChart={handleFocusChart}
        onClearChat={handleClearChat}
      >
        {renderContent()}
      </AppShell>
      
      <NewProjectDialog
        open={isNewProjectDialogOpen}
        onOpenChange={setIsNewProjectDialogOpen}
        onSubmit={handleCreateProject}
      />
    </>
  )
}

// 项目结果视图
function ProjectResultView({ project }: { project: WorkspaceProject }) {
  const [metrics, setMetrics] = useState<Record<string, number> | null>(null)
  const [topicWords, setTopicWords] = useState<Record<string, string[]> | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const loadResults = async () => {
      if (!project.datasetName) {
        setLoading(false)
        return
      }

      try {
        const [metricsData, wordsData] = await Promise.all([
          ETMAgentAPI.getMetrics(project.datasetName, project.mode || "zero_shot").catch(() => null),
          ETMAgentAPI.getTopicWords(project.datasetName, project.mode || "zero_shot").catch(() => null),
        ])
        setMetrics(metricsData as Record<string, number> | null)
        setTopicWords(wordsData)
      } catch (error) {
        console.error("Failed to load results:", error)
      } finally {
        setLoading(false)
      }
    }

    loadResults()
  }, [project.datasetName, project.mode])

  if (loading) {
    return (
      <div className="p-8 flex items-center justify-center">
        <div className="text-center">
          <div className="w-8 h-8 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-slate-500">加载结果中...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="p-6 lg:p-8">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-slate-900 mb-2">{project.name}</h1>
        <p className="text-slate-500">
          数据集: {project.datasetName || project.name} · 
          模式: {project.mode || "zero_shot"} · 
          主题数: {project.numTopics || 20}
        </p>
      </div>

      {/* 评估指标 */}
      {metrics && Object.keys(metrics).length > 0 && (
        <div className="mb-6">
          <h3 className="font-semibold text-slate-900 mb-3">评估指标</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {Object.entries(metrics).filter(([_, v]) => typeof v === "number").slice(0, 8).map(([key, value]) => (
              <div key={key} className="p-4 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl border border-blue-100">
                <p className="text-xs text-blue-600 uppercase mb-1">{key}</p>
                <p className="text-2xl font-bold text-blue-700">
                  {typeof value === "number" ? value.toFixed(4) : value}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 主题词 */}
      {topicWords && Object.keys(topicWords).length > 0 && (
        <div>
          <h3 className="font-semibold text-slate-900 mb-3">主题分析结果</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {Object.entries(topicWords).map(([topicId, words]) => (
              <div key={topicId} className="p-4 bg-white rounded-xl border border-slate-200 shadow-sm">
                <p className="text-sm font-semibold text-slate-700 mb-2">
                  主题 {parseInt(topicId) + 1}
                </p>
                <div className="flex flex-wrap gap-1">
                  {(words as string[]).slice(0, 8).map((word, idx) => (
                    <span
                      key={idx}
                      className={`px-2 py-1 rounded text-xs font-medium ${
                        idx < 3
                          ? "bg-blue-100 text-blue-700"
                          : "bg-slate-100 text-slate-600"
                      }`}
                    >
                      {word}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 空状态 */}
      {(!metrics || Object.keys(metrics).length === 0) && (!topicWords || Object.keys(topicWords).length === 0) && (
        <div className="text-center py-12">
          <div className="w-16 h-16 bg-slate-100 rounded-2xl flex items-center justify-center mx-auto mb-4">
            <svg className="w-8 h-8 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </div>
          <p className="text-slate-500">暂无分析结果</p>
          <p className="text-sm text-slate-400 mt-1">该项目可能尚未完成训练</p>
        </div>
      )}
    </div>
  )
}

// Dashboard page with auth protection
export default function DashboardPage() {
  return (
    <ProtectedRoute>
      <DashboardContent />
    </ProtectedRoute>
  )
}
