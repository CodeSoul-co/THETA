"use client"

import { useState, useCallback, useEffect, useRef } from "react"
import { AppShell, type Tab } from "@/components/layout/app-shell"
import { ProjectHub, type Project } from "@/components/dashboard/project-hub"
import { NewProjectDialog, type NewProjectData } from "@/components/dashboard/new-project-dialog"
import { AutoPipeline } from "@/components/project/auto-pipeline"
import { ProjectAssistant } from "@/components/chat/project-assistant"
import { apiFetch, API_BASE } from "@/lib/api/config"
import { ETMAgentAPI, DatasetInfo } from "@/lib/api/etm-agent"
import { Spinner } from "@/components/ui/spinner"
import { ResultWorkspace } from "@/components/results/result-workspace"
import { useProjectDraft } from "@/lib/use-project-draft"
import { reconcileProjectIdentity } from "@/lib/project-identity"

// Extended project type with additional fields
interface WorkspaceProject extends Project {
  description?: string
  datasetName?: string
  mode?: "zero_shot" | "unsupervised" | "supervised"
  models?: string[]
  numTopics?: number
  pipelineStatus?: "running" | "completed" | "error" | "draft"
  dbProjectId?: number  // 数据库项目 ID，用于更新/删除
  taskId?: string | null  // 关联的训练任务 ID
}

const STORAGE_KEYS = {
  TABS: "theta_dashboard_tabs",
  ACTIVE_TAB: "theta_dashboard_active_tab",
} as const

/** Owns manual projects and training; only completed artifacts enter ResultWorkspace. */
export function ManualWorkbench() {
  // Initialize tabs from sessionStorage
  const [tabs, setTabs] = useState<Tab[]>(() => {
    if (typeof window !== "undefined") {
      const saved = sessionStorage.getItem(STORAGE_KEYS.TABS)
      if (saved) {
        try {
          const parsed = JSON.parse(saved)
          if (Array.isArray(parsed) && parsed.length > 0) return parsed
        } catch { /* ignore */ }
      }
    }
    return [{ id: "hub", title: "项目中心", closable: false }]
  })

  // Initialize activeTabId from sessionStorage
  const [activeTabId, setActiveTabId] = useState<string>(() => {
    if (typeof window !== "undefined") {
      const saved = sessionStorage.getItem(STORAGE_KEYS.ACTIVE_TAB)
      if (saved) return saved
    }
    return "hub"
  })
  const [resultProjectId, setResultProjectId] = useProjectDraft<string | null>(`manual:${activeTabId}:result-project`, null)
  const [isNewProjectDialogOpen, setIsNewProjectDialogOpen] = useState(false)
  const [projects, setProjects] = useState<WorkspaceProject[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [projectTransitionName, setProjectTransitionName] = useState<string | null>(null)
  /** 用于强制重新渲染 renderContent（PLC 完成切换到结果视图时使用） */
  const [renderKey, setRenderKey] = useState(0)

  const transitionTimerRef = useRef<number | null>(null)
  const pollingTimerRef = useRef<number | null>(null)
  const syncTimerRef = useRef<number | null>(null)
  const refreshProjectsRef = useRef<() => Promise<void>>(async () => {})


  useEffect(() => {
    return () => {
      if (transitionTimerRef.current) {
        window.clearTimeout(transitionTimerRef.current)
      }
      if (syncTimerRef.current) {
        window.clearInterval(syncTimerRef.current)
      }
    }
  }, [])

  // Load projects: 优先数据库（用户关联），再合并 datasets + tasks
  const projectLoadStarted = useRef(false)
  useEffect(() => {
    if (projectLoadStarted.current) return
    projectLoadStarted.current = true
    const loadProjects = async () => {
      try {
        console.log("[Dashboard] Loading projects...");
        const [dbProjects, datasets, tasks, ossInfo] = await Promise.all([
          ETMAgentAPI.getProjects(),
          ETMAgentAPI.getDatasets(),
          ETMAgentAPI.getTasks({ limit: 100 }).catch(() => []),
          ETMAgentAPI.listOssDatasets().catch(() => ({ datasets: [] as { name: string; chart_count: number }[] })),
        ])

        // 构建有结果的 dataset 名称集合（OSS 上有图表文件即为有结果）
        const datasetsWithResults = new Set(ossInfo.datasets.map((d: { name: string }) => d.name))

        console.log("[Dashboard] Results - datasets:", datasets);
        console.log("[Dashboard] OSS datasets with results:", datasetsWithResults);
        const seen = new Set<string>()
        const list: WorkspaceProject[] = []

        // 预构建 dataset→task 映射：优先已完成的任务
        const taskByDataset = new Map<string, { task_id: string; status: string; pipeline_status?: string }>()
        for (const t of tasks) {
          const ds = t.dataset || (t as any).dataset_name
          if (!ds) continue
          const existing = taskByDataset.get(ds)
          if (!existing || (t.status === "completed" && existing.status !== "completed")) {
            taskByDataset.set(ds, { task_id: t.task_id, status: t.status, pipeline_status: t.status === "completed" ? "completed" : t.status === "failed" ? "error" : "running" })
          }
        }

        // 1. 数据库中的项目（用户关联，跨设备同步）
        for (const p of dbProjects) {
          const key = p.dataset_name || `db-${p.id}`
          seen.add(key)
          // 如果项目没有 task_id，尝试从任务列表中匹配
          let effectiveTaskId = p.task_id ?? null
          let effectivePipelineStatus = p.pipeline_status
          if (!effectiveTaskId && p.dataset_name && p.pipeline_status !== 'draft') {
            const matched = taskByDataset.get(p.dataset_name)
            if (matched) {
              effectiveTaskId = matched.task_id
              effectivePipelineStatus = effectivePipelineStatus || matched.pipeline_status
            }
          }
          // 如果数据库已标记完成，即使 OSS API 返回空也认为有结果（解决未登录时 OSS API 401 问题）
          const hasResults = p.dataset_name ? (datasetsWithResults.has(p.dataset_name) || p.pipeline_status === "completed") : false
          const derivedPipelineStatus = effectivePipelineStatus === "draft" ? "draft" : effectivePipelineStatus === "completed" ? "completed"
            : effectivePipelineStatus === "error" ? "error"
            : effectivePipelineStatus === "running" ? "running"
            : effectiveTaskId ? "running"
            : p.dataset_name && datasetsWithResults.has(p.dataset_name) ? "completed"
            : p.dataset_name ? "draft"
            : undefined
          list.push({
            id: `proj-db-${p.id}`,
            name: p.name,
            rows: 0,
            createdAt: p.created_at ? "已保存" : "刚刚",
            status: derivedPipelineStatus === "completed" ? "completed"
              : derivedPipelineStatus === "error" ? "no_result"
              : derivedPipelineStatus === "running" ? "vectorizing"
              : p.dataset_name ? "draft"
              : "draft" as const,
            datasetName: p.dataset_name ?? undefined,
            mode: (p.mode as any) ?? "zero_shot",
            models: ["theta"],
            numTopics: p.num_topics ?? 20,
            pipelineStatus: derivedPipelineStatus as any,
            hasResults,
            dbProjectId: p.id,
            taskId: effectiveTaskId,
          })
        }

        // 2. 数据集（未在 DB 中的）
        for (const ds of datasets) {
          if (seen.has(ds.name)) continue
          seen.add(ds.name)
          const hasResults = datasetsWithResults.has(ds.name)
          // 检查是否有正在运行的任务，如果有，使用任务状态
          let effectivePipelineStatus: "running" | "completed" | "error" | "draft" =
            hasResults ? "completed" : "draft"
          let effectiveTaskId: string | null = null
          const matchedTask = taskByDataset.get(ds.name)
          if (matchedTask) {
            effectivePipelineStatus = matchedTask.pipeline_status as any
            effectiveTaskId = matchedTask.task_id
          }
          list.push({
            id: `proj-${ds.name}`,
            name: ds.name,
            rows: ds.size ?? (ds as any).file_count ?? 0,
            createdAt: "已上传",
            status: hasResults && !effectivePipelineStatus ? "completed" as const : "draft" as const,
            pipelineStatus: effectivePipelineStatus,
            hasResults,
            datasetName: ds.name,
            taskId: effectiveTaskId,
            models: ["theta"],
          })
        }

        // 3. OSS 上已有结果但不在 DB files 中的数据集
        for (const ossDatasetName of Array.from(datasetsWithResults)) {
          if (seen.has(ossDatasetName)) continue
          seen.add(ossDatasetName)
          list.push({
            id: `proj-${ossDatasetName}`,
            name: ossDatasetName,
            rows: 0,
            createdAt: "已分析",
            status: "completed" as const,
            pipelineStatus: "completed" as const,
            hasResults: true,
            datasetName: ossDatasetName,
            models: ["theta"],
          })
        }

        // 3. 任务中的数据集
        for (const t of tasks) {
          const ds = t.dataset || (t as any).dataset_name
          if (ds && !seen.has(ds)) {
            seen.add(ds)
            const hasResults = datasetsWithResults.has(ds)
            list.push({
              id: `proj-${ds}`,
              name: ds,
              rows: 0,
              createdAt: "已分析",
              status: hasResults ? "completed" as const : (t.status === "completed" ? "no_result" as const : "vectorizing" as const),
              pipelineStatus: t.status === "completed" ? "completed" : t.status === "failed" ? "error" : "running",
              hasResults,
              datasetName: ds,
              models: ["theta"],
              taskId: t.task_id || null,
            })
          }
        }

        setProjects(list)
      } catch (error) {
        console.error("Failed to load projects:", error)
      } finally {
        setIsLoading(false)
      }
    }
    loadProjects()
  }, [])

  // Tabs only refer to persisted project identities; stale temporary tabs are not projects.
  useEffect(() => {
    if (isLoading) return
    const validIds = new Set(['hub', ...projects.map(project => project.id)])
    setTabs(previous => {
      const seen = new Set<string>()
      const next = previous.filter(tab => validIds.has(tab.id) && !seen.has(tab.id) && Boolean(seen.add(tab.id)))
      return next.length === previous.length ? previous : next
    })
    if (!validIds.has(activeTabId)) setActiveTabId("hub")
  }, [isLoading, projects, tabs, activeTabId])

  // 轮询训练状态
  useEffect(() => {
    const pollTrainingStatus = async () => {
      const runningJobs = projects.filter(p => p.pipelineStatus === "running" || p.status === "vectorizing");
      if (runningJobs.length === 0) return;

      const jobIds: number[] = [];
      for (const job of runningJobs) {
        if (job.taskId) {
          const numId = parseInt(job.taskId.replace("job-", ""), 10);
          if (!isNaN(numId)) jobIds.push(numId);
        }
      }
      if (jobIds.length === 0) return;

      try {
        const results = await Promise.all(jobIds.map(id => ETMAgentAPI.getTrainStatusByJobId(id)));
        setProjects(prev => prev.map(p => {
          if (p.taskId) {
            const numId = parseInt(p.taskId.replace("job-", ""), 10);
            const result = results.find((r, i) => jobIds[i] === numId);
            if (result) {
              const newStatus = result.status === "succeeded" ? "completed"
                : result.status === "failed" ? "error"
                : result.status === "running" ? "running"
                : p.pipelineStatus;
              return { ...p, pipelineStatus: newStatus };
            }
          }
          return p;
        }));
      } catch (err) {
        console.error("[Polling] Error:", err);
      }
    };

    pollingTimerRef.current = window.setInterval(pollTrainingStatus, 10000);
    return () => { if (pollingTimerRef.current) window.clearInterval(pollingTimerRef.current); };
  }, [projects]);

  // 定期同步项目列表已移除，避免打断上传操作
  // 需要刷新请手动点击刷新按钮
  syncTimerRef.current = null;

  // 刷新项目列表（与 load 相同逻辑，保留正在运行的项目）
  const refreshProjects = useCallback(async () => {
    setIsLoading(true)
    try {
      console.log("[Dashboard] Refreshing projects...");
      const [dbProjects, datasets, tasks, ossInfo] = await Promise.all([
        ETMAgentAPI.getProjects(),
        ETMAgentAPI.getDatasets(),
        ETMAgentAPI.getTasks({ limit: 100 }).catch(() => []),
        ETMAgentAPI.listOssDatasets().catch(() => ({ datasets: [] as { name: string; chart_count: number }[] })),
      ])

      // 构建有结果的 dataset 名称集合（OSS 上有图表文件即为有结果）
      const datasetsWithResults = new Set(ossInfo.datasets.map((d: { name: string }) => d.name))
      const seen = new Set<string>()
      const list: WorkspaceProject[] = []

      // 预构建 dataset→task 映射：优先已完成的任务，也包含失败的任务
      const taskByDataset = new Map<string, { task_id: string; status: string; pipeline_status?: string; error_message?: string }>()
      for (const t of tasks) {
        const ds = t.dataset || (t as any).dataset_name
        if (!ds) continue
        const existing = taskByDataset.get(ds)
        if (!existing || (t.status === "completed" && existing.status !== "completed")) {
          taskByDataset.set(ds, { task_id: t.task_id, status: t.status, pipeline_status: t.status === "completed" ? "completed" : t.status === "failed" ? "error" : "running" })
        }
      }

      for (const p of dbProjects) {
        const key = p.dataset_name || `db-${p.id}`
        seen.add(key)
        let effectiveTaskId = p.task_id ?? null
        let effectivePipelineStatus = p.pipeline_status
        if (!effectiveTaskId && p.dataset_name && p.pipeline_status !== 'draft') {
          const matched = taskByDataset.get(p.dataset_name)
          if (matched) {
            effectiveTaskId = matched.task_id
            effectivePipelineStatus = effectivePipelineStatus || matched.pipeline_status
          }
        }
        // 如果数据库已标记完成，即使 OSS API 返回空也认为有结果（解决未登录时 OSS API 401 问题）
        const hasResults = p.dataset_name ? (datasetsWithResults.has(p.dataset_name) || p.pipeline_status === "completed") : false
        const derivedPipelineStatus = effectivePipelineStatus === "draft" ? "draft" : effectivePipelineStatus === "completed" ? "completed"
          : effectivePipelineStatus === "error" ? "error"
          : effectivePipelineStatus === "running" ? "running"
          : effectiveTaskId ? "running"
          : p.dataset_name && datasetsWithResults.has(p.dataset_name) ? "completed"
          : p.dataset_name ? "draft"
          : undefined
        list.push({
          id: `proj-db-${p.id}`,
          name: p.name,
          rows: 0,
          createdAt: p.created_at ? "已保存" : "刚刚",
          status: derivedPipelineStatus === "completed" ? "completed"
            : derivedPipelineStatus === "error" ? "no_result"
            : derivedPipelineStatus === "running" ? "vectorizing"
            : p.dataset_name ? "draft"
            : "draft" as const,
          datasetName: p.dataset_name ?? undefined,
          mode: (p.mode as any) ?? "zero_shot",
          models: ["theta"],
          numTopics: p.num_topics ?? 20,
          pipelineStatus: derivedPipelineStatus as any,
          hasResults,
          dbProjectId: p.id,
          taskId: effectiveTaskId,
        })
      }
      for (const ds of datasets) {
        if (seen.has(ds.name)) continue
        seen.add(ds.name)
        const hasResults = datasetsWithResults.has(ds.name)
        list.push({
          id: `proj-${ds.name}`,
          name: ds.name,
          rows: ds.size ?? (ds as any).file_count ?? 0,
          createdAt: "已上传",
          status: hasResults ? "completed" as const : "draft" as const,
          pipelineStatus: hasResults ? "completed" : "draft",
          hasResults,
          datasetName: ds.name,
          models: ["theta"],
        })
      }
      for (const t of tasks) {
        const ds = t.dataset || (t as any).dataset_name
        if (ds && !seen.has(ds)) {
          seen.add(ds)
          const hasResults = datasetsWithResults.has(ds)
          list.push({
            id: `proj-${ds}`,
            name: ds,
            rows: 0,
            createdAt: "已分析",
            status: hasResults ? "completed" as const : (t.status === "completed" ? "no_result" as const : "vectorizing" as const),
            pipelineStatus: t.status === "completed" ? "completed" : t.status === "failed" ? "error" : "running",
            hasResults,
            datasetName: ds,
            models: ["theta"],
            taskId: t.task_id || null,
          })
        }
      }
      setProjects(prev => {
        const normalizedList = reconcileProjectIdentity(list, prev)

        // 构建 dbProjectId → 旧项目 ID 的映射，用于迁移 temp ID
        const oldIdByDbId = new Map<number, string>()
        for (const p of prev) {
          if (p.dbProjectId) oldIdByDbId.set(p.dbProjectId, p.id)
        }

        // 迁移：如果旧列表中有 temp ID（如 new-xxx）指向同一个 dbProjectId，迁移 tab
        for (const np of normalizedList) {
          if (np.dbProjectId && oldIdByDbId.has(np.dbProjectId)) {
            const oldId = oldIdByDbId.get(np.dbProjectId)!
            if (oldId !== np.id) {
              setTabs(t => t.map(tab => tab.id === oldId ? { ...tab, id: np.id } : tab))
              setActiveTabId(a => a === oldId ? np.id : a)
            }
          }
        }

        // 保留正在运行的项目，但用新 ID 替换旧 temp ID
        const runningProjects = prev
          .filter(p => p.pipelineStatus === "running")
          .map(rp => {
            if (rp.dbProjectId) {
              const newVersion = normalizedList.find(np => np.dbProjectId === rp.dbProjectId)
              if (newVersion) return { ...rp, id: newVersion.id }
            }
            return rp
          })
        const runningIds = new Set(runningProjects.map(p => p.id))
        const runningDbIds = new Set(runningProjects.filter(p => p.dbProjectId).map(p => p.dbProjectId))
        const newProjects = normalizedList.filter(np =>
          !runningIds.has(np.id) && !(np.dbProjectId && runningDbIds.has(np.dbProjectId))
        )
        return [...runningProjects, ...newProjects]
      })
    } catch (error) {
      console.error("Failed to refresh projects:", error)
    } finally {
      setIsLoading(false)
    }
  }, [])

  // 同步 refreshProjects 到 ref，确保定时器能访问最新版本
  refreshProjectsRef.current = refreshProjects

  const handleOpenProject = (projectId: string) => {
    const existingTab = tabs.find((tab) => tab.id === projectId)
    if (existingTab) {
      setActiveTabId(projectId)
    } else {
      const project = projects.find(p => p.id === projectId)
      const projectName = project?.name || "分析项目"
      const newTab: Tab = {
        id: projectId,
        title: projectName,
        closable: true,
      }
      setTabs([...tabs, newTab])
      setActiveTabId(projectId)
    }
  }

  // 创建成功后才打开工作台，后续始终使用服务端项目 ID。
  const handleCreateProject = useCallback(async (data: NewProjectData) => {
    const created = await ETMAgentAPI.createProject({ name: data.name, mode: "zero_shot", num_topics: 20 })
    const projectId = `proj-db-${created.id}`
    const project: WorkspaceProject = {
      id: projectId,
      dbProjectId: created.id,
      name: created.name,
      datasetName: created.dataset_name ?? `project-${created.id}`,
      mode: "zero_shot",
      models: ["theta"],
      numTopics: created.num_topics,
      rows: 0,
      createdAt: "刚刚",
      status: "draft",
      pipelineStatus: undefined,
    }

    setProjects(prev => [project, ...prev])
    setTabs(prev => [...prev, { id: projectId, title: created.name, closable: true }])
    setActiveTabId(projectId)
    setProjectTransitionName(data.name)

    if (transitionTimerRef.current) {
      window.clearTimeout(transitionTimerRef.current)
    }
    transitionTimerRef.current = window.setTimeout(() => {
      setProjectTransitionName(null)
      transitionTimerRef.current = null
    }, 900)

  }, [])

  // Pipeline 完成回调：更新本地状态，并同步到数据库（若有 dbProjectId）
  const handlePipelineComplete = useCallback(async (
    projectId: string,
    result?: { dataset?: string; taskId?: string } | null,
    dbProjectId?: number,
  ) => {
    const updates = {
      status: "completed" as const,
      pipelineStatus: "completed" as const,
      ...(result?.dataset && { datasetName: result.dataset }),
    }
    setProjects(prev => prev.map(p => (p.id === projectId ? { ...p, ...updates } : p)))

    if (dbProjectId && result) {
      try {
        await ETMAgentAPI.updateProject(dbProjectId, {
          dataset_name: result.dataset,
          status: "completed",
          pipeline_status: "completed",
          task_id: result.taskId,
        })
      } catch {
        // 忽略同步失败
      }
    }
    // 刷新项目列表，确保 UI 立即从 AutoPipeline 切换到 ProjectResultView
    refreshProjects()
    // 强制重新渲染 renderContent，切换到结果页面
    setRenderKey(k => k + 1)
  }, [refreshProjects])

  const handlePipelineError = useCallback((projectId: string) => {
    setProjects(prev => prev.map(p =>
      p.id === projectId ? { ...p, status: "no_result" as const, pipelineStatus: "error" } : p
    ))
  }, [])

  // Persist tabs to sessionStorage
  useEffect(() => {
    sessionStorage.setItem(STORAGE_KEYS.TABS, JSON.stringify(tabs))
  }, [tabs])

  // Persist activeTabId to sessionStorage
  useEffect(() => {
    sessionStorage.setItem(STORAGE_KEYS.ACTIVE_TAB, activeTabId)
  }, [activeTabId])

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

  // 删除项目：删除 OSS 数据集目录
  const handleDeleteProject = useCallback(async (projectId: string) => {
    const project = projects.find((p) => p.id === projectId)
    if (!project) return

    const datasetName = project.datasetName || (projectId.startsWith("proj-") && !projectId.startsWith("proj-db-") ? projectId.replace(/^proj-/, "") : null)
    if (datasetName) {
      try {
        await ETMAgentAPI.deleteDataset(datasetName)
      } catch (error) {
        const msg = error instanceof Error ? error.message : String(error)
        if (!msg.includes("404") && !msg.includes("not found")) {
          console.error("删除数据集失败:", error)
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

  // 批量删除：并发执行，部分失败不阻断其他项目
  const handleBatchDelete = useCallback(async (projectIds: string[]) => {
    const results = await Promise.allSettled(
      projectIds.map(async (projectId) => {
        const project = projects.find((p) => p.id === projectId)
        if (!project) return
        const datasetName = project.datasetName || (projectId.startsWith("proj-") && !projectId.startsWith("proj-db-") ? projectId.replace(/^proj-/, "") : null)
        if (datasetName) {
          try {
            await ETMAgentAPI.deleteDataset(datasetName)
          } catch (error) {
            const msg = error instanceof Error ? error.message : String(error)
            if (!msg.includes("404") && !msg.includes("not found")) throw error
          }
        }
      })
    )
    const deletedIds = new Set(
      projectIds.filter((_, i) => results[i].status === "fulfilled")
    )
    if (deletedIds.size === 0) return
    setProjects((prev) => prev.filter((p) => !deletedIds.has(p.id)))
    setTabs((prev) => prev.filter((t) => !deletedIds.has(t.id)))
    if (deletedIds.has(activeTabId)) setActiveTabId("hub")
  }, [projects, activeTabId])

  const currentProject = projects.find(project => project.id === activeTabId)
  const shouldShowConfig = !!currentProject && (
    currentProject.pipelineStatus === "draft" || currentProject.pipelineStatus === "running" ||
    currentProject.pipelineStatus === "error" || !currentProject.hasResults && (
      currentProject.status === "draft" || currentProject.pipelineStatus === undefined
    )
  )
  const showingResults = !isLoading && !!currentProject && (!shouldShowConfig || resultProjectId === currentProject.id)

  const renderContent = () => {
    if (activeTabId === "hub") {
      return (
        <div className="min-h-full">
          <ProjectHub
            onProjectSelect={handleOpenProject}
            onNewProject={() => setIsNewProjectDialogOpen(true)}
            onDeleteProject={handleDeleteProject}
            onBatchDelete={handleBatchDelete}
            onRefresh={refreshProjects}
            projects={projects}
            isLoading={isLoading}
          />
        </div>
      )
    }

    // 查找当前项目
    const currentProject = projects.find(p => p.id === activeTabId)

    if (isLoading) {
      return (
        <div className="p-8 text-center">
          <div className="flex items-center justify-center mb-4">
            <Spinner className="h-8 w-8 text-slate-600" />
          </div>
          <h2 className="text-xl font-semibold text-slate-900 mb-2">正在加载</h2>
          <p className="text-slate-500">正在获取项目数据，请稍候...</p>
        </div>
      )
    }

    // 如果项目找不到，自动切回项目中心
    if (!currentProject) {
      setActiveTabId("hub")
      return null
    }

    if (shouldShowConfig && resultProjectId !== currentProject.id) {
      return (
        <AutoPipeline
          key={currentProject.id}
          projectKey={currentProject.id}
          datasetName={currentProject.datasetName}
          projectName={currentProject.name}
          mode={currentProject.mode || "zero_shot"}
          numTopics={currentProject.numTopics || 20}
          initialTaskId={currentProject.taskId}
          pipelineStatus={currentProject.pipelineStatus}
          onViewResults={currentProject.hasResults ? () => setResultProjectId(currentProject.id) : undefined}
          onDlcStarted={() => {
            // DLC 训练开始，自动返回项目中心
            setActiveTabId("hub")
          }}
          onConfigConfirmed={async (config) => {
            const configuredTopics = Number(config.parameters[config.models[0]]?.num_topics ?? config.parameters[config.models[0]]?.max_topics ?? currentProject.numTopics ?? 20)
            setProjects(prev => prev.map(p =>
              p.id === currentProject.id
                ? {
                    ...p,
                    mode: config.mode,
                    numTopics: configuredTopics,
                    models: config.models.length > 0 ? config.models : ["theta"],
                  }
                : p
            ))
            if (currentProject.dbProjectId) {
              try {
                await ETMAgentAPI.updateProject(currentProject.dbProjectId, {
                  mode: config.mode,
                  num_topics: configuredTopics,
                })
              } catch { /* skip */ }
            }
          }}
          onComplete={(result) => handlePipelineComplete(currentProject.id, result, currentProject.dbProjectId)}
          onError={() => handlePipelineError(currentProject.id)}
          onTaskCreated={async (tid) => {
            setProjects(prev => prev.map(p =>
              p.id === currentProject.id
                ? { ...p, taskId: tid, pipelineStatus: "running" }
                : p
            ))
            if (currentProject.dbProjectId) {
              try {
                await ETMAgentAPI.updateProject(currentProject.dbProjectId, {
                  task_id: tid,
                  pipeline_status: "running",
                })
              } catch { /* skip */ }
            }
          }}
          onUploadComplete={async (datasetName) => {
            setProjects(prev => prev.map(p => p.id === currentProject.id ? { ...p, datasetName, status: "draft", pipelineStatus: "draft" } : p))
            if (currentProject.dbProjectId) {
              try {
                await ETMAgentAPI.updateProject(currentProject.dbProjectId, {
                  dataset_name: datasetName,
                  status: "draft",
                  pipeline_status: "draft",
                })
                setProjects(prev => prev.map(p =>
                  p.id === currentProject.id ? { ...p, datasetName } : p
                ))
              } catch { /* skip */ }
            }
          }}
        />
      )
    }

    // 已完成的项目显示结果概览
    return (
      <ResultWorkspace source={{ kind: "dataset", project: currentProject }} onBack={() => { if (shouldShowConfig) setResultProjectId(null); else setActiveTabId("hub") }} onNewAnalysis={shouldShowConfig ? undefined : async () => {
        if (currentProject.dbProjectId) await ETMAgentAPI.updateProject(currentProject.dbProjectId, { task_id: null, status: 'draft', pipeline_status: 'draft' })
        setProjects(prev => prev.map(project => project.id === currentProject.id ? { ...project, taskId: null, status: 'draft', pipelineStatus: 'draft' } : project))
        setResultProjectId(null)
      }} />
    )
  }

  if (showingResults) return renderContent()

  return (
    <>
      <AppShell
        embedded
        tabs={tabs}
        activeTabId={activeTabId}
        onTabChange={handleTabChange}
        onTabClose={handleTabClose}
        onTabsReorder={(fromIdx, toIdx) => {
          const newTabs = [...tabs]
          const [moved] = newTabs.splice(fromIdx, 1)
          newTabs.splice(toIdx, 0, moved)
          setTabs(newTabs)
        }}
        assistant={showingResults ? undefined : <ProjectAssistant key={activeTabId} scopeKey={`manual:${activeTabId}`} context={{ current_project: currentProject, current_view_name: currentProject?.name ?? "项目中心" }} />}
      >
        <div className={showingResults ? "relative h-full min-h-0" : "relative min-h-[360px]"}>
          <div
            key={renderKey}
            className={`${showingResults ? "h-full min-h-0" : ""} transition-all duration-500 ${projectTransitionName ? "opacity-70 scale-[0.995]" : "opacity-100 scale-100"}`}
          >
            {renderContent()}
          </div>

          {projectTransitionName && (
            <div className="absolute inset-0 z-20 flex items-center justify-center bg-white/70 backdrop-blur-[1px]">
              <div className="rounded-xl border border-slate-200 bg-white px-5 py-4 shadow-sm">
                <div className="flex items-center gap-3">
                  <div className="h-5 w-5 rounded-full border-2 border-blue-600 border-t-transparent animate-spin" />
                  <div>
                    <p className="text-sm font-medium text-slate-800">正在创建项目</p>
                    <p className="text-xs text-slate-500">{projectTransitionName} 初始化中，请稍候...</p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </AppShell>

      <NewProjectDialog
        open={isNewProjectDialogOpen}
        onOpenChange={setIsNewProjectDialogOpen}
        onSubmit={handleCreateProject}
      />
    </>
  )
}
