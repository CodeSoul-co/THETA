"use client"

import dynamic from 'next/dynamic'
import { useRef, useState } from 'react'
import { ArrowLeft } from 'lucide-react'
import { toast } from 'sonner'
import { importManualResults } from '@/components/theta-workbench/api/client'
import { ProjectAssistant } from '@/components/chat/project-assistant'
import { resultScope, resultProjectName, resultDatasetName, type ResultSource, type ResultDestination } from './result-source'
export type { ResultSource, ResultDestination } from './result-source'

const ResearchResults = dynamic(() => import('./research-result-view').then(module => module.ResearchResultView))

/** Final convergence: one result page, one assistant, regardless of training workflow. */
export function ResultWorkspace({ source, initialDestination, onBack, onNewAnalysis }: {
  source: ResultSource
  initialDestination?: ResultDestination
  onBack: () => void
  onNewAnalysis?: () => void
}) {
  const [openRequest, setOpenRequest] = useState(0)
  const [selection, setSelection] = useState<{ model: string; jobId: string } | null>(null)
  const [copying, setCopying] = useState(false)
  const copyPending = useRef(false)
  const continueResearch = async () => {
    if (copyPending.current) return
    copyPending.current = true
    setCopying(true)
    try {
      if (source.kind === 'run') {
        window.dispatchEvent(new CustomEvent('theta:open-conversation', { detail: { runId: source.runId, projectId: source.projectId } }))
      } else {
        const projectId = source.project.dbProjectId ?? Number(source.project.id?.replace(/^proj-db-/, ''))
        const imported = await importManualResults({ ...(Number.isSafeInteger(projectId) && projectId > 0 ? { projectId } : { datasetName: resultDatasetName(source) }), selectedJobId: selection?.jobId })
        window.dispatchEvent(new CustomEvent('theta:open-conversation', { detail: { runId: imported.runId, projectId: imported.project.id, project: imported.project } }))
        toast.success(imported.reused ? '已打开这批结果的对话副本' : `已复制 ${imported.copiedJobs} 个模型结果，可继续对话研究`)
      }
    } catch (error) { toast.error(error instanceof Error ? error.message : String(error)) }
    finally { copyPending.current = false; setCopying(false) }
  }
  const scope = resultScope(source)
  const assistantScope = source.kind === 'dataset' ? `manual:${source.project.id ?? scope}` : `results:${scope}`
  return <section className="flex h-full min-h-0 min-w-0 flex-1 overflow-hidden bg-slate-50" aria-label="研究结果">
    <div className="flex min-h-0 min-w-0 flex-1 flex-col">
      <div className="flex shrink-0 flex-wrap items-center gap-3 border-b border-slate-200 bg-white px-5 py-3">
        <button type="button" onClick={onBack} className="inline-flex items-center gap-2 rounded-lg px-2 py-1.5 text-sm text-slate-600 hover:bg-slate-100"><ArrowLeft size={16} />返回工作台</button>
        <span className="text-sm font-medium text-slate-900">研究结果</span>
        {onNewAnalysis && <button type="button" onClick={onNewAnalysis} className="ml-auto rounded-lg border border-slate-200 px-3 py-2 text-sm text-indigo-700 hover:bg-indigo-50">更换数据 / 追加分析</button>}
      </div>
      <div className="min-h-0 min-w-0 flex-1 overflow-auto">
        <ResearchResults key={scope} source={source} initialDestination={initialDestination} onSelection={setSelection} onOpenAssistant={() => setOpenRequest(n => n + 1)} onContinueResearch={() => void continueResearch()} continuingResearch={copying} />
      </div>
    </div>
    <ProjectAssistant key={assistantScope} scopeKey={assistantScope} openRequest={openRequest} context={{ project_name: resultProjectName(source), dataset: resultDatasetName(source), result_scope: scope, selected_result: selection }} />
  </section>
}
