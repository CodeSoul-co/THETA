"use client"
import { useEffect, useState } from 'react'
import { apiFetch } from '@/lib/api/config'
import { analysisReportEndpoint } from '@/lib/analysis-report'
import { resultDatasetName, type ResultSource } from './result-source'

type Report = { status: 'idle' | 'running' | 'complete' | 'failed'; phase: string; error?: string; markdown?: boolean; pdf?: boolean; bundle?: boolean }
export function AnalysisReport({ source, jobId }: { source: ResultSource; jobId: string }) {
  const endpoint = analysisReportEndpoint(source.kind, source.kind === 'run' ? source.runId : resultDatasetName(source)!, jobId)
  const [loaded, setLoaded] = useState<{ endpoint: string; state: Report }>()
  const [error, setError] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [question, setQuestion] = useState('')
  const [refresh, setRefresh] = useState(0)
  const report = loaded?.endpoint === endpoint ? loaded.state : undefined
  useEffect(() => {
    let stopped = false, timer: ReturnType<typeof setTimeout>
    setError('')
    const read = async () => {
      try {
        const response = await apiFetch<Report | { data: Report }>('', endpoint)
        if (stopped) return
        const state = 'data' in response ? response.data : response
        setLoaded({ endpoint, state })
        if (state.status === 'running') timer = setTimeout(read, 2500)
      } catch (e) { if (!stopped) setError(e instanceof Error ? e.message : String(e)) }
    }
    void read()
    return () => { stopped = true; clearTimeout(timer) }
  }, [endpoint, submitting, refresh])
  useEffect(() => { setQuestion('') }, [endpoint])
  const generate = async (regenerate = false) => {
    setSubmitting(true); setError('')
    try {
      const response = await apiFetch<Report | { data: Report }>('', endpoint, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ researchQuestion: question, regenerate }) })
      setLoaded({ endpoint, state: 'data' in response ? response.data : response })
    } catch (e) { setError(e instanceof Error ? e.message : String(e)) }
    finally { setSubmitting(false) }
  }
  const download = (format: string) => `${endpoint}${endpoint.includes('?') ? '&' : '?'}format=${format}`
  return <section className="mt-4 space-y-3 rounded-2xl border border-indigo-100 bg-indigo-50/40 p-5">
    <h3 className="font-semibold text-slate-900">完整分析报告</h3>
    <p className="text-sm leading-6 text-slate-600">按数据描述性统计、数据分析、任务分析、建模描述、结论分析生成报告，涵盖当前模型的全量与各划分结果。使用“设置 → 模型 API”中配置的服务，并向该服务发送统计摘要、脱敏文本摘录和结果证据。正文引用并插入当前任务的原生图表，区分各数据集范围。仅在点击后生成，不会重新训练。</p>
    {report && !report.markdown && ['idle', 'failed'].includes(report.status) && <label className="block space-y-2 text-sm text-slate-700">
      <span>研究问题（可选）</span>
      <textarea key={endpoint} maxLength={6000} rows={3} value={question} onChange={e => setQuestion(e.target.value)} placeholder="希望通过这次数据分析回答什么问题？未填写时，依据当前任务目标组织报告。" className="w-full rounded-xl border border-slate-200 bg-white p-3" />
      <span className="block text-xs text-slate-500">以连贯的学术段落展开分析，目标约 5000–8000 字；篇幅根据实际证据调整。</span>
    </label>}
    {report && report.status !== 'idle' && <p role="status" className="text-sm text-indigo-800">{report.phase}{report.status === 'running' ? '，可切换页面，后台继续处理。' : ''}</p>}
    {(error || report?.error) && <p role="alert" className="text-sm text-red-700">{error || report?.error}</p>}
    <div className="flex flex-wrap gap-3">
      {!report && error && <button type="button" onClick={() => setRefresh(value => value + 1)} className="rounded-xl border bg-white px-4 py-2 text-sm">重新读取报告状态</button>}
      {report?.status !== 'complete' && !report?.pdf && <button type="button" disabled={!report || submitting || report.status === 'running'} onClick={() => void generate()} className="rounded-xl bg-indigo-600 px-4 py-2 text-sm text-white disabled:opacity-50">{submitting || report?.status === 'running' ? '正在生成…' : report?.status === 'failed' ? report.markdown ? '重试 PDF 导出' : '重试生成报告' : '生成完整分析报告'}</button>}
      {report?.markdown && <a href={download('md')} download className="rounded-xl border bg-white px-4 py-2 text-sm">下载 Markdown</a>}
      {report?.pdf && <a href={download('pdf')} download className="rounded-xl border bg-white px-4 py-2 text-sm">下载 PDF</a>}
      {report?.bundle && <a href={download('zip')} download className="rounded-xl border bg-white px-4 py-2 text-sm">下载 Markdown 图文包</a>}
      {report?.markdown && report.status !== 'running' && <button type="button" disabled={submitting} onClick={() => void generate(true)} className="rounded-xl border bg-white px-4 py-2 text-sm disabled:opacity-50">重新生成（调用模型 API）</button>}
    </div>
    {report?.markdown && <p className="text-xs text-slate-500">{report.bundle ? 'PDF 包含完整图文；编辑 Markdown 时，请下载图文包并保留 assets 文件夹。' : '这是此前生成的报告；点击重新生成，可应用新的图文分析与排版。'}重新生成会更新报告，生成正文失败时仍保留原报告。</p>}
  </section>
}
