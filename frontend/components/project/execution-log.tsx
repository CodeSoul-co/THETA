"use client"

import { useEffect, useRef, useState } from 'react'
import { statusLabel, systemText } from '@/lib/presentation'
import { trainingElapsed, trainingEventText, type TrainingWorkerState } from '@/lib/training-progress'

export function ExecutionLog({ states = [], workers = [], logs, running }: {
  states?: TrainingWorkerState[]; workers?: { id: string; model: string }[]; logs: string[]; running: boolean
}) {
  const [follow, setFollow] = useState(true)
  const scroll = useRef<HTMLDivElement>(null)
  const modelName = (id: string) => workers.find(worker => worker.id === id)?.model.toUpperCase() ?? '模型'
  const entries = states.flatMap(state => [
    ...(state.phaseHistory ?? []).map((phase, index) => ({ id: `${state.id}-phase-${index}`, at: phase.at, text: `${modelName(state.id)} · ${phase.phase === 'uploading' ? '整理模型与图表产物' : systemText(phase.phase)}` })),
    ...(state.telemetry?.events ?? []).filter(event => event.kind !== 'command').map(event => ({ id: `${state.id}-${event.id}`, at: event.at, text: `${modelName(state.id)} · ${trainingEventText(event)}` })),
  ]).sort((a, b) => (a.at ?? 0) - (b.at ?? 0)).slice(-500)
  const lastEntry = entries.at(-1)?.id
  useEffect(() => { if (follow && scroll.current) scroll.current.scrollTop = scroll.current.scrollHeight }, [follow, lastEntry, logs.length])
  return <details className="rounded-xl border bg-white p-4" open>
    <summary className="cursor-pointer text-sm font-medium">执行日志{running ? ' · 实时更新' : ''}</summary>
    <div className="mt-3 space-y-3">
      {states.map(state => <div key={state.id} className="rounded-lg bg-slate-50 p-3 text-sm">
        <p className="font-medium">{modelName(state.id)} · {statusLabel(state.status)} · {systemText(state.phase)}</p>
        {state.computeDevice && <p className={`mt-1 text-xs ${state.gpuFallback ? 'font-medium text-amber-700' : 'text-slate-600'}`}>{state.gpuFallback ? 'GPU 计算失败，已回退到 CPU 重新执行；耗时可能增加' : state.computeDevice.startsWith('cuda:') ? `NVIDIA GPU · ${state.computeDevice}` : 'CPU 计算'}</p>}
        {state.telemetry?.detail && <p className="mt-1 font-mono text-blue-700">{trainingEventText(state.telemetry.detail)}</p>}
        <p className="mt-1 text-xs text-slate-500">{state.telemetry?.elapsedSeconds != null && `已用时 ${trainingElapsed(state.telemetry.elapsedSeconds)}`}
          {running && state.status === 'running' && state.telemetry?.lastLogAgeSeconds != null && ` · ${Math.floor(state.telemetry.lastLogAgeSeconds)} 秒前收到训练输出`}</p>
        {state.status === 'running' && state.telemetry?.health === 'unresponsive' && <p role="status" className="mt-1 text-xs text-amber-700">超过 30 秒未收到计算进程心跳，当前状态待确认。</p>}
      </div>)}
      <div className="flex items-center justify-between gap-3 text-xs text-slate-500"><p>约每 2.5 秒同步实际输出；展示近期记录，完整训练日志保存在任务目录。</p><label className="flex shrink-0 items-center gap-1"><input type="checkbox" checked={follow} onChange={event => setFollow(event.target.checked)} />跟随最新</label></div>
      <div ref={scroll} className="max-h-80 overflow-auto rounded-lg bg-slate-950 p-4 font-mono text-xs leading-6 text-slate-200" tabIndex={0} aria-label="训练执行日志">
        {logs.map((line, index) => <p key={`local-${index}`} className="whitespace-pre-wrap break-words">{line}</p>)}
        {entries.map(entry => <p key={entry.id} className="whitespace-pre-wrap break-words">{entry.at != null ? `[${new Date(entry.at * 1000).toLocaleTimeString('zh-CN')}] ` : ''}{entry.text}</p>)}
        {!entries.length && <p className="text-slate-400">{running ? '等待模型输出轮次、批次或指标…' : logs.length ? '开始训练后会显示模型的实际迭代与指标。' : '暂无执行记录'}</p>}
      </div>
    </div>
  </details>
}
