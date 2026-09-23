import { computationNotice, errorGuidance, openSetup } from '@/lib/workbench-guidance'
import { Clock3 } from 'lucide-react'

export function ComputationNotice({ sizeBytes, models, locale }: { sizeBytes?: number; models?: string[]; locale?: string }) {
  const message = computationNotice(sizeBytes, models, locale)
  return message ? <aside role="status" aria-label={locale === 'en-US' ? 'Computation time notice' : '计算耗时提醒'} className="my-3 flex gap-3 rounded-xl border-2 border-amber-400 bg-amber-50 px-4 py-3 text-amber-950 shadow-sm">
    <Clock3 className="mt-0.5 size-5 shrink-0 text-amber-700" aria-hidden="true" />
    <div><strong className="block text-sm font-semibold">{locale && locale !== 'zh-CN' ? 'This analysis may take a while' : '计算耗时提醒：本次分析可能需要较长时间'}</strong><p className="mt-1 text-sm leading-6">{message}</p></div>
  </aside> : null
}

export function SetupError({ error, locale = 'zh-CN' }: { error: string; locale?: string }) {
  const guidance = errorGuidance(error, locale)
  return <div role="alert" className="text-sm leading-6"><p>{guidance.message}</p>{guidance.settingsTab && <button type="button" className="mt-2 underline underline-offset-4" onClick={() => openSetup(guidance.settingsTab!)}>{locale === 'zh-CN' ? '打开设置' : 'Open settings'}</button>}</div>
}
