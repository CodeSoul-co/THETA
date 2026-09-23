"use client"

import { useState, useEffect } from "react"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Checkbox } from "@/components/ui/checkbox"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Loader2 } from "lucide-react"
import { ETMAgentAPI } from "@/lib/api/etm-agent"
import { useProjectDraft } from "@/lib/use-project-draft"

export interface ColumnSelection {
  textColumn: string
  metaColumns: string[]
  timeColumn?: string
  labelColumn?: string
}

interface ColumnSelectPanelProps {
  projectKey: string
  open: boolean
  onOpenChange: (open: boolean) => void
  onConfirm: (selection: ColumnSelection) => void
  onSkip?: () => void
  datasetName: string
  /** 本地上传返回的文件 ID，确保预览用户当前选择的文件。 */
  jobId?: string | null
}

export function ColumnSelectPanel({
  open,
  onOpenChange,
  onConfirm,
  onSkip,
  datasetName,
  jobId,
  projectKey,
}: ColumnSelectPanelProps) {
  const [columns, setColumns] = useState<string[]>([])
  const [rows, setRows] = useState<string[][]>([])
  const [loading, setLoading] = useState(false)
  const [selection, setSelection] = useProjectDraft<ColumnSelection>(`${projectKey}:columns:${jobId}`, { textColumn: '', metaColumns: [] })
  const { textColumn, metaColumns, timeColumn = '', labelColumn = '' } = selection
  const setTextColumn = (textColumn: string) => setSelection(prev => ({ ...prev, textColumn, metaColumns: prev.metaColumns.filter(c => c !== textColumn), timeColumn: prev.timeColumn === textColumn ? undefined : prev.timeColumn, labelColumn: prev.labelColumn === textColumn ? undefined : prev.labelColumn }))
  const setTimeColumn = (timeColumn: string) => setSelection(prev => ({ ...prev, timeColumn }))
  const setLabelColumn = (labelColumn: string) => setSelection(prev => ({ ...prev, labelColumn }))
  const [error, setError] = useState('')

  useEffect(() => {
    if (!open || !datasetName) return
    let cancelled = false
    setLoading(true)
    setError('')
    ETMAgentAPI.getDatasetPreview(datasetName, jobId ?? undefined)
      .then(({ columns: c, rows: r }) => {
        if (cancelled) return
        setColumns(c)
        setRows(r)
      })
      .catch(error => { if (!cancelled) { setColumns([]); setError(error.message || '数据预览失败，请重试') } })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [open, datasetName, jobId])

  const handleConfirm = () => {
    if (loading || !columns.includes(textColumn)) return
    onConfirm({
      textColumn,
      metaColumns,
      timeColumn: timeColumn || undefined,
      labelColumn: labelColumn || undefined,
    })
    onOpenChange(false)
  }

  const toggleMeta = (col: string) => {
    setSelection(prev => ({ ...prev, metaColumns: prev.metaColumns.includes(col) ? prev.metaColumns.filter(c => c !== col) : [...prev.metaColumns, col] }))
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="flex max-h-[90dvh] min-w-0 flex-col gap-0 overflow-hidden p-0 sm:max-w-3xl">
        <DialogHeader className="shrink-0 border-b px-5 py-5 pr-12 text-left sm:px-6 sm:pr-12">
          <DialogTitle>选择分析数据列</DialogTitle>
          <DialogDescription className="break-words [overflow-wrap:anywhere]">
            选择正文、真实时间与可选元数据；分词自动处理。数据集：{datasetName}
          </DialogDescription>
        </DialogHeader>

        {loading ? (
          <div className="flex items-center justify-center py-12 gap-2">
            <Loader2 className="w-5 h-5 animate-spin" />
            <span>加载预览...</span>
          </div>
        ) : (
          <div className="min-h-0 min-w-0 flex-1 space-y-6 overflow-y-auto px-5 py-5 sm:px-6">
            <div>
              <Label>文本列（必选）</Label>
              <RadioGroup value={textColumn} onValueChange={setTextColumn} className="mt-2 grid min-w-0 grid-cols-2 gap-2 sm:grid-cols-3">
                {columns.map(col => (
                  <div key={col} className="flex min-w-0 items-start gap-2 rounded-lg border p-2.5">
                    <RadioGroupItem value={col} id={`text-${col}`} />
                    <Label htmlFor={`text-${col}`} className="min-w-0 cursor-pointer font-normal leading-5 [overflow-wrap:anywhere]">{col}</Label>
                  </div>
                ))}
              </RadioGroup>
            </div>

            {columns.length > 1 && (
              <div>
                <Label>元数据列（可选，用于 STM 协变量）</Label>
                <div className="mt-2 grid min-w-0 grid-cols-2 gap-2 sm:grid-cols-3">
                  {columns.filter(c => c !== textColumn).map(col => (
                    <div key={col} className="flex min-w-0 items-start gap-2 rounded-lg border p-2.5">
                      <Checkbox
                        id={`meta-${col}`}
                        checked={metaColumns.includes(col)}
                        onCheckedChange={() => toggleMeta(col)}
                      />
                      <Label htmlFor={`meta-${col}`} className="min-w-0 cursor-pointer font-normal leading-5 [overflow-wrap:anywhere]">{col}</Label>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="grid gap-4 sm:grid-cols-2">
              <div className="min-w-0 space-y-2">
                <Label htmlFor="time-column">时间列（DTM 必选）</Label>
                <select id="time-column" className="w-full min-w-0 max-w-full rounded-lg border bg-white p-2 text-sm" value={timeColumn} onChange={e => setTimeColumn(e.target.value)}>
                  <option value="">不使用时间列</option>
                  {columns.filter(c => c !== textColumn).map(c => <option key={c} value={c}>{c}</option>)}
                </select>
              </div>
              <div className="min-w-0 space-y-2">
                <Label htmlFor="label-column">标签列（有监督嵌入必选）</Label>
                <select id="label-column" className="w-full min-w-0 max-w-full rounded-lg border bg-white p-2 text-sm" value={labelColumn} onChange={e => setLabelColumn(e.target.value)}>
                  <option value="">不使用标签列</option>
                  {columns.filter(c => c !== textColumn).map(c => <option key={c} value={c}>{c}</option>)}
                </select>
              </div>
            </div>

            {rows.length > 0 && (
              <div className="min-w-0">
                <Label>前 5 行预览</Label>
                <p className="mt-1 text-xs text-slate-500">表格可横向滚动查看全部列；长文本可展开查看完整内容。</p>
                <div role="region" aria-label="数据预览表格，可横向滚动" tabIndex={0} className="mt-2 max-h-60 w-full min-w-0 overflow-auto rounded-lg border">
                  <table className="w-max min-w-full table-fixed text-xs">
                    <thead>
                      <tr className="border-b">
                        {columns.map(c => (
                          <th key={c} className="text-left p-2 font-medium">{c}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((row, i) => (
                        <tr key={i} className="border-b last:border-0">
                          {row.map((cell, j) => (
                            <td key={j} className="w-52 min-w-40 max-w-64 p-2 align-top">
                              {String(cell).length > 100 ? <details className="w-48 [overflow-wrap:anywhere]"><summary className="cursor-pointer leading-5">{String(cell).slice(0, 80)}…<span className="block text-blue-700">展开全文</span></summary><div className="mt-2 max-h-48 overflow-y-auto whitespace-pre-wrap leading-5">{String(cell)}</div></details> : <div className="w-48 whitespace-pre-wrap leading-5 [overflow-wrap:anywhere]">{String(cell)}</div>}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            <p className="rounded-xl bg-slate-50 p-4 text-sm leading-6 text-slate-600">分词与停用词由训练流程自动处理，支持混合语言。下一步可上传自定义停用词表或导出内置词表；绘图语言与文本处理独立。</p>
          </div>
        )}

        <DialogFooter className="shrink-0 border-t bg-slate-50 px-5 py-4 sm:px-6">
          {error && <p role="alert" className="text-sm text-red-600">{error}</p>}
          <Button variant="outline" onClick={() => onOpenChange(false)}>稍后继续</Button>
          <Button onClick={handleConfirm} disabled={loading || !columns.includes(textColumn)} className="bg-blue-600">
            确认并继续
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
