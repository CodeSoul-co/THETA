"use client"

import { useEffect, useState } from "react"
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
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { Info } from "lucide-react"
import { cn } from "@/lib/utils"
import { apiFetch, API_BASE } from "@/lib/api/config"
import { ModelParameterFields } from "./model-parameter-fields"
import type { ModelContract } from "@/lib/model-parameters"
import { DEFAULT_ANALYSIS_CONFIG, type AnalysisConfig } from "@/lib/analysis-config"
export type { AnalysisConfig } from "@/lib/analysis-config"
import { useProjectDraft } from "@/lib/use-project-draft"
import { ComputationNotice, SetupError } from '@/components/theta-workbench/panels/WorkbenchNotice'
import styles from './analysis-config-panel.module.css'

// ==================== 模型配置 ====================

const LANGUAGES = [
  { value: "zh", label: "中文" },
  { value: "en", label: "英文" },
] as const

const MODEL_LIST = [
  { id: "theta", name: "THETA", group: "neural" as const },
  { id: "nvdm", name: "NVDM", group: "neural" as const },
  { id: "gsm", name: "GSM", group: "neural" as const },
  { id: "prodlda", name: "ProdLDA", group: "neural" as const },
  { id: "ctm", name: "CTM", group: "neural" as const },
  { id: "etm", name: "ETM", group: "neural" as const },
  { id: "dtm", name: "DTM", group: "neural" as const },
  { id: "bertopic", name: "BERTopic", group: "neural" as const },
  { id: "lda", name: "LDA", group: "traditional" as const },
  { id: "hdp", name: "HDP", group: "traditional" as const },
  { id: "stm", name: "STM", group: "traditional" as const },
  { id: "btm", name: "BTM", group: "traditional" as const },
] as const

const MODEL_GROUPS = [
  {
    label: "神经",
    ids: ["theta", "nvdm", "gsm", "prodlda", "ctm", "etm", "dtm", "bertopic"],
  },
  {
    label: "传统",
    ids: ["lda", "hdp", "stm", "btm"],
  },
] as const

const QWEN_SIZES = [
  { value: "0.6B", label: "0.6B（默认）" },
  { value: "4B", label: "4B" },
  { value: "8B", label: "8B" },
] as const

const EMBEDDING_MODES = [
  { value: "zero_shot", label: "零样本（默认）", tip: "不训练嵌入，最快" },
  { value: "unsupervised", label: "无监督", tip: "无监督嵌入" },
  { value: "supervised", label: "有监督", tip: "需额外指定标签列" },
] as const

// ==================== 类型 ====================

export interface AnalysisConfigService {
  runtime(): Promise<{ embedding: { configured: boolean; provider: string; endpoint: string; model: string } }>
  model(id: string): Promise<ModelContract>
  uploadStopwords(file: File): Promise<NonNullable<AnalysisConfig['stopwords']>>
  stopwordsDownloadUrl: string
}
const manualService: AnalysisConfigService = {
  runtime: () => apiFetch(API_BASE, '/api/runtime/config'),
  model: id => apiFetch(API_BASE, `/api/models/${id}`),
  uploadStopwords: file => apiFetch(API_BASE, `/api/stopwords?filename=${encodeURIComponent(file.name)}`, { method: 'POST', headers: { 'Content-Type': 'text/plain; charset=utf-8' }, body: file }),
  stopwordsDownloadUrl: `${API_BASE}/api/stopwords/default`,
}

// ==================== 组件 ====================

interface AnalysisConfigPanelProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  onConfirm: (config: AnalysisConfig) => void | boolean | Promise<void | boolean>
  datasetName: string
  datasetSizeBytes?: number
  error?: string | null
  projectKey: string
  initialConfig?: AnalysisConfig
  service?: AnalysisConfigService
  columns?: string[]
  confirmLabel?: string
  description?: string
}

// Get model display name
function getModelName(modelId: string): string {
  return MODEL_LIST.find(m => m.id === modelId)?.name || modelId
}

export function AnalysisConfigPanel({
  open,
  onOpenChange,
  onConfirm,
  datasetName,
  datasetSizeBytes,
  error,
  projectKey,
  initialConfig = DEFAULT_ANALYSIS_CONFIG,
  service = manualService,
  columns,
  confirmLabel = '开始分析',
  description,
}: AnalysisConfigPanelProps) {
  const [config, setConfig] = useProjectDraft<AnalysisConfig>(`${projectKey}:config`, initialConfig)
  const [submitting, setSubmitting] = useState(false)
  const [submitError, setSubmitError] = useState('')
  const [valid, setValid] = useState(false)
  const [activeTab, setActiveTab] = useProjectDraft<string>(`${projectKey}:parameter-tab`, initialConfig.models[0])
  useEffect(() => { if (!config.models.includes(activeTab)) setActiveTab(config.models[0]) }, [config.models, activeTab])
  const [uploadingStopwords, setUploadingStopwords] = useState(false)
  const [stopwordError, setStopwordError] = useState<string | null>(null)
  const [sharedTopics, setSharedTopics] = useProjectDraft(`${projectKey}:shared-topics`, Number(initialConfig.parameters[initialConfig.models[0]]?.num_topics ?? initialConfig.parameters[initialConfig.models[0]]?.max_topics ?? 20))
  const [embedding, setEmbedding] = useState<{ configured: boolean; provider: string; endpoint: string; model: string } | null>(null)
  useEffect(() => {
    if (!open) return
    let cancelled = false
    service.runtime()
      .then(result => { if (!cancelled) {
        setEmbedding(result.embedding)
        setConfig(prev => ({ ...prev, cloudConfirmed: false }))
      } })
      .catch(() => { if (!cancelled) setEmbedding(null) })
    return () => { cancelled = true }
  }, [open, service])

  const applySharedTopics = () => setConfig(prev => ({ ...prev, parameters: {
    ...prev.parameters,
    ...Object.fromEntries(MODEL_LIST.map(({ id }) => [id, { ...prev.parameters[id], [id === "hdp" ? "max_topics" : "num_topics"]: sharedTopics }]))
  } }))

  const uploadStopwords = async (file: File) => {
    setStopwordError(null)
    if (!/\.txt$/i.test(file.name) || !file.size || file.size > 512 * 1024) {
      setStopwordError("请选择非空的 UTF-8 TXT 词表，大小不超过 512 KB。")
      return
    }
    setUploadingStopwords(true)
    try {
      const stopwords = await service.uploadStopwords(file)
      setConfig(prev => ({ ...prev, stopwords }))
    } catch (error) {
      setStopwordError(error instanceof Error ? error.message : "词表上传失败，请重试。")
    } finally { setUploadingStopwords(false) }
  }

  const handleModelToggle = (modelId: string, checked: boolean) => {
    const models = checked ? [...config.models, modelId] : config.models.filter(id => id !== modelId)
    if (!models.length) return // 至少保留一个模型，避免页签与模型选择脱节。
    setConfig(prev => ({ ...prev, models }))
    if (!models.includes(activeTab)) setActiveTab(models[0])
  }

  const handleConfirm = async () => {
    if (submitting || uploadingStopwords || !valid) return
    if (config.models.includes("theta") && config.embeddingProvider === "cloud" && (!embedding?.configured || !config.cloudConfirmed || config.mode !== "zero_shot")) return
    if (config.models.length === 0) {
      setConfig(prev => ({ ...prev, models: ["theta"] }))
    }
    setSubmitting(true); setSubmitError('')
    try {
      if (await onConfirm({ ...config, cloudSelection: embedding ? { provider: embedding.provider, endpoint: embedding.endpoint, model: embedding.model } : undefined }) !== false) onOpenChange(false)
    } catch (cause) { setSubmitError(cause instanceof Error ? cause.message : String(cause)) }
    finally { setSubmitting(false) }
  }

  return (
    <TooltipProvider>
      <Dialog open={open} onOpenChange={value => { if (!submitting) onOpenChange(value) }}>
        <DialogContent className={styles.dialog}>
          <DialogHeader className={styles.header}>
            <DialogTitle>分析配置</DialogTitle>
            <DialogDescription>
              数据集：{datasetName} · {description ?? '选择模型后，可分别调整每个模型的参数。'}
            </DialogDescription>
            <ComputationNotice sizeBytes={datasetSizeBytes} models={config.models} />
          </DialogHeader>

          {/* Fieldset has an anonymous internal box in browsers; keep scrolling on a normal block. */}
          <div className={styles.body} role="region" aria-label="参数配置内容" tabIndex={0}>
          <fieldset disabled={submitting} className={`${styles.fields} space-y-6`}>
            {columns && <section className="grid gap-4 rounded-xl border bg-slate-50/70 p-4 sm:grid-cols-2">
              {([{ key: 'textColumn', label: '正文列（必选）' }, { key: 'timeColumn', label: '时间列（DTM 必选）' }, { key: 'labelColumn', label: '标签列（有监督必选）' }] as const).map(({key, label}) => <label key={key} className="space-y-2 text-sm"><span>{label}</span><select aria-label={label} className="w-full rounded-lg border bg-white p-2" value={config[key] ?? ''} onChange={event => setConfig(prev => ({ ...prev, [key]: event.target.value || undefined }))}><option value="">{key === 'textColumn' ? '请选择正文列' : '不使用'}</option>{columns.map(column => <option key={column} value={column}>{column}</option>)}</select></label>)}
              <div className="space-y-2 text-sm"><p>元数据列（STM 协变量）</p><div className="flex flex-wrap gap-3">{columns.filter(column => column !== config.textColumn).map(column => <label key={column} className="flex items-center gap-1"><input type="checkbox" checked={config.covariates?.includes(column) ?? false} onChange={event => setConfig(prev => ({ ...prev, covariates: event.target.checked ? [...(prev.covariates ?? []), column] : (prev.covariates ?? []).filter(item => item !== column) }))} />{column}</label>)}</div></div>
            </section>}
            {/* 基础配置 - 语言和模型选择 */}
            <div className="space-y-4">
              {/* 仅控制图表展示，不作为文本分词的语言提示。 */}
              <div className="space-y-2">
                <Label id="plot-language-label">绘图语言</Label>
                <RadioGroup
                  aria-labelledby="plot-language-label"
                  value={config.plotLanguage}
                  onValueChange={v => setConfig(prev => ({ ...prev, plotLanguage: v }))}
                  className="flex flex-wrap gap-4"
                >
                  {LANGUAGES.map(lang => {
                  return (
                    <div key={lang.value} className="flex min-h-9 items-center gap-2">
                      <RadioGroupItem value={lang.value} id={`lang-${lang.value}`} />
                      <Label htmlFor={`lang-${lang.value}`} className="font-normal cursor-pointer">
                        {lang.label}
                      </Label>
                    </div>
                  );
                })}
                </RadioGroup>
                <p className="text-xs leading-5 text-slate-500">仅用于图表标题、坐标轴和图例；不会改变数据的分词语言，也不会翻译原始文本。</p>
              </div>

              <section className="space-y-3 rounded-xl border border-slate-200 bg-slate-50/70 p-4" aria-labelledby="text-processing-label">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <Label id="text-processing-label">文本处理与停用词</Label>
                  <span className="rounded-full bg-blue-50 px-2.5 py-1 text-xs text-blue-700">多语言自动处理</span>
                </div>
                <p className="text-xs leading-5 text-slate-500">自动识别多语言及混合文本，按文字体系分词并加载对应停用词。内置中、英、德、法、西、意、葡、俄、日、韩语及通用词表。</p>
                <div aria-live="polite" className="text-sm text-slate-700">
                  {uploadingStopwords ? "正在上传词表…" : config.stopwords
                    ? <><span className="break-all font-medium">{config.stopwords.name}</span><span className="ml-2 text-xs text-slate-500">{config.stopwords.count} 个词 · 替换内置词表</span></>
                    : "当前使用内置词表（默认）"}
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  <label className={cn("cursor-pointer rounded-lg border bg-white px-3 py-2 text-xs font-medium text-slate-700 hover:bg-slate-100 focus-within:ring-2 focus-within:ring-blue-500", uploadingStopwords && "pointer-events-none opacity-50")}>
                    {config.stopwords ? "更换停用词表" : "上传自定义停用词表"}
                    <input aria-label="上传自定义停用词表" className="sr-only" type="file" accept=".txt,text/plain" disabled={uploadingStopwords} onChange={event => {
                      const file = event.target.files?.[0]
                      event.target.value = ""
                      if (file) void uploadStopwords(file)
                    }} />
                  </label>
                  <a href={service.stopwordsDownloadUrl} download className="rounded-lg border bg-white px-3 py-2 text-xs font-medium text-slate-700 hover:bg-slate-100">导出内置词表</a>
                  {config.stopwords && <Button size="sm" variant="ghost" disabled={uploadingStopwords} onClick={() => { setConfig(prev => ({ ...prev, stopwords: undefined })); setStopwordError(null) }}>恢复默认</Button>}
                </div>
                <p className="text-xs leading-5 text-slate-500">UTF-8 TXT，每行一个词，支持混合语言；最多 512 KB / 20000 个词。上传后替换本次分析的内置词表，恢复默认即可重新使用内置词表。导出包含全部内置语言词表。</p>
                {stopwordError && <p role="alert" className="text-xs text-red-600">{stopwordError} 当前词表未更改。</p>}
              </section>

              {/* 模型选择 */}
              <div className="space-y-3">
                <Label>选择模型（可多选）</Label>
                {MODEL_GROUPS.map(group => (
                  <div key={group.label}>
                    <p className="text-xs text-slate-500 mb-2">{group.label}</p>
                    <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                      {group.ids.map(modelId => {
                        const model = MODEL_LIST.find(m => m.id === modelId)!
                        return (
                          <div key={modelId} className={cn("flex min-w-0 items-center gap-2 rounded-xl border px-3 py-3", config.models.includes(modelId) ? "border-blue-200 bg-blue-50" : "border-slate-200 bg-white")}>
                            <Checkbox
                              id={`model-${modelId}`}
                              checked={config.models.includes(modelId)}
                              onCheckedChange={checked =>
                                handleModelToggle(modelId, !!checked)
                              }
                            />
                            <Label
                              htmlFor={`model-${modelId}`}
                              className={cn(
                                "flex-1 whitespace-nowrap font-normal cursor-pointer text-sm",
                                config.models.includes(modelId) && "font-medium text-blue-700"
                              )}
                            >
                              {model.name}
                            </Label>
                          </div>
                        )
                      })}
                    </div>
                  </div>
                ))}
              </div>

              {/* 全局词汇表大小 */}
              <div className="space-y-2">
                <Label htmlFor="shared-topics">统一主题数</Label>
                <div className="flex gap-2">
                  <Input id="shared-topics" type="number" min={2} max={100} value={sharedTopics} onChange={e => setSharedTopics(Math.max(2, Math.min(100, Number(e.target.value) || 2)))} />
                  <Button variant="outline" onClick={applySharedTopics}>应用到全部模型</Button>
                </div>
                <p className="text-xs leading-5 text-slate-500">HDP 使用该值作为截断上限；BERTopic 使用该值作为合并目标，不保证恰好得到相同数量。其他模型可在下方分别调整。</p>
              </div>
              <div className="space-y-2">
                <Label htmlFor="vocab-size" className="text-sm">全局词汇表大小</Label>
                <Input
                  type="number"
                  min={1000}
                  max={20000}
                  id="vocab-size"
                  value={config.vocabSize}
                  onChange={e =>
                    setConfig(prev => ({
                      ...prev,
                      vocabSize: parseInt(e.target.value) || 5000,
                    }))
                  }
                />
                <p className="text-xs text-slate-500">所有模型共用，推荐: 3000-10000</p>
              </div>
            </div>

            {/* 每个模型单独的参数配置选项卡 */}
            {config.models.length > 0 && (
              <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
                <TabsList className="w-full justify-start gap-1 overflow-x-auto h-auto p-1 bg-slate-100">
                  {config.models.map(modelId => (
                    <TabsTrigger
                      key={modelId}
                      value={modelId}
                      className="shrink-0 min-w-[80px] data-[state=active]:bg-white"
                    >
                      {getModelName(modelId)}
                    </TabsTrigger>
                  ))}
                </TabsList>

                {/* THETA 配置 */}
                <TabsContent value="theta" className="mt-4">
                  <div className="space-y-6 rounded-xl border border-slate-200 bg-slate-50/70 p-4 sm:p-5">
                    {/* THETA 专属配置 */}
                    <div className="space-y-4">
                      <h4 className="font-semibold text-blue-800">THETA 专属配置</h4>
                      <div className="space-y-3 rounded-xl border bg-white p-4">
                        <Label>嵌入计算位置</Label>
                        <RadioGroup aria-label="嵌入计算位置" value={config.embeddingProvider} onValueChange={v => setConfig(prev => ({ ...prev, embeddingProvider: v as "local" | "cloud", cloudConfirmed: false, ...(v === "cloud" ? { mode: "zero_shot" } : {}) }))} className="grid gap-3 sm:grid-cols-2">
                          <div className="flex items-center gap-2"><RadioGroupItem id="embedding-local" value="local" /><Label htmlFor="embedding-local">本地嵌入 · Qwen</Label></div>
                          <div className="flex items-center gap-2"><RadioGroupItem id="embedding-cloud" value="cloud" /><Label htmlFor="embedding-cloud">云端嵌入 · 已配置服务</Label></div>
                        </RadioGroup>
                        {config.embeddingProvider === "cloud" && <div className="space-y-3 text-xs leading-5 text-slate-600">
                          {embedding?.configured ? <p className="break-all">服务：{embedding.provider} · {embedding.model}<br />接收地址：{embedding.endpoint}</p> : <p role="alert" className="text-red-600">云端嵌入 API 未配置完整。请在「设置 → 嵌入模型」填写地址、模型与密钥后重试。</p>}
                          <Label htmlFor="cloud-limit">本轮最多外部请求次数</Label>
                          <Input id="cloud-limit" type="number" min={1} max={1000} value={config.externalRequestLimit} onChange={e => setConfig(prev => ({ ...prev, externalRequestLimit: Math.max(1, Math.min(1000, Number(e.target.value) || 1)) }))} />
                          <Label htmlFor="embedding-batch">每次请求的文本块数</Label>
                          <Input id="embedding-batch" type="number" min={1} max={64} value={(config.parameters.theta?.['prepare.batch_size'] as number ?? 32)} onChange={e => setConfig(prev => ({ ...prev, parameters: { ...prev.parameters, theta: { ...prev.parameters.theta, 'prepare.batch_size': Math.max(1, Math.min(64, Number(e.target.value) || 1)) } } }))} />
                          <p>长文会分块编码后合并；请求预算包含正文、词表和重试。增大批量可减少请求次数，不改变原始文本。</p>
                          <div className="flex items-start gap-2"><Checkbox id="cloud-consent" checked={config.cloudConfirmed} onCheckedChange={v => setConfig(prev => ({ ...prev, cloudConfirmed: v === true }))} /><Label htmlFor="cloud-consent" className="text-xs font-normal leading-5">允许将本次文本发送给上述服务并消耗 API 额度；达到请求上限即停止。云端仅支持零样本嵌入。</Label></div>
                        </div>}
                        <p className="text-xs text-slate-500">此选项仅用于 THETA；CTM、BERTopic 使用本地多语言 SBERT，其他模型不调用云端嵌入。</p>
                      </div>
                      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2">
                        <div className="space-y-2">
                          <Label className="text-sm">Qwen 模型尺寸</Label>
                          <RadioGroup
                            value={config.modelSize}
                            onValueChange={v =>
                              setConfig(prev => ({ ...prev, modelSize: v }))
                            }
                            className="grid gap-2"
                          >
                            {QWEN_SIZES.map(s => {
                            return (
                              <div key={s.value} className="flex min-h-9 items-center gap-2">
                                <RadioGroupItem value={s.value} id={`size-${s.value}`} />
                                <Label htmlFor={`size-${s.value}`} className="whitespace-nowrap font-normal text-sm">
                                  {s.label}
                                </Label>
                              </div>
                            );
                          })}
                          </RadioGroup>
                        </div>
                        <div className="space-y-2">
                          <Label className="text-sm">嵌入模式</Label>
                          <RadioGroup
                            value={config.mode}
                            onValueChange={v =>
                              setConfig(prev => ({ ...prev, mode: v as any }))
                            }
                            className="grid gap-2"
                          >
                            {EMBEDDING_MODES.map(mode => {
                            return (
                              <Tooltip key={mode.value}>
                                <TooltipTrigger asChild>
                                  <div className="flex min-h-9 items-center gap-2">
                                <RadioGroupItem value={mode.value} id={`mode-${mode.value}`} disabled={config.embeddingProvider === "cloud" && mode.value !== "zero_shot"} />
                                    <Label htmlFor={`mode-${mode.value}`} className="whitespace-nowrap font-normal text-sm cursor-pointer">
                                      {mode.label}
                                    </Label>
                                    <Info className="w-3.5 h-3.5 text-slate-400" />
                                  </div>
                                </TooltipTrigger>
                                <TooltipContent>{mode.tip}</TooltipContent>
                              </Tooltip>
                            );
                          })}
                          </RadioGroup>
                        </div>
                      </div>
                    </div>

                  </div>
                </TabsContent>
                <ModelParameterFields key={activeTab} model={activeTab} value={config.parameters[activeTab] ?? {}}
                  loadModel={service.model}
                  onValidity={setValid}
                  onChange={value => setConfig(prev => ({ ...prev, parameters: { ...prev.parameters, [activeTab]: value } }))} />
              </Tabs>
            )}
          </fieldset>
          </div>

          <DialogFooter className={styles.footer}>
            {(error || submitError) && <div className={styles.error}><SetupError error={submitError || error || ''} /></div>}
            <Button variant="outline" disabled={submitting} onClick={() => onOpenChange(false)}>
              取消
            </Button>
            <Button
              onClick={handleConfirm}
              disabled={submitting || !valid || config.models.length === 0 || uploadingStopwords || (!!columns && !config.textColumn) || (config.models.includes("theta") && config.embeddingProvider === "cloud" && (!embedding?.configured || !config.cloudConfirmed))}
              className="bg-blue-600 hover:bg-blue-700"
            >
              {submitting ? '正在校验并提交…' : confirmLabel}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </TooltipProvider>
  )
}
