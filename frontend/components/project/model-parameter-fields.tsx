"use client"

import { useEffect, useState } from 'react'
import { apiFetch, API_BASE } from '@/lib/api/config'
import { editableParameters, parameterLabel, parameterChoiceLabel, parameterGroup, parseParameter, type ModelContract, type ModelParameters, type ParameterSpec, type ParameterValue } from '@/lib/model-parameters'

function ParameterInput({ label, spec, value, onChange, onValidity }: { label: string; spec: ParameterSpec; value: ParameterValue | undefined; onChange: (value: ParameterValue | undefined) => void; onValidity: (error: string) => void }) {
  const stringify = (value: ParameterValue | undefined) => value === null ? 'null' : Array.isArray(value) ? value.join(', ') : String(value ?? '')
  const [raw, setRaw] = useState(stringify(value))
  useEffect(() => { setRaw(stringify(value)) }, [value])
  return <input aria-label={label} className="w-full rounded-md border p-2 text-sm" type="text" inputMode={['int', 'float'].includes(spec.type) ? 'decimal' : 'text'} value={raw} placeholder={spec.default == null ? '使用引擎默认值' : `接口默认：${spec.default}`} onChange={e => {
    setRaw(e.target.value)
    try { onChange(parseParameter(e.target.value, spec)); e.target.setCustomValidity(''); onValidity('') }
    catch (error) { const message = (error as Error).message; e.target.setCustomValidity(message); onValidity(message) }
  }} />
}

const defaultLoadModel = (model: string) => apiFetch<ModelContract>(API_BASE, `/api/models/${model}`)
export function ModelParameterFields({ model, value, onChange, onValidity, loadModel = defaultLoadModel }: {
  model: string; value: ModelParameters; onChange: (value: ModelParameters) => void; onValidity: (valid: boolean) => void
  loadModel?: (model: string) => Promise<ModelContract>
}) {
  const [contract, setContract] = useState<ModelContract | null>(null)
  const [error, setError] = useState('')
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({})
  useEffect(() => {
    let cancelled = false
    onValidity(false)
    loadModel(model).then(data => {
      if (!cancelled) { setContract(data); setError(''); onValidity(true) }
    }).catch(error => { if (!cancelled) setError(`无法加载模型参数：${error.message}`) })
    return () => { cancelled = true }
  }, [model, loadModel])
  if (!contract) return <p role="status" className="text-sm text-slate-600">{error || '正在读取模型实现与参数契约…'}</p>
  return <section className="space-y-4 rounded-xl border bg-slate-50/70 p-5">
    <h3 className="font-semibold">{model.toUpperCase()} 模型参数</h3>
    <p className="text-sm leading-6 text-slate-600">{contract.description}</p>
    <p className="text-xs leading-5 text-slate-500">空白表示不覆盖执行引擎。各字段展示对应接口默认值；同名的不同调用入口不需要重复填写。所有参数在提交后由模型契约校验。</p>
    {['训练参数', '预处理参数', '模型专属与高级参数'].map(group => <details key={group} className="rounded-xl border bg-white p-4" open={group === '训练参数'}><summary className="cursor-pointer text-sm font-semibold">{group}</summary>
    <div className="mt-4 grid items-start gap-4 sm:grid-cols-2">{editableParameters(contract).filter(([key]) => parameterGroup(key) === group).map(([key, spec]) => <label key={key} className="min-w-0 space-y-1.5 rounded-lg border bg-white p-3">
      <span className="block text-sm font-medium">{parameterLabel(key)}</span>
      <span className="block break-all font-mono text-[11px] text-slate-500">{key}</span>
      {spec.type === 'bool' || spec.choices ? <select aria-label={`${model} ${key}`} className="w-full rounded-md border p-2 text-sm" value={value[key] === undefined ? '' : String(value[key])} onChange={e => {
        const next = { ...value }; const parsed = parseParameter(e.target.value, spec)
        if (parsed === undefined) delete next[key]; else next[key] = parsed
        onChange(next)
      }}><option value="">使用引擎默认值{spec.default != null ? `（${parameterChoiceLabel(String(spec.default))}）` : ''}</option>{(spec.choices ?? ['true', 'false']).map(option => <option key={String(option)} value={String(option)}>{parameterChoiceLabel(option)}</option>)}</select>
      : <ParameterInput label={`${model} ${key}`} spec={spec} value={value[key]} onChange={parsed => {
        const next = { ...value }; if (parsed === undefined) delete next[key]; else next[key] = parsed; onChange(next)
      }} onValidity={message => { const next = { ...fieldErrors, [key]: message }; setFieldErrors(next); onValidity(!Object.values(next).some(Boolean)) }} />}
      {fieldErrors[key] && <span role="alert" className="block text-xs text-red-600">{fieldErrors[key]}</span>}
      {spec.nullable && <span className="block text-xs text-slate-500">输入 null 可显式使用自动值；留空则不覆盖。</span>}
      {spec.note && <span className="block text-xs leading-5 text-slate-500">{spec.note}</span>}
      <span className="block break-all text-[10px] text-slate-400">实现：{spec.source}</span>
    </label>)}</div></details>)}
    {error && <p role="alert" className="text-sm text-red-600">{error}</p>}
  </section>
}
