import { useState } from 'react'
import { useInferenceSettings } from '../inference-settings.tsx'
import { usePreferences } from '../preferences.tsx'
import css from '../styles/app.module.css'
import {
  ProviderModelPicker,
} from './InferenceMenus.tsx'

export const InferenceSelector = ({ disabled = false }: { disabled?: boolean }): React.ReactElement => {
  const { catalog, settings, loading, update, error: loadError, refresh } = useInferenceSettings()
  const { locale } = usePreferences()
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState<string>()

  const apply = async (input: {
    providerId?: string
    model?: string
  }): Promise<void> => {
    if (!settings || busy) return
    const nextProvider = catalog?.providers.find((item) =>
      item.id === (input.providerId ?? settings.llm.providerId),
    )
    const model = input.model ?? settings.llm.model
    if (!nextProvider || !model) return
    setBusy(true)
    setMessage(undefined)
    try {
      await update({
        llm: {
          providerId: nextProvider.id,
          model,
          baseUrl: nextProvider.baseUrl,
          models: [...new Set([...nextProvider.models, model])],
        },
      })
    } catch (cause) {
      setMessage(cause instanceof Error ? cause.message : String(cause))
    } finally {
      setBusy(false)
    }
  }

  if (!loading && loadError && (!settings || !catalog)) {
    return <button type="button" className={css.modelPickerLoading} title={loadError} onClick={() => void refresh()}>{locale === 'zh-CN' ? '模型连接失败 · 重试' : 'Connection failed · Retry'}</button>
  }

  if (loading || !settings || !catalog) {
    return <button type="button" className={css.modelPickerLoading} disabled>{locale === 'zh-CN' ? '正在检查 AI' : 'Checking AI'}</button>
  }

  const selectedProvider = catalog.providers.find((provider) => provider.id === settings.llm.providerId)
  if (settings.readOnly) {
    return <span className={css.modelPickerLoading} title={locale === 'zh-CN' ? '与 CLI 共用模型配置；修改 agent/.env.local 后重启 Agent API。' : 'Shared with CLI; edit agent/.env.local and restart the Agent API.'}>
      {settings.llm.apiKeyConfigured ? `${selectedProvider?.displayName ?? settings.llm.providerId} · ${settings.llm.model}` : locale === 'zh-CN' ? '请配置 agent/.env.local' : 'Configure agent/.env.local'}
    </span>
  }
  const inferenceAvailable = selectedProvider?.configured === true && settings.llm.apiKeyConfigured
  if (!inferenceAvailable) {
    return (
      <button
        type="button"
        className={`${css.modelPickerLoading} ${css.modelPickerUnavailable}`}
        title={locale === 'zh-CN' ? '当前模型供应商未配置 API Key，请在设置中配置。' : 'Current model provider is missing API key. Configure it in Settings.'}
        disabled
      >
        {locale === 'zh-CN' ? '模型供应商未配置' : 'Model provider not configured'}
      </button>
    )
  }

  return (
    <div className={css.quickModelPicker} title={message} aria-busy={busy}>
      <ProviderModelPicker
        catalog={catalog}
        providerId={settings.llm.providerId}
        model={settings.llm.model}
        disabled={busy || disabled}
        onChange={(selection) => void apply(selection)}
      />
      {busy && <span className={css.quickModelBusy} aria-hidden="true" />}
    </div>
  )
}
