import { useMemo, useState } from 'react'
import type {
  WebInferenceCatalog,
  WebReasoningEffort,
  WebReasoningMode,
} from '../api/client.ts'
import {
  IconChevronDownOutline14,
  Menu,
  type MenuEntry,
} from '../ui/index.ts'
import { usePreferences } from '../preferences.tsx'
import css from './InferenceMenus.module.css'

export type ReasoningProfile = 'auto' | 'low' | 'mid' | 'high' | 'xhigh'

export const reasoningProfile = (
  mode: WebReasoningMode,
  effort: WebReasoningEffort,
): ReasoningProfile => {
  if (mode !== 'reasoning') return 'auto'
  return effort === 'medium' ? 'mid' : effort
}

export const reasoningProfileSettings = (
  profile: ReasoningProfile,
): { reasoningMode: WebReasoningMode; reasoningEffort: WebReasoningEffort } =>
  profile === 'auto'
    ? { reasoningMode: 'auto', reasoningEffort: 'medium' }
    : {
        reasoningMode: 'reasoning',
        reasoningEffort: profile === 'mid' ? 'medium' : profile,
      }

interface PickerPresentationProps {
  presentation?: 'compact' | 'field'
  disabled?: boolean
  side?: 'top' | 'bottom'
}

export const ProviderModelPicker = ({
  catalog,
  providerId,
  model,
  onChange,
  presentation = 'compact',
  disabled = false,
  side = 'top',
}: PickerPresentationProps & {
  catalog: WebInferenceCatalog
  providerId: string | null
  model: string
  onChange: (selection: { providerId: string; model: string }) => void
}): React.ReactElement => {
  const { locale } = usePreferences()
  const [open, setOpen] = useState(false)
  const choices = useMemo(() => {
    const byId = new Map<string, { providerId: string; model: string }>()
    const entries: MenuEntry[] = []
    let index = 0
    for (const provider of catalog.providers) {
      if (provider.models.length === 0) continue
      entries.push({
        type: 'label',
        id: `provider-label-${provider.id}`,
        text: `${provider.displayName}${provider.configured ? '' : locale === 'zh-CN' ? ' · 未配置' : ' · Not configured'}`,
      })
      for (const candidate of provider.models) {
        const id = `model-choice-${index++}`
        byId.set(id, { providerId: provider.id, model: candidate })
        entries.push({
          id,
          label: candidate,
          disabled: !provider.configured,
        })
      }
    }
    return { byId, entries }
  }, [catalog.providers, locale])
  const selectedProvider = catalog.providers.find((provider) => provider.id === providerId)
  const selectedId = [...choices.byId.entries()].find(([, value]) =>
    value.providerId === providerId && value.model === model,
  )?.[0]
  const label = presentation === 'field'
    ? `${selectedProvider?.displayName ?? providerId ?? '—'} · ${model || '—'}`
    : `${selectedProvider?.displayName ?? providerId ?? 'Model'} · ${model || '—'}`

  return (
    <Menu
      open={open}
      className={presentation === 'field' ? css.fieldRoot : undefined}
      portal
      side={side}
      compact={presentation === 'compact'}
      selectedId={selectedId}
      items={choices.entries}
      onClose={() => setOpen(false)}
      onSelect={(id) => {
        const selection = choices.byId.get(id)
        if (!selection) return
        onChange(selection)
        setOpen(false)
      }}
      anchor={(
        <button
          type="button"
          className={`${css.trigger} ${css[presentation]}`}
          aria-label={locale === 'zh-CN' ? '选择供应商与模型' : 'Choose provider and model'}
          aria-haspopup="menu"
          aria-expanded={open}
          disabled={disabled}
          onClick={() => setOpen((current) => !current)}
        >
          <span>{label}</span>
          <IconChevronDownOutline14 />
        </button>
      )}
    />
  )
}

const profileLabels = {
  auto: ['Auto', 'Auto'],
  low: ['Low', 'Low'],
  mid: ['Mid', 'Mid'],
  high: ['High', 'High'],
  xhigh: ['XHigh', 'XHigh'],
} satisfies Record<ReasoningProfile, [string, string]>

export const ReasoningProfilePicker = ({
  value,
  onChange,
  presentation = 'compact',
  disabled = false,
  side = 'top',
}: PickerPresentationProps & {
  value: ReasoningProfile
  onChange: (profile: ReasoningProfile) => void
}): React.ReactElement => {
  const { locale } = usePreferences()
  const [open, setOpen] = useState(false)
  const profiles: ReasoningProfile[] = ['auto', 'low', 'mid', 'high', 'xhigh']
  const items: MenuEntry[] = profiles.map((profile) => ({
    id: profile,
    label: profileLabels[profile][locale === 'zh-CN' ? 0 : 1],
  }))
  return (
    <Menu
      open={open}
      className={presentation === 'field' ? css.fieldRoot : undefined}
      portal
      side={side}
      compact={presentation === 'compact'}
      selectedId={value}
      items={items}
      onClose={() => setOpen(false)}
      onSelect={(id) => {
        if (!profiles.includes(id as ReasoningProfile)) return
        onChange(id as ReasoningProfile)
        setOpen(false)
      }}
      anchor={(
        <button
          type="button"
          className={`${css.trigger} ${css[presentation]}`}
          aria-label={locale === 'zh-CN' ? '选择推理强度' : 'Choose reasoning effort'}
          aria-haspopup="menu"
          aria-expanded={open}
          disabled={disabled}
          onClick={() => setOpen((current) => !current)}
        >
          <span>{profileLabels[value][locale === 'zh-CN' ? 0 : 1]}</span>
          <IconChevronDownOutline14 />
        </button>
      )}
    />
  )
}
