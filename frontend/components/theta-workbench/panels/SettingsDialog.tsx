
import { OPEN_SOURCE_EDITION } from '@/lib/edition'
import { useEffect, useState } from 'react'
import {
  Button,
  IconSettingsOutline16,
  IconGlobeOutline14,
  IconLightOutline16,
  IconDarkOutline16,
  IconUserOutline16,
  Modal,
} from '../ui/index.ts'
import { useInferenceSettings } from '../inference-settings.tsx'
import { usePreferences } from '../preferences.tsx'
import { useAuth } from '@/contexts/auth-context'
import { accountStorageKey } from '../storage-scope.ts'
import css from './SettingsDialog.module.css'
import type { WebInferenceSettingsUpdate } from '../api/client.ts'
import type { DesktopEmbedding } from '@/types/desktop'

interface SettingsDialogProps { open: boolean; onClose: () => void; onAccountNameChange?: (name: string) => void }

type SettingsTab = 'account' | 'appearance' | 'language' | 'inference' | 'embedding'

const ACCOUNT_NAME_KEY = 'theta.frontend.account-name.v1'

export const SettingsDialog = ({ open, onClose, onAccountNameChange }: SettingsDialogProps): React.ReactElement => {
  const { locale, theme, setTheme } = usePreferences()
  const { user } = useAuth()
  const defaultAccountName = user?.username?.trim() || user?.full_name?.trim() || 'user'
  const { catalog, settings, update, loading: settingsLoading } = useInferenceSettings()
  const [tab, setTab] = useState<SettingsTab>(OPEN_SOURCE_EDITION ? 'appearance' : 'account')
  const [accountName, setAccountName] = useState(defaultAccountName)
  const [selectedProviderId, setSelectedProviderId] = useState<string>('')
  const [baseUrl, setBaseUrl] = useState('')
  const [model, setModel] = useState('')
  const [modelList, setModelList] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [clearApiKey, setClearApiKey] = useState(false)
  const [saving, setSaving] = useState(false)
  const [message, setMessage] = useState<string>()
  const [messageType, setMessageType] = useState<'error' | 'success'>('success')
  const [embedding, setEmbedding] = useState<DesktopEmbedding>()
  const [embeddingKey, setEmbeddingKey] = useState('')
  const [clearEmbeddingKey, setClearEmbeddingKey] = useState(false)

  useEffect(() => {
    if (!open || !window.thetaDesktop) return
    let active = true
    void window.thetaDesktop.read().then(value => {
      if (active) { setEmbedding(value.embedding); setEmbeddingKey(''); setClearEmbeddingKey(false) }
    }).catch(error => { if (active) { setMessageType('error'); setMessage(String(error)) } })
    return () => { active = false }
  }, [open])

  const chooseEmbeddingDirectory = async (field: 'localPath' | 'sbertPath') => {
    try {
      const directory = await window.thetaDesktop?.selectModel()
      if (directory) setEmbedding(current => current ? { ...current, [field]: directory } : current)
    } catch (error) { setMessageType('error'); setMessage(error instanceof Error ? error.message : String(error)) }
  }
  const saveEmbeddingSettings = async () => {
    if (!embedding || !window.thetaDesktop) return
    setSaving(true); setMessage(undefined)
    try {
      await window.thetaDesktop.saveEmbedding({ ...embedding, apiKey: embeddingKey, clearApiKey: clearEmbeddingKey })
      setEmbedding((await window.thetaDesktop.read()).embedding)
      setEmbeddingKey(''); setClearEmbeddingKey(false)
      setMessageType('success'); setMessage('Embedding 配置已保存。新任务将使用这份配置。')
    } catch (error) { setMessageType('error'); setMessage(error instanceof Error ? error.message : String(error)) }
    finally { setSaving(false) }
  }


  useEffect(() => {
    if (!open) return
    setTab(OPEN_SOURCE_EDITION ? 'appearance' : 'account')
    setMessage(undefined)
    setMessageType('success')
    setAccountName(localStorage.getItem(accountStorageKey(ACCOUNT_NAME_KEY, user?.id)) || defaultAccountName)
  }, [defaultAccountName, open, user?.id])

  useEffect(() => {
    if (!open || !settings || !catalog) return
    const provider = catalog.providers.find((item) => item.id === settings.llm.providerId)
      ?? catalog.providers.find((item) => item.selected)
      ?? catalog.providers[0]
    if (!provider) return
    setSelectedProviderId(provider.id)
    setBaseUrl(provider.baseUrl)
    setModel(provider.configuredModel || provider.models[0] || '')
    setModelList(provider.models.join(', '))
    setApiKey('')
    setClearApiKey(false)
  }, [open, settings?.llm.providerId, catalog])

  const currentProvider = catalog?.providers.find((item) => item.id === selectedProviderId)

  const normalizedModelList = (value: string): string[] => [...new Set(value.split(',')
    .map((entry) => entry.trim())
    .filter((entry) => entry.length > 0))]

  const saveInferenceSettings = async (): Promise<void> => {
    if (!currentProvider || !settings) return
    const nextBaseUrl = baseUrl.trim()
    const nextModel = model.trim()
    const nextModels = normalizedModelList(modelList)
    if (!nextBaseUrl) {
      setMessageType('error')
      setMessage(locale === 'zh-CN' ? 'Base URL 不能为空。' : 'Base URL is required.')
      return
    }
    if (!nextModel) {
      setMessageType('error')
      setMessage(locale === 'zh-CN' ? '模型不能为空。' : 'Model is required.')
      return
    }
    const models = nextModels.length > 0 ? [...nextModels.filter((candidate) => candidate !== nextModel), nextModel] : [nextModel]
    const input: WebInferenceSettingsUpdate = {
      llm: {
        providerId: currentProvider.id,
        baseUrl: nextBaseUrl,
        model: nextModel,
        models: [...new Set(models)],
        ...(apiKey.trim() ? { apiKey: apiKey.trim() } : {}),
        ...(clearApiKey ? { clearApiKey: true } : {}),
      },
    }
    setSaving(true)
    setMessage(undefined)
    try {
      await update(input)
      setMessageType('success')
      setMessage(locale === 'zh-CN' ? '模型配置已保存。' : 'Model configuration saved.')
      setApiKey('')
      setClearApiKey(false)
    } catch (cause) {
      setMessageType('error')
      setMessage(cause instanceof Error ? cause.message : String(cause))
    } finally {
      setSaving(false)
    }
  }

  const onSelectProvider = (value: string): void => {
    const provider = catalog?.providers.find((entry) => entry.id === value)
    if (!provider) return
    setSelectedProviderId(provider.id)
    setBaseUrl(provider.baseUrl)
    setModel(provider.configuredModel || provider.models[0] || '')
    setModelList(provider.models.join(', '))
    setApiKey('')
    setClearApiKey(false)
    setMessage(undefined)
  }

  const saveAccountName = (value: string): void => {
    const next = value.trim() || defaultAccountName
    setAccountName(next)
    localStorage.setItem(accountStorageKey(ACCOUNT_NAME_KEY, user?.id), next)
    onAccountNameChange?.(next)
  }

  const zh = locale === 'zh-CN'
  return (
    <Modal
      open={open}
      onClose={onClose}
      title={zh ? '设置' : 'Settings'}
      description={OPEN_SOURCE_EDITION ? (zh ? '管理界面与模型偏好。' : 'Manage interface and model preferences.') : (zh ? '管理当前浏览器中的账号与界面偏好。' : 'Manage account and interface preferences in this browser.')}
      className={css.dialog}
      contentClassName={css.content}
      footer={<Button variant="primary" className={css.doneButton} onClick={onClose}>{zh ? '完成' : 'Done'}</Button>}
    >
      <div className={css.layout}>
        <nav className={css.nav} aria-label={zh ? '设置分类' : 'Settings sections'}>
          {!OPEN_SOURCE_EDITION && <button type="button" className={tab === 'account' ? css.active : ''} onClick={() => setTab(OPEN_SOURCE_EDITION ? 'appearance' : 'account')}><IconUserOutline16 />{zh ? '账号管理' : 'Account'}</button>}
          <button type="button" className={tab === 'appearance' ? css.active : ''} onClick={() => setTab('appearance')}><IconLightOutline16 />{zh ? '外观' : 'Appearance'}</button>
          <button type="button" className={tab === 'language' ? css.active : ''} onClick={() => setTab('language')}><IconGlobeOutline14 size={16} />{zh ? '语言' : 'Language'}</button>
          <button type="button" className={tab === 'inference' ? css.active : ''} onClick={() => setTab('inference')}><IconSettingsOutline16 />{zh ? '模型 API' : 'Model API'}</button>
          {embedding && <button type="button" className={tab === 'embedding' ? css.active : ''} onClick={() => { setTab('embedding'); setMessage(undefined) }}><IconSettingsOutline16 />Embedding 模型</button>}
        </nav>

        {tab === 'account' && (
          <section className={css.section} aria-labelledby="account-settings-title">
            <div className={css.sectionHeader}><h3 id="account-settings-title">{zh ? '账号管理' : 'Account'}</h3><p>{zh ? '账号信息仅保存在当前浏览器中。' : 'Account details are stored only in this browser.'}</p></div>
            <div className={css.accountCard}>
              <span className={css.accountAvatar}>U</span>
              <div><strong>{accountName}</strong><small>{zh ? '本地账号' : 'Local account'}</small></div>
            </div>
            <label className={css.field}><span>{zh ? '显示名称' : 'Display name'}</span><input value={accountName} maxLength={40} onChange={(event) => setAccountName(event.target.value)} onBlur={(event) => saveAccountName(event.target.value)} /></label>
          </section>
        )}

        {tab === 'inference' && settings?.readOnly && (
          <section className={css.section}>
            <div className={css.sectionHeader}><h3>{zh ? '当前模型' : 'Current model'}</h3><p>{settings.llm.providerId} · {settings.llm.model}</p></div>
            <p>{zh ? '网页与 CLI 共用模型配置。请修改 agent/.env.local，然后重启 Agent API。' : 'The web app shares the CLI configuration. Edit agent/.env.local, then restart the Agent API.'}</p>
          </section>
        )}
        {tab === 'inference' && !settings?.readOnly && (
          <section className={css.section} aria-labelledby="inference-api-title">
            <div className={css.sectionHeader}><h3 id="inference-api-title">{zh ? '模型服务配置' : 'Model Service Settings'}</h3><p>{zh ? '为本地模型切换与 API Key 配置。' : 'Configure your local model provider and API key.'}</p></div>
            {settingsLoading ? (
              <p className={css.fieldDescription}>{zh ? '正在加载推理配置...' : 'Loading inference settings...'}</p>
            ) : catalog ? (
              <>
                <label className={css.field}>
                  <span>{zh ? '供应商' : 'Provider'}</span>
                  <select
                    value={selectedProviderId}
                    onChange={(event) => onSelectProvider(event.target.value)}
                    disabled={saving}
                    className={css.select}
                  >
                    {catalog.providers.map((provider) => (
                      <option key={provider.id} value={provider.id}>
                        {provider.displayName}
                        {provider.configured ? '' : zh ? '（未配置）' : ' (Not configured)'}
                      </option>
                    ))}
                  </select>
                </label>
                <label className={css.field}>
                  <span>{zh ? 'Base URL' : 'Base URL'}</span>
                  <input
                    value={baseUrl}
                    onChange={(event) => setBaseUrl(event.target.value)}
                    placeholder={zh ? '请输入 OpenAI 兼容端点' : 'Enter OpenAI compatible base URL'}
                  />
                </label>
                <label className={css.field}>
                  <span>{zh ? '模型' : 'Model'}</span>
                  <input
                    value={model}
                    onChange={(event) => setModel(event.target.value)}
                    placeholder={zh ? '例如 gpt-4o-mini' : 'e.g., gpt-4o-mini'}
                    list="inference-model-list"
                  />
                  <datalist id="inference-model-list">
                    {(currentProvider?.models ?? []).map((candidate) => (
                      <option key={candidate} value={candidate} />
                    ))}
                  </datalist>
                </label>
                <label className={css.field}>
                  <span>{zh ? '候选模型（逗号分隔）' : 'Available models (comma separated)'}</span>
                  <input
                    value={modelList}
                    onChange={(event) => setModelList(event.target.value)}
                    placeholder={zh ? '例如 gpt-4o-mini, gpt-4.1-mini' : 'e.g., gpt-4o-mini, gpt-4.1-mini'}
                  />
                </label>
                <div className={css.rowLabel}><span>{zh ? 'API Key' : 'API Key'}</span></div>
                <input
                  className={css.fieldInput}
                  type="password"
                  value={apiKey}
                  onChange={(event) => setApiKey(event.target.value)}
                  placeholder={currentProvider?.configured ? (zh ? '留空表示不修改' : 'Leave empty to keep existing') : ''}
                />
                <label className={css.optionLabel}>
                  <input
                    type="checkbox"
                    checked={clearApiKey}
                    onChange={(event) => setClearApiKey(event.target.checked)}
                  />
                  <span>{zh ? '清除当前供应商 API Key' : 'Clear API key for current provider'}</span>
                </label>
                <div className={css.actionRow}>
                  <Button
                    variant="outline"
                    onClick={() => void saveInferenceSettings()}
                    disabled={saving || settingsLoading}
                  >
                    {saving ? (zh ? '保存中…' : 'Saving...') : zh ? '保存配置' : 'Save Settings'}
                  </Button>
                  <span className={css.statusText}>
                    {currentProvider?.configured
                      ? zh ? '当前状态：已配置密钥' : 'Status: Configured'
                      : zh ? '当前状态：未配置密钥' : 'Status: Missing key'}
                  </span>
                </div>
                {message && <p className={css.inlineMessage} data-type={messageType}>{message}</p>}
              </>
            ) : (
              <p className={css.fieldDescription}>{zh ? '未获取到供应商配置。' : 'Provider catalog is unavailable.'}</p>
            )}
          </section>
        )}

        {tab === 'embedding' && embedding && (
          <section className={css.section} aria-labelledby="embedding-settings-title">
            <div className={css.sectionHeader}><h3 id="embedding-settings-title">Embedding 模型</h3><p>本地计算环境已内置。根据需要选择本地模型或云端向量服务。</p></div>
            <label className={css.field}><span>使用方式</span><select className={css.select} value={embedding.mode} onChange={event => setEmbedding({ ...embedding, mode: event.target.value as 'local' | 'cloud' })}><option value="local">本地模型</option><option value="cloud">云端 Embedding API</option></select></label>
            {embedding.mode === 'local' ? (
              <>
                <label className={css.field}><span>Embedding 模型名称</span><input value={embedding.localModel} placeholder="Qwen/Qwen3-Embedding-0.6B" onChange={event => setEmbedding({ ...embedding, localModel: event.target.value })} /></label>
                <p className={css.fieldDescription}>默认建议 Qwen Embedding，也可选择兼容当前引擎的本地模型。名称用于标识，实际加载所选目录中的权重。</p>
                <label className={css.field}><span>模型目录</span><input value={embedding.localPath} readOnly placeholder="按需下载模型后选择目录" /></label>
                <div className={css.actionRow}><Button variant="outline" onClick={() => void chooseEmbeddingDirectory('localPath')}>选择目录</Button><Button variant="ghost" onClick={() => setEmbedding({ ...embedding, localPath: '' })}>清除</Button><Button variant="ghost" onClick={() => void window.thetaDesktop?.openModels('qwen')}>模型下载 ↗</Button></div>
                <details><summary>CTM / BERTopic 兼容模型（可选）</summary><label className={css.field}><span>Sentence Transformers 模型目录</span><input value={embedding.sbertPath} readOnly placeholder="使用 CTM / BERTopic 时配置" /></label><div className={css.actionRow}><Button variant="outline" onClick={() => void chooseEmbeddingDirectory('sbertPath')}>选择目录</Button><Button variant="ghost" onClick={() => setEmbedding({ ...embedding, sbertPath: '' })}>清除</Button><Button variant="ghost" onClick={() => void window.thetaDesktop?.openModels('sbert')}>模型下载 ↗</Button></div></details>
                <p className={css.fieldDescription}>模型目录需包含 config.json、分词器与权重。应用不会自动下载大模型。</p>
              </>
            ) : (
              <>
                <label className={css.field}><span>供应商</span><select className={css.select} value={embedding.provider} onChange={event => setEmbedding({ ...embedding, provider: event.target.value as 'zhipu' | 'openai_compatible' })}><option value="zhipu">GLM / 智谱</option><option value="openai_compatible">自定义 OpenAI 兼容服务</option></select></label>
                <label className={css.field}><span>Base URL</span><input value={embedding.baseUrl} onChange={event => setEmbedding({ ...embedding, baseUrl: event.target.value })} placeholder="https://open.bigmodel.cn/api/paas/v4" /></label>
                <label className={css.field}><span>Embedding 模型</span><input value={embedding.model} onChange={event => setEmbedding({ ...embedding, model: event.target.value })} placeholder="embedding-3" /></label>
                <label className={css.field}><span>向量维度（可选）</span><input type="number" min="1" max="65536" value={embedding.dimensions ?? ''} onChange={event => setEmbedding({ ...embedding, dimensions: event.target.value === '' ? null : Number(event.target.value) })} placeholder="留空使用服务默认值" /></label>
                <label className={css.field}><span>Embedding API Key</span><input type="password" autoComplete="off" value={embeddingKey} onChange={event => setEmbeddingKey(event.target.value)} placeholder={embedding.apiKeyConfigured ? '已保存，留空保留' : '填写 Embedding 服务密钥'} /></label>
                <label className={css.optionLabel}><input type="checkbox" checked={clearEmbeddingKey} onChange={event => setClearEmbeddingKey(event.target.checked)} /><span>清除已保存的 Embedding API Key</span></label>
                <p className={css.fieldDescription}>云端 Embedding 当前用于 THETA zero-shot。执行时会发送文本到所选服务，仍需确认具体任务与请求预算；微调和其他模型使用本地权重。密钥加密保存在本机。</p>
              </>
            )}
            <div className={css.actionRow}><Button variant="outline" disabled={saving} onClick={() => void saveEmbeddingSettings()}>{saving ? '保存中…' : '保存 Embedding 配置'}</Button><Button variant="ghost" onClick={() => void window.thetaDesktop?.openData()}>打开数据目录</Button></div>
            {message && <p className={css.inlineMessage} data-type={messageType}>{message}</p>}
          </section>
        )}

        {tab === 'appearance' && (
          <section className={css.section} aria-labelledby="appearance-settings-title">
            <div className={css.sectionHeader}><h3 id="appearance-settings-title">{zh ? '外观' : 'Appearance'}</h3><p>{zh ? '选择适合当前环境的界面主题。' : 'Choose the interface theme for this environment.'}</p></div>
            <div className={css.choiceGrid}>
              {([
                ['light', <IconLightOutline16 />, zh ? '浅色' : 'Light'],
                ['dark', <IconDarkOutline16 />, zh ? '深色' : 'Dark'],
              ] as const).map(([value, icon, label]) => (
                <button key={value} type="button" className={theme === value ? css.choiceActive : ''} onClick={() => setTheme(value)}>{icon}<span>{label}</span><i aria-hidden="true" /></button>
              ))}
            </div>
          </section>
        )}

        {tab === 'language' && (
          <section className={css.section} aria-labelledby="language-settings-title">
            <div className={css.sectionHeader}><h3 id="language-settings-title">{zh ? '语言' : 'Language'}</h3><p>{zh ? '界面、训练状态与结果名称统一使用简体中文。' : 'Change the interface display language.'}</p></div>
            <div className={css.languageList}>
              <button type="button" className={css.choiceActive} aria-pressed="true" disabled><span><strong>简体中文</strong><small>系统显示语言</small></span><i aria-hidden="true" /></button>
            </div>
          </section>
        )}
      </div>
    </Modal>
  )
}
