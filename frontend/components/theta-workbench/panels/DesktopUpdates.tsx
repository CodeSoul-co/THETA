import { useEffect, useState } from 'react'
import type { DesktopUpdateState } from '@/types/desktop'
import { Button } from '../ui/index.ts'
import css from './SettingsDialog.module.css'

export function DesktopUpdates({ zh }: { zh: boolean }) {
  const [state, setState] = useState<DesktopUpdateState>()
  const [error, setError] = useState('')
  useEffect(() => {
    const api = window.thetaDesktop?.updates
    if (!api) return
    let active = true
    const unsubscribe = api.subscribe(value => { if (active) setState(value) })
    void api.state().then(value => { if (active) setState(value) }).catch(cause => { if (active) setError(String(cause)) })
    return () => { active = false; unsubscribe() }
  }, [])
  const action = async (work: () => Promise<unknown>) => {
    setError('')
    try { await work() } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)) }
  }
  if (!state) return <p>{error || (zh ? '正在读取更新信息…' : 'Loading update information…')}</p>
  const labels = zh ? {
    disabled: '更新功能在安装后的桌面应用中启用。', idle: '自动检查新版本，也可以立即检查。', checking: '正在检查更新…', current: '当前已是最新可用版本。',
    available: '发现新版本，可以在此下载。', downloading: '正在下载更新，期间可以继续使用。', ready: '更新已下载并通过校验。', installing: '正在退出并安装更新…', error: '更新未完成，请检查网络后重试。',
  } : {
    disabled: 'Updates are enabled in the installed desktop app.', idle: 'New versions are checked automatically, or check now.', checking: 'Checking for updates…', current: 'You have the latest available version.',
    available: 'A new version is available to download here.', downloading: 'Downloading. You can keep using the app.', ready: 'Update downloaded and verified.', installing: 'Quitting to install the update…', error: 'Update failed. Check your connection and retry.',
  }
  const busy = ['checking', 'downloading', 'installing', 'disabled'].includes(state.status)
  const api = window.thetaDesktop!.updates
  return <section className={css.section} aria-labelledby="desktop-updates-title">
    <div className={css.sectionHeader}><h3 id="desktop-updates-title">{zh ? '应用更新' : 'App updates'}</h3><p>THETA {state.currentVersion}{state.availableVersion ? ` → ${state.availableVersion}` : ''}</p></div>
    <p role="status" aria-live="polite">{labels[state.status]}</p>
    {state.status === 'downloading' && <div><progress className="w-full" aria-label={zh ? '更新下载进度' : 'Update download progress'} value={state.percent} max={100} /><p>{state.percent}%</p></div>}
    <label className="flex items-center gap-2"><input type="checkbox" checked={state.automatic} disabled={state.status === 'disabled'} onChange={event => void action(() => api.configure(event.target.checked))} />{zh ? '启动后自动检查更新' : 'Automatically check for updates after launch'}</label>
    <p className={css.fieldDescription}>{state.manualInstall
      ? zh ? '更新会直接下载到本机。下载后打开安装包，退出 THETA 并拖入“应用程序”替换旧版，无需前往社区下载。' : 'Updates download directly to your Mac. Open the installer, quit THETA, and drag it into Applications to replace the old version.'
      : zh ? '下载完成后可重启安装。安装前请保存工作并等待分析完成，项目和配置会保留。' : 'Restart to install after downloading. Save your work and wait for analyses to finish. Projects and settings are retained.'}</p>
    {(error || state.error) && <p role="alert" className="break-words text-sm text-red-600">{error || state.error}</p>}
    <div className={css.actionRow}>
      <Button variant="outline" disabled={busy || state.status === 'ready'} onClick={() => void action(() => api.check())}>{zh ? '检查更新' : 'Check for updates'}</Button>
      {(state.status === 'available' || (state.status === 'error' && state.availableVersion)) && <Button variant="primary" onClick={() => void action(() => api.download())}>{zh ? '下载更新' : 'Download update'}</Button>}
      {state.status === 'ready' && <Button variant="primary" onClick={() => void action(() => api.install())}>{state.manualInstall ? zh ? '打开安装包' : 'Open installer' : zh ? '重启并安装' : 'Restart and install'}</Button>}
    </div>
  </section>
}
