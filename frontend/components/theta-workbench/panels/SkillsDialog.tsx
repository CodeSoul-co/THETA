'use client'
import { useEffect, useRef, useState } from 'react'
import { BookOpen, Download, FolderPlus, PackagePlus, Puzzle, RefreshCw, Trash2 } from 'lucide-react'
import { Modal, Button } from '../ui/index'
import css from './SkillsDialog.module.css'

interface Skill { id: string; name: string; description: string; enabled: boolean; bundled: boolean; source?: string; files: number }
export function SkillsDialog({ open, onClose, locale }: { open: boolean; onClose(): void; locale: string }) {
  const zh = locale === 'zh-CN'
  const [skills, setSkills] = useState<Skill[]>([])
  const [url, setUrl] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const folder = useRef<HTMLInputElement>(null), archive = useRef<HTMLInputElement>(null)
  async function request(route = '', init?: RequestInit) {
    const response = await fetch('/api/v3/skills' + route, init)
    const payload = await response.json()
    if (!response.ok || !payload.ok) throw new Error(payload.error?.message ?? payload.error ?? `HTTP ${response.status}`)
    return payload.data
  }
  async function refresh() { setSkills((await request()).skills) }
  async function action(work: () => Promise<unknown>, message?: string) {
    setBusy(true); setError(''); setNotice('')
    try { await work(); await refresh(); if (message) setNotice(message) }
    catch (error) { setError(error instanceof Error ? error.message : String(error)) }
    finally { setBusy(false) }
  }
  useEffect(() => {
    if (!open) return
    void action(refresh)
    const timer = setInterval(() => { void refresh().catch(error => setError(String(error))) }, 5000)
    return () => clearInterval(timer)
  }, [open])
  async function importFiles(list: FileList | null) {
    if (!list?.length) return
    const selected = Array.from(list)
    await action(async () => {
      let files: { path: string; content: string }[]
      if (selected.length === 1 && /\.(tar\.gz|tgz)$/i.test(selected[0].name)) {
        await request('/import-archive', {method:'POST',headers:{'content-type':'application/gzip'},body:selected[0]});return
      }
      if (selected.length === 1 && /\.zip$/i.test(selected[0].name)) {
        const JSZip = (await import('jszip')).default
        const zip = await JSZip.loadAsync(await selected[0].arrayBuffer())
        files = await Promise.all(Object.values(zip.files).filter(file => !file.dir).map(async file => ({ path: file.name, content: await file.async('base64') })))
      } else {
        files = await Promise.all(selected.map(async file => ({ path: file.webkitRelativePath || file.name, content: await new Promise<string>((resolve, reject) => {
          const reader = new FileReader(); reader.onload = () => resolve(String(reader.result).split(',')[1]); reader.onerror = () => reject(reader.error); reader.readAsDataURL(file)
        }) })))
      }
      const skill = files.find(file => /(^|\/)SKILL\.md$/.test(file.path))
      if (!skill) throw new Error(zh ? '请选择包含 SKILL.md 的文件夹或 ZIP 文件' : 'Choose a folder or ZIP containing SKILL.md')
      const prefix = skill.path.slice(0, -'SKILL.md'.length)
      files = files.filter(file => file.path.startsWith(prefix)).map(file => ({ ...file, path: file.path.slice(prefix.length) }))
      await request('/import', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ files }) })
    }, zh ? '导入成功，Agent 现在可以使用此技能。' : 'Imported. The Agent can now use this skill.')
  }
  return <Modal open={open} onClose={onClose} title={zh ? '技能管理' : 'Skill Manager'} closeLabel={zh ? '关闭' : 'Close'} className={css.dialog}
    description={zh ? '为 Agent 添加第三方技能，管理内置绘图模板。' : 'Add third-party skills and manage bundled plotting templates.'}>
    <div className={css.actions}>
      <Button disabled={busy} onClick={() => folder.current?.click()}><FolderPlus size={16} />{zh ? '导入文件夹' : 'Import folder'}</Button>
      <Button disabled={busy} onClick={() => archive.current?.click()}><PackagePlus size={16} />{zh ? '导入压缩包 / SKILL.md' : 'Import archive / SKILL.md'}</Button>
      <Button disabled={busy} onClick={() => void action(refresh)} aria-label={zh ? '刷新技能' : 'Refresh skills'}><RefreshCw size={16} /></Button>
      <input ref={folder} hidden type="file" multiple {...{ webkitdirectory: '', directory: '' }} onChange={event => { void importFiles(event.target.files); event.target.value = '' }} />
      <input ref={archive} hidden type="file" accept=".zip,.tar.gz,.tgz,.md" onChange={event => { void importFiles(event.target.files); event.target.value = '' }} />
    </div>
    <form className={css.install} onSubmit={event => { event.preventDefault(); void action(async () => { await request('/install', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ url }) }); setUrl('') }, zh ? '下载成功，已添加到技能列表。' : 'Downloaded and added to your skills.') }}>
      <label htmlFor="skill-source">{zh ? 'GitHub 仓库或技能文件夹地址' : 'GitHub repository or skill folder URL'}</label>
      <div><input id="skill-source" type="url" placeholder="https://github.com/owner/repository/tree/main/skills/example" value={url} onChange={event => setUrl(event.target.value)} required disabled={busy} /><Button type="submit" disabled={busy || !url.trim()}><Download size={16} />{zh ? '下载并添加' : 'Download and add'}</Button></div>
    </form>
    {busy && <p role="status">{zh ? '正在处理…' : 'Working…'}</p>}
    {error && <p className={css.error} role="alert">{error}</p>}
    {notice && <p className={css.notice} role="status">{notice}</p>}
    <div className={css.list}>{skills.map(skill => <article className={css.card} key={skill.id}>
      <div className={css.heading}><Puzzle size={20} /><strong>{skill.name}</strong><span>{skill.bundled ? (zh ? '内置' : 'Bundled') : (zh ? '已导入' : 'Imported')}</span></div>
      <p>{skill.id === 'data-viz' ? (zh ? '科学绘图模板、样式与使用指南。Agent 可读取并按实际数据选用。' : 'Scientific plotting templates, styles and instructions for the Agent.') : skill.description || skill.id}</p>
      {skill.source && skill.source !== 'local' && <p className={css.source}>{skill.source}</p>}
      <div className={css.controls}><label><input type="checkbox" checked={skill.enabled} disabled={busy} onChange={event => void action(() => request('/' + skill.id, { method: 'PATCH', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ enabled: event.target.checked }) }))} />{zh ? 'Agent 可使用' : 'Available to Agent'}</label>
        <span>{skill.files} {zh ? '个文件' : 'files'}</span>
        <a href={`/api/v3/skills/${skill.id}/download`} download><Download size={14} />{zh ? '导出' : 'Export'} (.tar.gz)</a>
        {!skill.bundled && <Button disabled={busy} aria-label={(zh ? '删除 ' : 'Remove ') + skill.name} onClick={() => void action(() => request('/' + skill.id, { method: 'DELETE' }))}><Trash2 size={15} /></Button>}
      </div>
    </article>)}</div>
    <p className={css.note}><BookOpen size={16} />{zh ? 'Agent 下载的技能也会显示在这里。导入只保存文件；执行脚本、安装依赖仍遵循任务授权。' : 'Agent downloads appear here too. Import saves files; script execution and dependencies still follow task permissions.'}</p>
  </Modal>
}
