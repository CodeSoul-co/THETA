import { useId, useState, type ReactNode } from 'react'
import { ChevronDown } from 'lucide-react'
import css from './CollapsibleMessage.module.css'

export function CollapsibleMessage({ text, children, enabled, zh }: { text: string; children: ReactNode; enabled: boolean; zh: boolean }) {
  const [collapsed, setCollapsed] = useState(false)
  const id = useId()
  const preview = text.replace(/!?\[([^\]]*)\]\([^)]*\)/gu, '$1').replace(/https?:\/\/\S+/gu, '').replace(/[#*`>\[\]]/gu, '').replace(/\s+/gu, ' ').trim()
  if (!enabled) return <>{children}</>
  return <div className={css.message}>
    <button type="button" className={css.toggle} aria-expanded={!collapsed} aria-controls={id} onClick={() => setCollapsed(value => !value)}>
      {collapsed ? (zh ? '展开回复' : 'Expand reply') : (zh ? '收起回复' : 'Collapse reply')}<ChevronDown size={13} />
    </button>
    {collapsed && <p className={css.preview}>{preview.slice(0, 110)}{preview.length > 110 ? '…' : ''}</p>}
    <div id={id} hidden={collapsed}>{children}</div>
  </div>
}
