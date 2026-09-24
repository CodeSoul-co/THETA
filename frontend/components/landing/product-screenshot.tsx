"use client"

import { useState } from 'react'
import Image from 'next/image'
import { Expand } from 'lucide-react'
import { Dialog, DialogContent, DialogTitle, DialogDescription } from '@/components/ui/dialog'

export function ProductScreenshot({ name, alt, priority = false }: { name: string; alt: string; priority?: boolean }) {
  const [open, setOpen] = useState(false)
  const src = `/screenshots/${name}.png`
  return <>
    <button type="button" onClick={() => setOpen(true)} aria-label={`放大查看：${alt}`} className="group relative block w-full overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-lg shadow-slate-200/40 transition hover:border-blue-300 focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-blue-600">
      <Image src={src} alt={alt} width={1278} height={768} sizes="(min-width: 1024px) 600px, 100vw" priority={priority} className="h-auto w-full" />
      <span className="absolute bottom-3 right-3 inline-flex items-center gap-1.5 rounded-full border border-slate-200 bg-white/95 px-3 py-1.5 text-xs text-slate-600 shadow-sm"><Expand className="size-3.5" />查看大图</span>
    </button>
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent className="w-[96vw] max-w-7xl p-3 sm:max-w-7xl sm:p-5">
        <DialogTitle className="pr-8 text-base">{alt}</DialogTitle>
        <DialogDescription>THETA 实际界面 · 演示项目</DialogDescription>
        <Image src={src} alt={alt} width={1278} height={768} sizes="96vw" className="max-h-[78vh] w-full rounded-lg object-contain" />
      </DialogContent>
    </Dialog>
  </>
}
