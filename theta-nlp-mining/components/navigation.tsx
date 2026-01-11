"use client"

import Link from "next/link"
import Image from "next/image"
import { usePathname } from "next/navigation"
import { cn } from "@/lib/utils"
import { Database, BarChart3, Microscope } from "lucide-react"

export function Navigation() {
  const pathname = usePathname()

  const links = [
    { href: "/setup", label: "数据治理", icon: Database },
    { href: "/analytics", label: "分析仪表盘", icon: BarChart3 },
    { href: "/rag", label: "RAG工作台", icon: Microscope },
  ]

  return (
    <nav className="border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-16 items-center gap-8">
        <Link href="/" className="flex items-center gap-2 font-bold text-xl">
          <Image 
            src="/thetalogo.jpeg" 
            alt="THETA Logo" 
            width={32} 
            height={32} 
            className="object-contain"
          />
          THETA
        </Link>

        <div className="flex gap-1">
          {links.map((link) => {
            const Icon = link.icon
            return (
              <Link
                key={link.href}
                href={link.href}
                className={cn(
                  "flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-colors",
                  pathname === link.href
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:bg-muted hover:text-foreground",
                )}
              >
                <Icon className="w-4 h-4" />
                {link.label}
              </Link>
            )
          })}
        </div>
      </div>
    </nav>
  )
}
