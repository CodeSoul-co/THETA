import type React from "react"
import type { Metadata } from "next"
import { Geist, Geist_Mono } from "next/font/google"
import { Analytics } from "@vercel/analytics/next"
import "./globals.css"
import { Navigation } from "@/components/navigation"

const _geist = Geist({ subsets: ["latin"] })
const _geistMono = Geist_Mono({ subsets: ["latin"] })

export const metadata: Metadata = {
  title: "THETA - 科研数据分析平台",
  description: "专业的社会科学与医疗科研数据分析平台，支持BERTopic主题建模和RAG知识检索",
  generator: "v0.app",
  icons: {
    icon: [
      {
        url: "/thetalogo.jpeg",
        sizes: "32x32",
        type: "image/jpeg",
      },
    ],
    apple: "/thetalogo.jpeg",
  },
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="zh-CN" className="dark">
      <body className={`font-sans antialiased`}>
        <Navigation />
        {children}
        <Analytics />
      </body>
    </html>
  )
}
