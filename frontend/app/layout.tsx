import type React from "react"
import type { Metadata } from "next"
import { Analytics } from "@vercel/analytics/next"
import { AuthProvider } from "@/contexts/auth-context"
import { Toaster } from "@/components/ui/sonner"
import "./globals.css"

// 使用 CSS 变量定义字体，避免构建时访问 Google Fonts
const fontClassName = "font-sans"

export const metadata: Metadata = {
  title: "THETA 智能分析平台",
  description: "Enterprise-grade intelligent data analysis platform",
  generator: "v0.app",
  icons: {
    icon: [
      { url: "/theta-cat-favicon-v1.ico", type: "image/x-icon", sizes: "16x16 32x32 48x48" },
      { url: "/theta-cat-favicon-32-v1.png", type: "image/png", sizes: "32x32" },
      { url: "/theta-cat-favicon-16-v1.png", type: "image/png", sizes: "16x16" },
    ],
    apple: { url: "/theta-cat-apple-touch-v1.png", type: "image/png", sizes: "180x180" },
  },
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="zh-CN" suppressHydrationWarning>
      <body className={`${fontClassName} antialiased`}>
        <AuthProvider>
          {children}
        </AuthProvider>
        <Toaster richColors position="top-center" />
        <Analytics />
      </body>
    </html>
  )
}
