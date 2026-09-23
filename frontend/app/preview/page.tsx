"use client"

import { Suspense, useEffect, useState } from "react"
import { useSearchParams } from "next/navigation"
import { API_BASE } from "@/lib/api/config"

function PreviewContent() {
  const searchParams = useSearchParams()
  const dataset = searchParams.get("dataset")
  const path = searchParams.get("path")
  const model = searchParams.get("model") || "theta"
  const [htmlContent, setHtmlContent] = useState<string>("")
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string>("")

  useEffect(() => {
    if (!dataset || !path) {
      setError("缺少参数")
      setLoading(false)
      return
    }

    const proxyUrl = `${API_BASE}/api/results/${encodeURIComponent(dataset)}/visualizations/file?model=${encodeURIComponent(model)}&path=${encodeURIComponent(path)}`

    fetch(proxyUrl, {
      credentials: "same-origin",
    })
      .then((res) => {
        if (!res.ok) throw new Error("加载失败")
        return res.text()
      })
      .then((text) => {
        setHtmlContent(text)
        setLoading(false)
      })
      .catch((e) => {
        setError(e.message || "加载失败")
        setLoading(false)
      })
  }, [dataset, path, model])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen bg-slate-100">
        <div className="text-slate-500">加载中...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-screen bg-slate-100">
        <div className="text-red-500">{error}</div>
      </div>
    )
  }

  // 构建 srcdoc，注入 base 标签使相对路径资源正确加载
  const proxyUrl = dataset && path
    ? `${API_BASE}/api/results/${encodeURIComponent(dataset)}/visualizations/file?model=${encodeURIComponent(model)}&path=${encodeURIComponent(path)}`
    : ""
  const baseUrl = proxyUrl.replace(/\?.*$/, "")
  const srcdoc = htmlContent.replace("<head>", `<head><base href="${baseUrl}">`)

  return (
    <div className="w-full h-screen">
      <iframe
        srcDoc={srcdoc}
        title="HTML Preview"
        className="w-full h-full border-0"
        sandbox="allow-scripts allow-same-origin allow-forms allow-popups"
      />
    </div>
  )
}

function LoadingFallback() {
  return (
    <div className="flex items-center justify-center h-screen bg-slate-100">
      <div className="text-slate-500">加载中...</div>
    </div>
  )
}

export default function PreviewPage() {
  return (
    <Suspense fallback={<LoadingFallback />}>
      <PreviewContent />
    </Suspense>
  )
}
