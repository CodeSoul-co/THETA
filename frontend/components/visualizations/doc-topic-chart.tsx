"use client"

import { useRef, useState } from "react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Download, ChevronLeft, ChevronRight } from "lucide-react"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts"

interface DocTopicChartProps {
  documents: string[]
  distributions: number[][]
  numTopics: number
  title?: string
}

const COLORS = [
  '#3b82f6', '#6366f1', '#8b5cf6', '#a855f7', '#d946ef',
  '#ec4899', '#f43f5e', '#ef4444', '#f97316', '#f59e0b',
  '#eab308', '#84cc16', '#22c55e', '#10b981', '#14b8a6',
  '#06b6d4', '#0ea5e9', '#0284c7', '#2563eb', '#4f46e5',
]

export function DocTopicChart({ 
  documents, 
  distributions, 
  numTopics,
  title = "文档-主题分布" 
}: DocTopicChartProps) {
  const chartRef = useRef<HTMLDivElement>(null)
  const [selectedDoc, setSelectedDoc] = useState(0)
  const pageSize = 10
  const [page, setPage] = useState(0)

  // Prepare data for stacked bar chart view
  const stackedData = documents.slice(page * pageSize, (page + 1) * pageSize).map((doc, idx) => {
    const actualIdx = page * pageSize + idx
    const entry: Record<string, string | number> = { name: doc }
    distributions[actualIdx]?.forEach((value, topicIdx) => {
      entry[`Topic ${topicIdx + 1}`] = value
    })
    return entry
  })

  // Prepare data for single document view
  const singleDocData = distributions[selectedDoc]?.map((value, idx) => ({
    name: `Topic ${idx + 1}`,
    value,
  })) || []

  const handleExport = (format: 'png' | 'svg') => {
    const svg = chartRef.current?.querySelector('svg')
    if (!svg) return

    if (format === 'svg') {
      const svgData = new XMLSerializer().serializeToString(svg)
      const blob = new Blob([svgData], { type: 'image/svg+xml' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `doc-topic-${Date.now()}.svg`
      a.click()
      URL.revokeObjectURL(url)
    } else {
      const canvas = document.createElement('canvas')
      const ctx = canvas.getContext('2d')
      const img = new Image()
      const svgData = new XMLSerializer().serializeToString(svg)
      const svgBlob = new Blob([svgData], { type: 'image/svg+xml' })
      const url = URL.createObjectURL(svgBlob)
      
      img.onload = () => {
        canvas.width = svg.clientWidth * 2
        canvas.height = svg.clientHeight * 2
        ctx?.scale(2, 2)
        ctx?.drawImage(img, 0, 0)
        
        const pngUrl = canvas.toDataURL('image/png')
        const a = document.createElement('a')
        a.href = pngUrl
        a.download = `doc-topic-${Date.now()}.png`
        a.click()
        URL.revokeObjectURL(url)
      }
      img.src = url
    }
  }

  const totalPages = Math.ceil(documents.length / pageSize)

  return (
    <Card className="p-4 bg-white border border-slate-200">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-medium text-slate-900">{title}</h3>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={() => handleExport('png')}>
            <Download className="w-4 h-4 mr-1" />
            PNG
          </Button>
          <Button variant="outline" size="sm" onClick={() => handleExport('svg')}>
            <Download className="w-4 h-4 mr-1" />
            SVG
          </Button>
        </div>
      </div>

      {/* Document selector */}
      <div className="mb-4 flex items-center gap-4">
        <label className="text-sm text-slate-600">选择文档:</label>
        <select 
          value={selectedDoc}
          onChange={(e) => setSelectedDoc(Number(e.target.value))}
          className="px-3 py-1.5 border border-slate-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          {documents.map((doc, idx) => (
            <option key={idx} value={idx}>{doc}</option>
          ))}
        </select>
      </div>
      
      {/* Single document view */}
      <div ref={chartRef} className="h-64 mb-6">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={singleDocData} margin={{ top: 10, right: 10, left: 0, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis 
              dataKey="name" 
              tick={{ fontSize: 10 }} 
              angle={-45} 
              textAnchor="end"
              height={50}
            />
            <YAxis 
              tick={{ fontSize: 11 }}
              tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
              domain={[0, 1]}
            />
            <Tooltip 
              formatter={(value: number) => [`${(value * 100).toFixed(2)}%`, '占比']}
            />
            <Bar dataKey="value" radius={[4, 4, 0, 0]}>
              {singleDocData.map((_, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Overview with pagination */}
      <div className="border-t border-slate-200 pt-4">
        <div className="flex items-center justify-between mb-3">
          <h4 className="text-sm font-medium text-slate-700">文档概览 (堆叠视图)</h4>
          <div className="flex items-center gap-2">
            <Button 
              variant="ghost" 
              size="sm" 
              onClick={() => setPage(p => Math.max(0, p - 1))}
              disabled={page === 0}
            >
              <ChevronLeft className="w-4 h-4" />
            </Button>
            <span className="text-sm text-slate-600">
              {page + 1} / {totalPages}
            </span>
            <Button 
              variant="ghost" 
              size="sm" 
              onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}
              disabled={page >= totalPages - 1}
            >
              <ChevronRight className="w-4 h-4" />
            </Button>
          </div>
        </div>
        
        <div className="h-48">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={stackedData} margin={{ top: 10, right: 10, left: 0, bottom: 30 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis 
                dataKey="name" 
                tick={{ fontSize: 9 }} 
                angle={-45} 
                textAnchor="end"
                height={50}
              />
              <YAxis 
                tick={{ fontSize: 10 }}
                tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
              />
              <Tooltip />
              {Array.from({ length: numTopics }).map((_, idx) => (
                <Bar 
                  key={idx}
                  dataKey={`Topic ${idx + 1}`}
                  stackId="a"
                  fill={COLORS[idx % COLORS.length]}
                />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </Card>
  )
}
