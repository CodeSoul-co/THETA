"use client"

import { useRef } from "react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Download } from "lucide-react"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  PieChart,
  Pie,
  Legend,
} from "recharts"

interface TopicDistributionChartProps {
  data: Array<{ name: string; value: number; words?: string[] }>
  title?: string
  chartType?: 'bar' | 'pie'
}

const COLORS = [
  '#3b82f6', '#6366f1', '#8b5cf6', '#a855f7', '#d946ef',
  '#ec4899', '#f43f5e', '#ef4444', '#f97316', '#f59e0b',
  '#eab308', '#84cc16', '#22c55e', '#10b981', '#14b8a6',
  '#06b6d4', '#0ea5e9', '#0284c7', '#2563eb', '#4f46e5',
]

export function TopicDistributionChart({ 
  data, 
  title = "主题分布", 
  chartType = 'bar' 
}: TopicDistributionChartProps) {
  const chartRef = useRef<HTMLDivElement>(null)

  const handleExport = (format: 'png' | 'svg') => {
    const svg = chartRef.current?.querySelector('svg')
    if (!svg) return

    if (format === 'svg') {
      const svgData = new XMLSerializer().serializeToString(svg)
      const blob = new Blob([svgData], { type: 'image/svg+xml' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `topic-distribution-${Date.now()}.svg`
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
        a.download = `topic-distribution-${Date.now()}.png`
        a.click()
        URL.revokeObjectURL(url)
      }
      img.src = url
    }
  }

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const item = payload[0].payload
      return (
        <div className="bg-white p-3 rounded-lg shadow-lg border border-slate-200">
          <p className="font-medium text-slate-900">{item.name}</p>
          <p className="text-sm text-blue-600">占比: {(item.value * 100).toFixed(2)}%</p>
          {item.words && item.words.length > 0 && (
            <div className="mt-2 text-xs text-slate-500">
              <p className="font-medium mb-1">关键词:</p>
              <p>{item.words.slice(0, 5).join(', ')}</p>
            </div>
          )}
        </div>
      )
    }
    return null
  }

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
      
      <div ref={chartRef} className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          {chartType === 'bar' ? (
            <BarChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis 
                dataKey="name" 
                tick={{ fontSize: 11 }} 
                angle={-45} 
                textAnchor="end"
                height={60}
              />
              <YAxis 
                tick={{ fontSize: 11 }}
                tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
              />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                {data.map((_, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          ) : (
            <PieChart>
              <Pie
                data={data}
                cx="50%"
                cy="50%"
                labelLine={false}
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
              >
                {data.map((_, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip content={<CustomTooltip />} />
              <Legend />
            </PieChart>
          )}
        </ResponsiveContainer>
      </div>
    </Card>
  )
}
