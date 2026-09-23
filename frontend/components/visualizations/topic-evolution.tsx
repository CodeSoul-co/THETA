"use client"

import { useRef, useState } from "react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Download, Play, Pause } from "lucide-react"
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts"

interface TopicEvolutionProps {
  timePoints: string[]
  topicProportions: number[][] // [timePoint][topic] = proportion
  topicLabels: string[]
  title?: string
}

const COLORS = [
  '#3b82f6', '#6366f1', '#8b5cf6', '#a855f7', '#d946ef',
  '#ec4899', '#f43f5e', '#ef4444', '#f97316', '#f59e0b',
  '#eab308', '#84cc16', '#22c55e', '#10b981', '#14b8a6',
  '#06b6d4', '#0ea5e9', '#0284c7', '#2563eb', '#4f46e5',
]

export function TopicEvolution({ 
  timePoints, 
  topicProportions, 
  topicLabels,
  title = "主题演化时间线" 
}: TopicEvolutionProps) {
  const chartRef = useRef<HTMLDivElement>(null)
  const [selectedTopics, setSelectedTopics] = useState<Set<number>>(
    new Set(topicLabels.slice(0, 5).map((_, i) => i))
  )
  const [isAnimating, setIsAnimating] = useState(false)
  const [animationIndex, setAnimationIndex] = useState(timePoints.length - 1)

  // Prepare data for recharts
  const chartData = timePoints.slice(0, animationIndex + 1).map((time, timeIdx) => {
    const entry: Record<string, string | number> = { time }
    topicLabels.forEach((label, topicIdx) => {
      if (selectedTopics.has(topicIdx)) {
        entry[label] = topicProportions[timeIdx]?.[topicIdx] || 0
      }
    })
    return entry
  })

  const handleTopicToggle = (topicIdx: number) => {
    const newSelected = new Set(selectedTopics)
    if (newSelected.has(topicIdx)) {
      newSelected.delete(topicIdx)
    } else {
      newSelected.add(topicIdx)
    }
    setSelectedTopics(newSelected)
  }

  const handlePlayAnimation = () => {
    if (isAnimating) {
      setIsAnimating(false)
      return
    }

    setIsAnimating(true)
    setAnimationIndex(0)
    
    const interval = setInterval(() => {
      setAnimationIndex(prev => {
        if (prev >= timePoints.length - 1) {
          clearInterval(interval)
          setIsAnimating(false)
          return timePoints.length - 1
        }
        return prev + 1
      })
    }, 500)
  }

  const handleExport = (format: 'png' | 'svg') => {
    const svg = chartRef.current?.querySelector('svg')
    if (!svg) return

    if (format === 'svg') {
      const svgData = new XMLSerializer().serializeToString(svg)
      const blob = new Blob([svgData], { type: 'image/svg+xml' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `topic-evolution-${Date.now()}.svg`
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
        a.download = `topic-evolution-${Date.now()}.png`
        a.click()
        URL.revokeObjectURL(url)
      }
      img.src = url
    }
  }

  return (
    <Card className="p-4 bg-white border border-slate-200">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-medium text-slate-900">{title}</h3>
        <div className="flex gap-2">
          <Button 
            variant="outline" 
            size="sm" 
            onClick={handlePlayAnimation}
            className="gap-1"
          >
            {isAnimating ? (
              <>
                <Pause className="w-4 h-4" />
                暂停
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                播放
              </>
            )}
          </Button>
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

      {/* Topic selector */}
      <div className="mb-4 flex flex-wrap gap-2">
        {topicLabels.map((label, idx) => (
          <button
            key={idx}
            onClick={() => handleTopicToggle(idx)}
            className={`px-3 py-1 text-xs rounded-full border transition-colors ${
              selectedTopics.has(idx)
                ? 'bg-blue-50 border-blue-300 text-blue-700'
                : 'bg-slate-50 border-slate-200 text-slate-500 hover:bg-slate-100'
            }`}
            style={selectedTopics.has(idx) ? { borderColor: COLORS[idx % COLORS.length] } : {}}
          >
            {label}
          </button>
        ))}
      </div>
      
      <div ref={chartRef} className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis 
              dataKey="time" 
              tick={{ fontSize: 11 }}
              angle={-45}
              textAnchor="end"
              height={50}
            />
            <YAxis 
              tick={{ fontSize: 11 }}
              tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
              domain={[0, 'auto']}
            />
            <Tooltip 
              formatter={(value: number, name: string) => [`${(value * 100).toFixed(2)}%`, name]}
            />
            <Legend />
            {topicLabels.map((label, idx) => (
              selectedTopics.has(idx) && (
                <Line
                  key={idx}
                  type="monotone"
                  dataKey={label}
                  stroke={COLORS[idx % COLORS.length]}
                  strokeWidth={2}
                  dot={{ r: 3 }}
                  activeDot={{ r: 5 }}
                />
              )
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Time slider */}
      <div className="mt-4">
        <input
          type="range"
          min={0}
          max={timePoints.length - 1}
          value={animationIndex}
          onChange={(e) => setAnimationIndex(Number(e.target.value))}
          className="w-full"
        />
        <div className="flex justify-between text-xs text-slate-500 mt-1">
          <span>{timePoints[0]}</span>
          <span className="font-medium text-blue-600">{timePoints[animationIndex]}</span>
          <span>{timePoints[timePoints.length - 1]}</span>
        </div>
      </div>
    </Card>
  )
}
