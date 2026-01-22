"use client"

import { useRef, useMemo } from "react"
import { motion } from "framer-motion"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Download } from "lucide-react"

interface TopicBubbleChartProps {
  topics: string[]
  proportions: number[]
  topicWords?: Record<string, string[]>
  title?: string
}

export function TopicBubbleChart({ 
  topics, 
  proportions, 
  topicWords = {},
  title = "主题气泡图" 
}: TopicBubbleChartProps) {
  const containerRef = useRef<HTMLDivElement>(null)

  const bubbles = useMemo(() => {
    const maxProportion = Math.max(...proportions)
    const minSize = 40
    const maxSize = 120

    // Pack circles using a simple algorithm
    const result: Array<{
      name: string
      x: number
      y: number
      size: number
      proportion: number
      words: string[]
      color: string
    }> = []

    const colors = [
      'rgba(59, 130, 246, 0.7)',
      'rgba(99, 102, 241, 0.7)',
      'rgba(168, 85, 247, 0.7)',
      'rgba(236, 72, 153, 0.7)',
      'rgba(34, 197, 94, 0.7)',
      'rgba(234, 179, 8, 0.7)',
      'rgba(249, 115, 22, 0.7)',
      'rgba(239, 68, 68, 0.7)',
      'rgba(20, 184, 166, 0.7)',
      'rgba(6, 182, 212, 0.7)',
    ]

    // Sort by proportion descending for better packing
    const sortedIndices = proportions
      .map((p, i) => ({ p, i }))
      .sort((a, b) => b.p - a.p)
      .map(item => item.i)

    sortedIndices.forEach((idx, sortIdx) => {
      const proportion = proportions[idx]
      const normalizedSize = (proportion / maxProportion)
      const size = minSize + normalizedSize * (maxSize - minSize)
      
      // Simple spiral placement
      const angle = sortIdx * 0.8
      const radius = 100 + sortIdx * 25
      const centerX = 200
      const centerY = 150
      
      let x = centerX + Math.cos(angle) * radius * 0.6
      let y = centerY + Math.sin(angle) * radius * 0.4

      // Keep within bounds
      x = Math.max(size / 2 + 10, Math.min(400 - size / 2 - 10, x))
      y = Math.max(size / 2 + 10, Math.min(300 - size / 2 - 10, y))

      result.push({
        name: topics[idx],
        x,
        y,
        size,
        proportion,
        words: topicWords[String(idx)] || [],
        color: colors[idx % colors.length]
      })
    })

    return result
  }, [topics, proportions, topicWords])

  const handleExport = () => {
    const svg = containerRef.current?.querySelector('svg')
    if (!svg) return

    const svgData = new XMLSerializer().serializeToString(svg)
    const blob = new Blob([svgData], { type: 'image/svg+xml' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `topic-bubbles-${Date.now()}.svg`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <Card className="p-4 bg-white border border-slate-200">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-medium text-slate-900">{title}</h3>
        <Button variant="outline" size="sm" onClick={handleExport}>
          <Download className="w-4 h-4 mr-1" />
          导出
        </Button>
      </div>
      
      <div ref={containerRef} className="relative">
        <svg width="100%" height="300" viewBox="0 0 400 300" className="overflow-visible">
          <defs>
            {bubbles.map((bubble, idx) => (
              <radialGradient key={`gradient-${idx}`} id={`bubble-gradient-${idx}`}>
                <stop offset="0%" stopColor={bubble.color.replace('0.7', '0.9')} />
                <stop offset="100%" stopColor={bubble.color.replace('0.7', '0.4')} />
              </radialGradient>
            ))}
          </defs>
          
          {bubbles.map((bubble, idx) => (
            <motion.g
              key={idx}
              initial={{ scale: 0, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: idx * 0.05, type: "spring", stiffness: 200 }}
            >
              <circle
                cx={bubble.x}
                cy={bubble.y}
                r={bubble.size / 2}
                fill={`url(#bubble-gradient-${idx})`}
                stroke={bubble.color.replace('0.7', '1')}
                strokeWidth={2}
                className="cursor-pointer hover:opacity-80 transition-opacity"
              >
                <title>
                  {`${bubble.name}\n占比: ${(bubble.proportion * 100).toFixed(2)}%`}
                  {bubble.words.length > 0 && `\n关键词: ${bubble.words.slice(0, 5).join(', ')}`}
                </title>
              </circle>
              
              {bubble.size > 50 && (
                <>
                  <text
                    x={bubble.x}
                    y={bubble.y - 5}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    fontSize={Math.min(12, bubble.size / 6)}
                    fill="#1e293b"
                    fontWeight="600"
                    className="pointer-events-none"
                  >
                    {bubble.name}
                  </text>
                  <text
                    x={bubble.x}
                    y={bubble.y + 10}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    fontSize={Math.min(10, bubble.size / 7)}
                    fill="#64748b"
                    className="pointer-events-none"
                  >
                    {(bubble.proportion * 100).toFixed(1)}%
                  </text>
                </>
              )}
            </motion.g>
          ))}
        </svg>
      </div>

      {/* Legend */}
      <div className="mt-4 flex flex-wrap gap-2">
        {bubbles.slice(0, 10).map((bubble, idx) => (
          <div key={idx} className="flex items-center gap-1.5 text-xs">
            <div 
              className="w-3 h-3 rounded-full" 
              style={{ backgroundColor: bubble.color }}
            />
            <span className="text-slate-600">{bubble.name}</span>
          </div>
        ))}
      </div>
    </Card>
  )
}
