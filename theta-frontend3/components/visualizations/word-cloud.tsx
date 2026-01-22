"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Download, RefreshCw } from "lucide-react"

interface WordCloudProps {
  words: Array<{ text: string; weight: number }>
  title?: string
  onExport?: (format: 'png' | 'svg') => void
}

export function WordCloud({ words, title = "词云图", onExport }: WordCloudProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [layout, setLayout] = useState<Array<{ text: string; weight: number; x: number; y: number; fontSize: number; color: string }>>([])

  // Generate word cloud layout
  const generateLayout = useCallback(() => {
    if (!containerRef.current || words.length === 0) return

    const width = containerRef.current.clientWidth
    const height = containerRef.current.clientHeight
    const centerX = width / 2
    const centerY = height / 2

    // Normalize weights
    const maxWeight = Math.max(...words.map(w => w.weight))
    const minWeight = Math.min(...words.map(w => w.weight))
    const weightRange = maxWeight - minWeight || 1

    // Color palette
    const colors = [
      'rgb(59, 130, 246)',   // blue-500
      'rgb(99, 102, 241)',   // indigo-500
      'rgb(168, 85, 247)',   // purple-500
      'rgb(236, 72, 153)',   // pink-500
      'rgb(34, 197, 94)',    // green-500
      'rgb(234, 179, 8)',    // yellow-500
      'rgb(249, 115, 22)',   // orange-500
    ]

    const newLayout = words.slice(0, 50).map((word, index) => {
      const normalizedWeight = (word.weight - minWeight) / weightRange
      const fontSize = 12 + normalizedWeight * 28 // 12px to 40px

      // Spiral layout
      const angle = index * 0.5
      const radius = 20 + index * 8
      const x = centerX + Math.cos(angle) * radius * 0.8
      const y = centerY + Math.sin(angle) * radius * 0.5

      return {
        text: word.text,
        weight: word.weight,
        x: Math.max(50, Math.min(width - 50, x)),
        y: Math.max(20, Math.min(height - 20, y)),
        fontSize,
        color: colors[index % colors.length]
      }
    })

    setLayout(newLayout)
  }, [words])

  useEffect(() => {
    generateLayout()
    window.addEventListener('resize', generateLayout)
    return () => window.removeEventListener('resize', generateLayout)
  }, [generateLayout])

  const handleExport = (format: 'png' | 'svg') => {
    if (onExport) {
      onExport(format)
    } else {
      // Default export using canvas
      const svg = containerRef.current?.querySelector('svg')
      if (!svg) return

      if (format === 'svg') {
        const svgData = new XMLSerializer().serializeToString(svg)
        const blob = new Blob([svgData], { type: 'image/svg+xml' })
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `wordcloud-${Date.now()}.svg`
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
          a.download = `wordcloud-${Date.now()}.png`
          a.click()
          URL.revokeObjectURL(url)
        }
        img.src = url
      }
    }
  }

  return (
    <Card className="p-4 bg-white border border-slate-200">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-medium text-slate-900">{title}</h3>
        <div className="flex gap-2">
          <Button variant="ghost" size="sm" onClick={generateLayout}>
            <RefreshCw className="w-4 h-4" />
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
      
      <div 
        ref={containerRef} 
        className="relative h-64 bg-gradient-to-br from-slate-50 to-slate-100 rounded-lg overflow-hidden"
      >
        <svg width="100%" height="100%" className="absolute inset-0">
          {layout.map((item, index) => (
            <text
              key={index}
              x={item.x}
              y={item.y}
              fontSize={item.fontSize}
              fill={item.color}
              textAnchor="middle"
              dominantBaseline="middle"
              className="font-medium transition-all duration-300 hover:opacity-70 cursor-pointer"
              style={{ fontFamily: 'system-ui, sans-serif' }}
            >
              {item.text}
            </text>
          ))}
        </svg>
      </div>
      
      {words.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center text-slate-400">
          暂无数据
        </div>
      )}
    </Card>
  )
}
