"use client"

import { useRef, useMemo } from "react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Download } from "lucide-react"
import { Tooltip } from "@/components/ui/tooltip"

interface TopicHeatmapProps {
  matrix: number[][]
  labels: string[]
  title?: string
  topicWords?: Record<string, string[]>
}

export function TopicHeatmap({ 
  matrix, 
  labels, 
  title = "主题相似度热力图",
  topicWords = {}
}: TopicHeatmapProps) {
  const containerRef = useRef<HTMLDivElement>(null)

  const colorScale = useMemo(() => {
    return (value: number) => {
      // Blue to red gradient
      const h = (1 - value) * 240 // Hue from blue (240) to red (0)
      const s = 70 + value * 30 // Saturation 70-100%
      const l = 95 - value * 45 // Lightness 95-50%
      return `hsl(${h}, ${s}%, ${l}%)`
    }
  }, [])

  const handleExport = (format: 'png' | 'svg') => {
    const svg = containerRef.current?.querySelector('svg')
    if (!svg) return

    if (format === 'svg') {
      const svgData = new XMLSerializer().serializeToString(svg)
      const blob = new Blob([svgData], { type: 'image/svg+xml' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `heatmap-${Date.now()}.svg`
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
        a.download = `heatmap-${Date.now()}.png`
        a.click()
        URL.revokeObjectURL(url)
      }
      img.src = url
    }
  }

  const cellSize = Math.min(400 / labels.length, 40)
  const margin = { top: 60, right: 20, bottom: 20, left: 60 }
  const width = cellSize * labels.length + margin.left + margin.right
  const height = cellSize * labels.length + margin.top + margin.bottom

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
      
      <div ref={containerRef} className="overflow-auto">
        <svg width={width} height={height} className="mx-auto">
          {/* Column labels */}
          <g transform={`translate(${margin.left}, ${margin.top - 5})`}>
            {labels.map((label, i) => (
              <text
                key={`col-${i}`}
                x={i * cellSize + cellSize / 2}
                y={0}
                fontSize={10}
                textAnchor="end"
                transform={`rotate(-45 ${i * cellSize + cellSize / 2} 0)`}
                fill="#64748b"
              >
                {label}
              </text>
            ))}
          </g>

          {/* Row labels */}
          <g transform={`translate(${margin.left - 5}, ${margin.top})`}>
            {labels.map((label, i) => (
              <text
                key={`row-${i}`}
                x={0}
                y={i * cellSize + cellSize / 2}
                fontSize={10}
                textAnchor="end"
                dominantBaseline="middle"
                fill="#64748b"
              >
                {label}
              </text>
            ))}
          </g>

          {/* Cells */}
          <g transform={`translate(${margin.left}, ${margin.top})`}>
            {matrix.map((row, i) =>
              row.map((value, j) => (
                <g key={`cell-${i}-${j}`}>
                  <rect
                    x={j * cellSize}
                    y={i * cellSize}
                    width={cellSize - 1}
                    height={cellSize - 1}
                    fill={colorScale(value)}
                    rx={2}
                    className="cursor-pointer hover:stroke-slate-400 hover:stroke-2"
                  >
                    <title>
                      {`${labels[i]} - ${labels[j]}: ${value.toFixed(3)}`}
                      {topicWords[String(i)] && `\n关键词: ${topicWords[String(i)].slice(0, 3).join(', ')}`}
                    </title>
                  </rect>
                  {cellSize > 25 && (
                    <text
                      x={j * cellSize + cellSize / 2}
                      y={i * cellSize + cellSize / 2}
                      fontSize={8}
                      textAnchor="middle"
                      dominantBaseline="middle"
                      fill={value > 0.5 ? '#fff' : '#1e293b'}
                    >
                      {value.toFixed(2)}
                    </text>
                  )}
                </g>
              ))
            )}
          </g>

          {/* Color legend */}
          <g transform={`translate(${width - 80}, ${margin.top})`}>
            <text x={0} y={-10} fontSize={10} fill="#64748b">相似度</text>
            {[0, 0.25, 0.5, 0.75, 1].map((v, i) => (
              <g key={`legend-${i}`} transform={`translate(0, ${i * 15})`}>
                <rect width={15} height={12} fill={colorScale(v)} rx={2} />
                <text x={20} y={10} fontSize={9} fill="#64748b">{v.toFixed(2)}</text>
              </g>
            ))}
          </g>
        </svg>
      </div>
    </Card>
  )
}
