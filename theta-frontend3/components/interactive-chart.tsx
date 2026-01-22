"use client"

import React from "react"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  ZAxis,
  LineChart,
  Line,
  Cell,
} from "recharts"

interface ChartProps {
  type: "bar" | "heatmap" | "bubble" | "line" | "wordcloud"
  data: any[]
  title?: string
  width?: number | string
  height?: number | string
  compact?: boolean // 紧凑模式，用于卡片内展示
}

export function InteractiveChart({ type, data, title, width = "100%", height = 300, compact = false }: ChartProps) {
  // 根据紧凑模式调整字体大小
  const fontSize = compact ? 10 : 12
  const tickFontSize = compact ? 9 : 11
  
  const renderChart = () => {
    switch (type) {
      case "bar":
        return (
          <ResponsiveContainer width={width} height={height}>
            <BarChart data={data} margin={{ top: 5, right: 5, left: compact ? -15 : 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
              <XAxis 
                dataKey="name" 
                fontSize={tickFontSize} 
                tickLine={false} 
                axisLine={false}
                tick={{ fill: '#64748b' }}
                interval={compact ? 'preserveStartEnd' : 0}
                angle={compact ? -45 : 0}
                textAnchor={compact ? 'end' : 'middle'}
                height={compact ? 50 : 30}
              />
              <YAxis 
                fontSize={tickFontSize} 
                tickLine={false} 
                axisLine={false} 
                tickFormatter={(value) => compact ? `${Math.round(value)}` : `${value}`}
                tick={{ fill: '#64748b' }}
                width={compact ? 30 : 40}
              />
              <Tooltip 
                cursor={{ fill: 'rgba(0,0,0,0.05)' }}
                contentStyle={{ 
                  borderRadius: '8px', 
                  border: 'none', 
                  boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
                  fontSize: fontSize,
                  padding: '8px 12px'
                }}
                labelStyle={{ fontWeight: 600, marginBottom: 4 }}
              />
              <Bar 
                dataKey="value" 
                fill="#3b82f6" 
                radius={[4, 4, 0, 0]} 
                name="数值"
                maxBarSize={compact ? 30 : 50}
              />
            </BarChart>
          </ResponsiveContainer>
        )
      
      case "line":
        return (
          <ResponsiveContainer width={width} height={height}>
            <LineChart data={data} margin={{ top: 5, right: 10, left: compact ? -15 : 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
              <XAxis 
                dataKey="name" 
                fontSize={tickFontSize} 
                tickLine={false} 
                axisLine={false}
                tick={{ fill: '#64748b' }}
                interval={compact ? 'preserveStartEnd' : 0}
              />
              <YAxis 
                fontSize={tickFontSize} 
                tickLine={false} 
                axisLine={false}
                tick={{ fill: '#64748b' }}
                width={compact ? 30 : 40}
              />
              <Tooltip 
                contentStyle={{ 
                  borderRadius: '8px', 
                  border: 'none', 
                  boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
                  fontSize: fontSize,
                  padding: '8px 12px'
                }}
              />
              {!compact && <Legend wrapperStyle={{ fontSize: fontSize }} />}
              <Line 
                type="monotone" 
                dataKey="value" 
                stroke="#3b82f6" 
                strokeWidth={2} 
                dot={{ r: compact ? 3 : 4, fill: '#3b82f6' }} 
                activeDot={{ r: compact ? 5 : 6 }} 
                name="趋势" 
              />
            </LineChart>
          </ResponsiveContainer>
        )
      
      case "bubble":
        return (
          <ResponsiveContainer width={width} height={height}>
            <ScatterChart margin={{ top: 10, right: 10, left: compact ? -10 : 0, bottom: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis 
                type="number" 
                dataKey="x" 
                name="X轴" 
                fontSize={tickFontSize} 
                tickLine={false} 
                axisLine={false}
                tick={{ fill: '#64748b' }}
                domain={[0, 100]}
              />
              <YAxis 
                type="number" 
                dataKey="y" 
                name="Y轴" 
                fontSize={tickFontSize} 
                tickLine={false} 
                axisLine={false}
                tick={{ fill: '#64748b' }}
                domain={[0, 100]}
                width={compact ? 25 : 35}
              />
              <ZAxis type="number" dataKey="z" range={compact ? [30, 200] : [50, 400]} name="大小" />
              <Tooltip 
                cursor={{ strokeDasharray: '3 3' }}
                contentStyle={{ 
                  borderRadius: '8px', 
                  border: 'none', 
                  boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
                  fontSize: fontSize,
                  padding: '8px 12px'
                }}
                formatter={(value: number, name: string) => [value.toFixed(1), name]}
              />
              <Scatter name="主题" data={data} fill="#8884d8" shape="circle">
                {data.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.fill || '#3b82f6'} fillOpacity={0.8} />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        )
        
      case "heatmap":
        // 响应式热力图
        const gridSize = Math.ceil(Math.sqrt(data.length))
        return (
          <div 
            className="grid gap-0.5 sm:gap-1 h-full w-full p-1 sm:p-2"
            style={{ 
              gridTemplateColumns: `repeat(${gridSize}, minmax(0, 1fr))`,
            }}
          >
            {data.map((item, i) => {
              const intensity = Math.min(1, Math.max(0, item.value))
              return (
                <div 
                  key={i} 
                  className="flex items-center justify-center rounded-sm sm:rounded text-white transition-all hover:scale-105 hover:z-10 cursor-pointer"
                  style={{ 
                    backgroundColor: `hsl(217, 91%, ${70 - intensity * 45}%)`,
                    aspectRatio: '1/1',
                    minWidth: 0,
                    minHeight: 0,
                  }}
                  title={`${item.name}: ${(item.value * 100).toFixed(0)}%`}
                >
                  <span className="text-[8px] sm:text-[10px] font-medium opacity-90">
                    {(item.value * 100).toFixed(0)}
                  </span>
                </div>
              )
            })}
          </div>
        )
        
      case "wordcloud":
        // 响应式词云
        return (
          <div className="flex flex-wrap items-center justify-center gap-1 sm:gap-2 h-full p-2 sm:p-4 overflow-hidden content-center">
            {data.map((item, i) => {
              const baseSize = compact ? 10 : 12
              const maxAddSize = compact ? 14 : 18
              const size = baseSize + (item.value * maxAddSize)
              const opacity = 0.6 + (item.value * 0.4)
              const colors = ['#3b82f6', '#8b5cf6', '#ec4899', '#f97316', '#10b981', '#06b6d4']
              return (
                <span 
                  key={i} 
                  style={{ 
                    fontSize: `${size}px`, 
                    opacity,
                    color: colors[i % colors.length],
                    fontWeight: item.value > 0.7 ? 600 : 400,
                    lineHeight: 1.2,
                  }}
                  className="transition-all hover:scale-110 cursor-default whitespace-nowrap"
                  title={`${item.name}: ${(item.value * 100).toFixed(0)}%`}
                >
                  {item.name}
                </span>
              )
            })}
          </div>
        )

      default:
        return (
          <div className="flex items-center justify-center h-full text-slate-400 text-sm">
            不支持的图表类型
          </div>
        )
    }
  }

  return (
    <div className="w-full h-full flex flex-col min-h-0">
      {title && (
        <h4 className={`font-medium text-slate-700 mb-2 sm:mb-4 ${compact ? 'text-xs' : 'text-sm'}`}>
          {title}
        </h4>
      )}
      <div className="flex-1 min-h-0 min-w-0">
        {renderChart()}
      </div>
    </div>
  )
}
