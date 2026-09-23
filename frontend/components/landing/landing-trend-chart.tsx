"use client"

import { Line, LineChart, ResponsiveContainer, XAxis } from "recharts"

const TREND_DATA = [
  { month: "1月", negative: 12, positive: 65 },
  { month: "2月", negative: 14, positive: 63 },
  { month: "3月", negative: 18, positive: 62 },
]

export function LandingTrendChart() {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={TREND_DATA}>
        <XAxis
          dataKey="month"
          axisLine={false}
          tickLine={false}
          tick={{ fontSize: 10, fill: "#94a3b8" }}
        />
        <Line
          type="monotone"
          dataKey="positive"
          stroke="#22c55e"
          strokeWidth={2}
          dot={{ fill: "#22c55e", strokeWidth: 0, r: 3 }}
        />
        <Line
          type="monotone"
          dataKey="negative"
          stroke="#ef4444"
          strokeWidth={2}
          dot={{ fill: "#ef4444", strokeWidth: 0, r: 3 }}
        />
      </LineChart>
    </ResponsiveContainer>
  )
}
