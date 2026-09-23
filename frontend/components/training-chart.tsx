"use client"

import { systemText } from "@/lib/presentation"

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts"

interface TrainingChartProps {
  epochs?: number[]
  loss?: number[]
  accuracy?: (number | null)[]
  /** 训练阶段显示的文字描述（如 "Training Loss"） */
  title?: string
}

export function TrainingChart({ epochs = [], loss = [], accuracy, title }: TrainingChartProps) {
  if (!epochs.length || !loss.length) {
    return (
      <div className="flex items-center justify-center h-[200px] text-slate-400 text-sm">
        {title ? systemText(title, "暂无训练曲线数据") : "暂无训练曲线数据"}
      </div>
    )
  }

  const data = epochs.map((e, i) => ({
    epoch: e,
    loss: loss[i] ?? null,
    accuracy: accuracy?.[i] ?? null,
  }))

  return (
    <ResponsiveContainer width="100%" height={200}>
      <LineChart data={data}>
        <XAxis
          dataKey="epoch"
          stroke="#475569"
          tick={{ fill: "#64748b", fontSize: 11 }}
          axisLine={{ stroke: "#334155" }}
        />
        <YAxis stroke="#475569" tick={{ fill: "#64748b", fontSize: 11 }} axisLine={{ stroke: "#334155" }} />
        <Tooltip
          labelFormatter={(value) => `训练轮次 ${value}`}
          contentStyle={{
            backgroundColor: "rgba(15, 23, 42, 0.95)",
            border: "1px solid rgba(255, 255, 255, 0.1)",
            borderRadius: "8px",
            backdropFilter: "blur(8px)",
          }}
          labelStyle={{ color: "#cbd5e1", fontSize: 11 }}
          itemStyle={{ fontSize: 11 }}
        />
        <Legend wrapperStyle={{ color: "#94a3b8", fontSize: 11 }} iconSize={10} />
        <Line
          type="monotone"
          dataKey="loss"
          stroke="#06b6d4"
          strokeWidth={1.5}
          name="损失"
          dot={false}
          connectNulls
        />
        {(accuracy ?? []).some((a) => a !== null) && (
          <Line
            type="monotone"
            dataKey="accuracy"
            stroke="#ec4899"
            strokeWidth={1.5}
            name="准确率"
            dot={false}
            connectNulls
          />
        )}
      </LineChart>
    </ResponsiveContainer>
  )
}
