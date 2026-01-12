"use client"

import { useState } from "react"
import { ChevronRight, ChevronDown } from "lucide-react"

export interface TreeNode {
  id: string
  name: string
  children?: TreeNode[]
  value?: number
  color?: string
}

interface HierarchyTreeProps {
  data: TreeNode
  width?: number
  height?: number
}

export function HierarchyTree({ data, width = 800, height = 400 }: HierarchyTreeProps) {
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set([data.id]))
  const [selectedNode, setSelectedNode] = useState<string | null>(null)

  const toggleNode = (nodeId: string) => {
    setExpandedNodes((prev) => {
      const next = new Set(prev)
      if (next.has(nodeId)) {
        next.delete(nodeId)
      } else {
        next.add(nodeId)
      }
      return next
    })
  }

  const renderNode = (
    node: TreeNode,
    x: number,
    y: number,
    level: number = 0,
    index: number = 0,
    totalSiblings: number = 1
  ): JSX.Element[] => {
    const isExpanded = expandedNodes.has(node.id)
    const isSelected = selectedNode === node.id
    const hasChildren = node.children && node.children.length > 0
    const nodeWidth = 120
    const nodeHeight = 40
    const horizontalSpacing = 150
    const verticalSpacing = 80

    const elements: JSX.Element[] = []

    // 计算子节点的位置
    if (hasChildren && isExpanded) {
      const childCount = node.children!.length
      const childY = y + verticalSpacing
      const startX = x - ((childCount - 1) * horizontalSpacing) / 2

      node.children!.forEach((child, childIndex) => {
        const childX = startX + childIndex * horizontalSpacing

        // 绘制连接线
        elements.push(
          <line
            key={`line-${child.id}`}
            x1={x}
            y1={y + nodeHeight / 2}
            x2={childX}
            y2={childY - nodeHeight / 2}
            stroke="hsl(var(--border))"
            strokeWidth={1.5}
          />
        )

        // 递归渲染子节点
        const childElements = renderNode(child, childX, childY, level + 1, childIndex, childCount)
        elements.push(...childElements)
      })
    }

    // 渲染当前节点
    const nodeColor = node.color || (isSelected ? "hsl(var(--primary))" : "hsl(var(--muted))")
    
    elements.push(
      <g key={node.id}>
        {/* 节点矩形 */}
        <rect
          x={x - nodeWidth / 2}
          y={y - nodeHeight / 2}
          width={nodeWidth}
          height={nodeHeight}
          fill={nodeColor}
          stroke={isSelected ? "hsl(var(--primary))" : "hsl(var(--border))"}
          strokeWidth={isSelected ? 2 : 1}
          rx={6}
          className="cursor-pointer transition-all hover:opacity-80"
          onClick={() => {
            if (hasChildren) {
              toggleNode(node.id)
            }
            setSelectedNode(node.id)
          }}
        />
        
        {/* 节点文本 */}
        <text
          x={x}
          y={y}
          textAnchor="middle"
          dominantBaseline="middle"
          className="text-xs font-medium fill-foreground pointer-events-none select-none"
        >
          {node.name}
        </text>

        {/* 展开/折叠图标 */}
        {hasChildren && (
          <foreignObject
            x={x + nodeWidth / 2 - 16}
            y={y - nodeHeight / 2 + 2}
            width={16}
            height={16}
          >
            <div className="flex items-center justify-center">
              {isExpanded ? (
                <ChevronDown className="w-4 h-4 text-foreground" />
              ) : (
                <ChevronRight className="w-4 h-4 text-foreground" />
              )}
            </div>
          </foreignObject>
        )}

        {/* 节点值（如果有） */}
        {node.value !== undefined && (
          <text
            x={x}
            y={y + nodeHeight / 2 + 12}
            textAnchor="middle"
            className="text-[10px] fill-muted-foreground pointer-events-none"
          >
            {node.value}
          </text>
        )}
      </g>
    )

    return elements
  }

  // 计算根节点位置（居中）
  const rootX = width / 2
  const rootY = 50

  return (
    <div className="w-full h-full overflow-auto">
      <svg width={width} height={height} className="border rounded-lg bg-background">
        {renderNode(data, rootX, rootY)}
      </svg>
    </div>
  )
}

// 生成示例数据
export function generateMockHierarchyData(): TreeNode {
  return {
    id: "root",
    name: "主题根节点",
    children: [
      {
        id: "topic-1",
        name: "政策改革",
        value: 45,
        children: [
          { id: "topic-1-1", name: "立法框架", value: 28 },
          { id: "topic-1-2", name: "执行机制", value: 17 },
        ],
      },
      {
        id: "topic-2",
        name: "社会治理",
        value: 38,
        children: [
          { id: "topic-2-1", name: "公共服务", value: 22 },
          { id: "topic-2-2", name: "民生保障", value: 16 },
        ],
      },
      {
        id: "topic-3",
        name: "数字化转型",
        value: 32,
        children: [
          { id: "topic-3-1", name: "创新驱动", value: 18 },
          { id: "topic-3-2", name: "技术应用", value: 14 },
        ],
      },
    ],
  }
}
