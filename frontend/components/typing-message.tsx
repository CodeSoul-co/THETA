"use client"

import { useState, useEffect, useRef, useCallback } from 'react'
import { MarkdownRenderer } from './markdown-renderer'

interface TypingMessageProps {
  content: string
  isLatest: boolean  // 是否是最新的消息
  className?: string
  speed?: number     // 每个字符的间隔毫秒数
}

export function TypingMessage({ content, isLatest, className, speed = 50 }: TypingMessageProps) {
  const [displayedLength, setDisplayedLength] = useState(0)
  const [isTyping, setIsTyping] = useState(false)
  const contentLengthRef = useRef(0)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const clearTimer = useCallback(() => {
    if (timerRef.current !== null) {
      clearTimeout(timerRef.current)
      timerRef.current = null
    }
  }, [])

  // 当 content 或 isLatest 变化时，重置打字状态
  // 依赖包含 isLatest 确保即使 content 不变但 isLatest 变成 true 也会重新启动
  useEffect(() => {
    if (!isLatest) {
      // 非最新消息，直接显示完整内容
      setDisplayedLength(content.length)
      setIsTyping(false)
      clearTimer()
      return
    }

    // 最新消息：从头开始打字
    contentLengthRef.current = content.length
    setDisplayedLength(0)
    setIsTyping(true)
    clearTimer()

    // 开始打字动画
    const tick = () => {
      setDisplayedLength((prev) => {
        const next = prev + 1
        if (next >= content.length) {
          setIsTyping(false)
          clearTimer()
          return content.length
        }
        // 继续打字
        timerRef.current = setTimeout(tick, speed)
        return next
      })
    }

    timerRef.current = setTimeout(tick, speed)

    return () => {
      clearTimer()
    }
  }, [content, isLatest, speed, clearTimer])

  // 组件卸载时清理
  useEffect(() => {
    return () => {
      clearTimer()
    }
  }, [clearTimer])

  // 点击跳过打字效果
  const handleClick = useCallback(() => {
    if (isTyping) {
      setDisplayedLength(content.length)
      setIsTyping(false)
      clearTimer()
    }
  }, [isTyping, content.length, clearTimer])

  const displayedText = content.slice(0, displayedLength)

  return (
    <div onClick={handleClick} className={isTyping ? 'cursor-pointer' : ''}>
      <MarkdownRenderer content={displayedText} className={className} />
      {isTyping && (
        <span className="inline-block w-[2px] h-[1em] bg-blue-500 ml-0.5 align-middle animate-pulse" />
      )}
    </div>
  )
}

export default TypingMessage