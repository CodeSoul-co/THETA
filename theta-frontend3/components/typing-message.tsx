"use client"

import { useState, useEffect, useRef } from 'react'
import { MarkdownRenderer } from './markdown-renderer'

interface TypingMessageProps {
  content: string
  isLatest: boolean  // 是否是最新的消息
  className?: string
  speed?: number
}

export function TypingMessage({ content, isLatest, className, speed = 15 }: TypingMessageProps) {
  const [displayedText, setDisplayedText] = useState(isLatest ? '' : content)
  const [isTyping, setIsTyping] = useState(isLatest)
  const indexRef = useRef(0)
  const contentRef = useRef(content)

  useEffect(() => {
    // 如果不是最新消息，直接显示完整内容
    if (!isLatest) {
      setDisplayedText(content)
      setIsTyping(false)
      return
    }

    // 如果内容改变了，重置
    if (contentRef.current !== content) {
      contentRef.current = content
      indexRef.current = 0
      setDisplayedText('')
      setIsTyping(true)
    }

    if (indexRef.current >= content.length) {
      setIsTyping(false)
      return
    }

    const timer = setInterval(() => {
      if (indexRef.current < content.length) {
        // 一次添加 1-3 个字符，让速度更自然
        const charsToAdd = Math.min(3, content.length - indexRef.current)
        const newText = content.slice(0, indexRef.current + charsToAdd)
        setDisplayedText(newText)
        indexRef.current += charsToAdd
      } else {
        setIsTyping(false)
        clearInterval(timer)
      }
    }, speed)

    return () => clearInterval(timer)
  }, [content, isLatest, speed])

  // 点击跳过打字效果
  const handleClick = () => {
    if (isTyping) {
      setDisplayedText(content)
      indexRef.current = content.length
      setIsTyping(false)
    }
  }

  return (
    <div onClick={handleClick} className={isTyping ? 'cursor-pointer' : ''}>
      <MarkdownRenderer content={displayedText} className={className} />
      {isTyping && (
        <span className="inline-block w-2 h-4 bg-blue-500 animate-pulse ml-0.5 align-middle" />
      )}
    </div>
  )
}

export default TypingMessage
