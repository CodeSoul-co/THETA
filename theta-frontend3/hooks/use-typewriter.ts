"use client"

import { useState, useEffect, useCallback } from 'react'

interface UseTypewriterOptions {
  text: string
  speed?: number  // 每个字符的延迟（毫秒）
  onComplete?: () => void
}

export function useTypewriter({ text, speed = 20, onComplete }: UseTypewriterOptions) {
  const [displayedText, setDisplayedText] = useState('')
  const [isTyping, setIsTyping] = useState(true)
  const [currentIndex, setCurrentIndex] = useState(0)

  useEffect(() => {
    // 重置状态当文本改变时
    setDisplayedText('')
    setCurrentIndex(0)
    setIsTyping(true)
  }, [text])

  useEffect(() => {
    if (currentIndex < text.length) {
      const timer = setTimeout(() => {
        // 一次添加多个字符以加快速度（对于中文尤其重要）
        const charsToAdd = text.slice(currentIndex, currentIndex + 2)
        setDisplayedText(prev => prev + charsToAdd)
        setCurrentIndex(prev => prev + charsToAdd.length)
      }, speed)

      return () => clearTimeout(timer)
    } else {
      setIsTyping(false)
      onComplete?.()
    }
  }, [currentIndex, text, speed, onComplete])

  const skipToEnd = useCallback(() => {
    setDisplayedText(text)
    setCurrentIndex(text.length)
    setIsTyping(false)
  }, [text])

  return { displayedText, isTyping, skipToEnd }
}

export default useTypewriter
