"use client"

import { useState, useEffect, useRef } from "react"

type Phase = "typing" | "holding" | "deleting"

interface UseCyclingTypewriterOptions {
  /** 循环展示的文案列表（每条不同） */
  phrases: string[]
  /** 打字时每个字符的间隔（毫秒） */
  typingSpeed?: number
  /** 删除时每个字符的间隔（毫秒） */
  deleteSpeed?: number
  /** 每句打完后的停留时间（毫秒） */
  holdDuration?: number
  /** 是否循环，默认 true */
  loop?: boolean
}

/**
 * 打字机循环效果：逐字打出 → 停留 → 逐字删除 → 下一条 → 再打出，循环。
 */
export function useCyclingTypewriter({
  phrases,
  typingSpeed = 80,
  deleteSpeed = 50,
  holdDuration = 1800,
  loop = true,
}: UseCyclingTypewriterOptions) {
  const [phraseIndex, setPhraseIndex] = useState(0)
  const [charIndex, setCharIndex] = useState(0)
  const [phase, setPhase] = useState<Phase>("typing")
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const currentPhrase = phrases[phraseIndex] ?? ""
  const displayedText = currentPhrase.slice(0, charIndex)

  useEffect(() => {
    if (phrases.length === 0) return

    const clearTimer = () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current)
        timerRef.current = null
      }
    }

    if (phase === "typing") {
      if (charIndex >= currentPhrase.length) {
        setPhase("holding")
        return () => clearTimer()
      }
      timerRef.current = setTimeout(() => {
        setCharIndex((prev) => Math.min(prev + 1, currentPhrase.length))
      }, typingSpeed)
      return () => clearTimer()
    }

    if (phase === "holding") {
      timerRef.current = setTimeout(() => setPhase("deleting"), holdDuration)
      return () => clearTimer()
    }

    // phase === "deleting"
    if (charIndex <= 0) {
      const nextIndex = loop ? (phraseIndex + 1) % phrases.length : Math.min(phraseIndex + 1, phrases.length - 1)
      setPhraseIndex(nextIndex)
      setCharIndex(0)
      setPhase("typing")
      return () => clearTimer()
    }
    timerRef.current = setTimeout(() => {
      setCharIndex((prev) => Math.max(0, prev - 1))
    }, deleteSpeed)
    return () => clearTimer()
  }, [phrases.length, phraseIndex, charIndex, phase, currentPhrase.length, typingSpeed, deleteSpeed, holdDuration, loop])

  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current)
    }
  }, [])

  const isTyping = phase === "typing" || phase === "deleting"

  return { displayedText, isTyping }
}

export default useCyclingTypewriter
