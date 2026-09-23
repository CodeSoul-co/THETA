"use client"

import { useEffect, useRef } from "react"

export interface ParticlesBgProps {
  /** CSS z-index (default -1, behind content) */
  zIndex?: number
  /** Canvas opacity 0–1 (default 0.35) */
  opacity?: number
  /** Line/dot color as "r,g,b" (default blue) */
  color?: string
  /** Number of floating particles (default 48) */
  count?: number
  /** Optional className for the wrapper */
  className?: string
}

export function ParticlesBg({
  zIndex = -1,
  opacity = 0.35,
  color = "59, 130, 246",
  count = 48,
  className = "",
}: ParticlesBgProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const rafRef = useRef<number>(0)
  const mouseRef = useRef<{ x: number | null; y: number | null }>({ x: null, y: null })

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const canvasElement = canvas

    const ctx = canvas.getContext("2d")
    if (!ctx) return
    const context = ctx

    let width = window.innerWidth
    let height = window.innerHeight
    let pixelRatio = Math.min(window.devicePixelRatio || 1, 1.5)
    let isRunning = !document.hidden
    let lastFrameAt = 0
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches

    function setSize() {
      width = window.innerWidth
      height = window.innerHeight
      pixelRatio = Math.min(window.devicePixelRatio || 1, 1.5)
      canvasElement.width = Math.round(width * pixelRatio)
      canvasElement.height = Math.round(height * pixelRatio)
      context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0)
    }

    setSize()
    const onResize = () => setSize()
    window.addEventListener("resize", onResize)

    const onMouseMove = (e: MouseEvent) => {
      mouseRef.current.x = e.clientX
      mouseRef.current.y = e.clientY
    }
    const onMouseOut = () => {
      mouseRef.current.x = null
      mouseRef.current.y = null
    }
    window.addEventListener("mousemove", onMouseMove, { passive: true })
    window.addEventListener("mouseout", onMouseOut)

    type Particle = { x: number; y: number; xa: number; ya: number; max: number }
    type PointerParticle = Omit<Particle, "x" | "y"> & { x: number | null; y: number | null }
    const mouse: PointerParticle = {
      x: null,
      y: null,
      xa: 0,
      ya: 0,
      max: 20000,
    }

    const particleCount = window.innerWidth < 768 ? Math.min(count, 24) : count
    const particles: Particle[] = []
    for (let i = 0; i < particleCount; i++) {
      particles.push({
        x: Math.random() * width,
        y: Math.random() * height,
        xa: 2 * Math.random() - 1,
        ya: 2 * Math.random() - 1,
        max: 6000,
      })
    }

    const list = [...particles, mouse]

    function draw(timestamp = 0) {
      if (!isRunning) return
      rafRef.current = requestAnimationFrame(draw)
      if (timestamp - lastFrameAt < 1000 / 30) return
      lastFrameAt = timestamp

      context.clearRect(0, 0, width, height)

      mouse.x = mouseRef.current.x
      mouse.y = mouseRef.current.y

      particles.forEach((p, i) => {
        p.x += p.xa
        p.y += p.ya
        if (p.x > width || p.x < 0) p.xa *= -1
        if (p.y > height || p.y < 0) p.ya *= -1
        context.fillStyle = `rgba(${color}, 0.6)`
        context.fillRect(p.x - 0.5, p.y - 0.5, 1, 1)

        for (let j = i + 1; j < list.length; j++) {
          const n = list[j]
          if (n.x == null || n.y == null) continue
          const dx = p.x - n.x
          const dy = p.y - n.y
          const distSq = dx * dx + dy * dy
          if (distSq >= n.max) continue

          const t = (n.max - distSq) / n.max
          context.beginPath()
          context.lineWidth = t / 2
          context.strokeStyle = `rgba(${color}, ${t + 0.2})`
          context.moveTo(p.x, p.y)
          context.lineTo(n.x, n.y)
          context.stroke()

          if (n === mouse && distSq >= n.max / 2) {
            p.x -= 0.03 * dx
            p.y -= 0.03 * dy
          }
        }
      })

    }

    const start = () => {
      isRunning = true
      cancelAnimationFrame(rafRef.current)
      if (reduceMotion) {
        draw()
        isRunning = false
        return
      }
      rafRef.current = requestAnimationFrame(draw)
    }
    const onVisibilityChange = () => {
      if (document.hidden) {
        isRunning = false
        cancelAnimationFrame(rafRef.current)
      } else {
        start()
      }
    }
    document.addEventListener("visibilitychange", onVisibilityChange)
    const t = window.setTimeout(start, 100)

    return () => {
      clearTimeout(t)
      isRunning = false
      cancelAnimationFrame(rafRef.current)
      window.removeEventListener("resize", onResize)
      window.removeEventListener("mousemove", onMouseMove)
      window.removeEventListener("mouseout", onMouseOut)
      document.removeEventListener("visibilitychange", onVisibilityChange)
    }
  }, [color, count])

  return (
    <div
      className={className}
      style={{
        position: "fixed",
        inset: 0,
        zIndex,
        opacity,
        pointerEvents: "none",
      }}
      aria-hidden
    >
      <canvas
        ref={canvasRef}
        style={{ display: "block", width: "100%", height: "100%" }}
      />
    </div>
  )
}
