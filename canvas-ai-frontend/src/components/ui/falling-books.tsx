import { useEffect, useRef } from 'react'

interface FallingItem {
  x: number
  y: number
  rotation: number
  speed: number
  rotationSpeed: number
  size: number
  emoji: string
  opacity: number
}

const emojis = [
  '📕', '📗', '📘', '📙', '📔', '📖', '📚',
  '✏️', '📝', '✒️', '🖊️', '🖍️',
  '📏', '📐', '🖇️', '📌', '📎',
  '🎓', '💡', '🧠',
]

export function FallingBooks() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const itemsRef = useRef<FallingItem[]>([])
  const animationRef = useRef<number>(0)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const resize = () => {
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
    }

    resize()

    const createItem = (startAbove: boolean): FallingItem => ({
      x: Math.random() * (canvas.width + 100) - 50,
      y: startAbove ? Math.random() * -canvas.height - 50 : Math.random() * canvas.height,
      rotation: Math.random() * 360,
      speed: 0.15 + Math.random() * 0.45,
      rotationSpeed: (Math.random() - 0.5) * 1.2,
      size: 24 + Math.random() * 24,
      emoji: emojis[Math.floor(Math.random() * emojis.length)],
      opacity: 0.15 + Math.random() * 0.2,
    })

    // Initialize items — scattered across the viewport
    itemsRef.current = Array.from({ length: 25 }, (_, i) => createItem(i < 10))

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      for (const item of itemsRef.current) {
        ctx.save()
        ctx.globalAlpha = item.opacity
        ctx.translate(item.x, item.y)
        ctx.rotate((item.rotation * Math.PI) / 180)
        ctx.font = `${item.size}px Arial`
        ctx.textAlign = 'center'
        ctx.textBaseline = 'middle'
        ctx.fillText(item.emoji, 0, 0)
        ctx.restore()

        item.y += item.speed
        item.rotation += item.rotationSpeed

        // Reset when off screen
        if (item.y > canvas.height + 60) {
          item.y = -60
          item.x = Math.random() * canvas.width
          item.emoji = emojis[Math.floor(Math.random() * emojis.length)]
          item.opacity = 0.15 + Math.random() * 0.2
        }
      }

      animationRef.current = requestAnimationFrame(animate)
    }

    animate()
    window.addEventListener('resize', resize)

    return () => {
      window.removeEventListener('resize', resize)
      cancelAnimationFrame(animationRef.current)
    }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      className="fixed top-0 left-0 w-full h-full pointer-events-none"
      style={{ zIndex: 1 }}
    />
  )
}
