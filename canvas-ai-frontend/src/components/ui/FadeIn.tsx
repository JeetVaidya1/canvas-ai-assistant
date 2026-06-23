import { useEffect, useRef, useState, type ReactNode, type CSSProperties } from 'react'

interface FadeInProps {
  children: ReactNode
  className?: string
  direction?: 'up' | 'down' | 'left' | 'right' | 'none'
  distance?: number
  duration?: number
  delay?: number
  once?: boolean
}

export default function FadeIn({
  children,
  className = '',
  direction = 'up',
  distance = 20,
  duration = 0.6,
  delay = 0,
  once = true,
}: FadeInProps) {
  const ref = useRef<HTMLDivElement>(null)
  const [isVisible, setIsVisible] = useState(false)

  useEffect(() => {
    const el = ref.current
    if (!el) return

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true)
          if (once) observer.unobserve(el)
        } else if (!once) {
          setIsVisible(false)
        }
      },
      { threshold: 0.1 }
    )

    observer.observe(el)
    return () => observer.disconnect()
  }, [once])

  const getTransform = () => {
    if (isVisible) return 'translate3d(0, 0, 0)'
    switch (direction) {
      case 'up': return `translate3d(0, ${distance}px, 0)`
      case 'down': return `translate3d(0, -${distance}px, 0)`
      case 'left': return `translate3d(${distance}px, 0, 0)`
      case 'right': return `translate3d(-${distance}px, 0, 0)`
      case 'none': return 'translate3d(0, 0, 0)'
    }
  }

  const style: CSSProperties = {
    opacity: isVisible ? 1 : 0,
    transform: getTransform(),
    transition: `opacity ${duration}s ease ${delay}s, transform ${duration}s ease ${delay}s`,
    willChange: 'opacity, transform',
  }

  return (
    <div ref={ref} className={className} style={style}>
      {children}
    </div>
  )
}
