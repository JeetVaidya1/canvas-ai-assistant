import { useEffect, useRef, useState } from 'react'
import { animate, useInView } from 'motion/react'

interface CountUpProps {
  from?: number
  to: number
  duration?: number
  className?: string
  separator?: string
  decimals?: number
}

export default function CountUp({
  from = 0,
  to,
  duration = 1.5,
  className = '',
  separator = '',
  decimals = 0,
}: CountUpProps) {
  const ref = useRef<HTMLSpanElement>(null)
  const inView = useInView(ref, { once: true })
  const [display, setDisplay] = useState(from.toFixed(decimals))

  useEffect(() => {
    if (!inView) return

    const controls = animate(from, to, {
      duration,
      ease: 'easeOut',
      onUpdate(value) {
        let formatted = value.toFixed(decimals)
        if (separator) {
          const parts = formatted.split('.')
          parts[0] = parts[0].replace(/\B(?=(\d{3})+(?!\d))/g, separator)
          formatted = parts.join('.')
        }
        setDisplay(formatted)
      },
    })

    return () => controls.stop()
  }, [inView, from, to, duration, separator, decimals])

  return (
    <span ref={ref} className={className}>
      {display}
    </span>
  )
}
