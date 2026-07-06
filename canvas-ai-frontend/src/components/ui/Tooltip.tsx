import { useState } from 'react'
import type { ReactNode } from 'react'
import { cn } from '@/lib/utils'

interface TooltipProps {
  content: ReactNode
  side?: 'top' | 'bottom' | 'left' | 'right'
  /** Delay before showing, ms. */
  delay?: number
  className?: string
  children: ReactNode
}

const sideClass = {
  top: 'bottom-full left-1/2 -translate-x-1/2 mb-1.5',
  bottom: 'top-full left-1/2 -translate-x-1/2 mt-1.5',
  left: 'right-full top-1/2 -translate-y-1/2 mr-1.5',
  right: 'left-full top-1/2 -translate-y-1/2 ml-1.5',
}

/**
 * Lightweight tooltip: ink chip on paper (dark chip for contrast — the one
 * deliberate dark element in the system, like pencil on a sticky label).
 */
export function Tooltip({ content, side = 'top', delay = 300, className, children }: TooltipProps) {
  const [visible, setVisible] = useState(false)
  const [timer, setTimer] = useState<number | null>(null)

  const show = () => setTimer(window.setTimeout(() => setVisible(true), delay))
  const hide = () => {
    if (timer) window.clearTimeout(timer)
    setTimer(null)
    setVisible(false)
  }

  return (
    <span className="relative inline-flex" onMouseEnter={show} onMouseLeave={hide} onFocus={show} onBlur={hide}>
      {children}
      {visible && (
        <span
          role="tooltip"
          className={cn(
            'absolute z-50 px-2 py-1 rounded-md bg-ink text-paper text-[11px] whitespace-nowrap elev-2 pointer-events-none',
            sideClass[side],
            className,
          )}
        >
          {content}
        </span>
      )}
    </span>
  )
}
