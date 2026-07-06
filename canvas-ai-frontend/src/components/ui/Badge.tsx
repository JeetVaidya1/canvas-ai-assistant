import type { HTMLAttributes, ReactNode } from 'react'
import { cn } from '@/lib/utils'

type Tone = 'neutral' | 'accent' | 'success' | 'warning' | 'danger' | 'info' | 'marker'

const tones: Record<Tone, string> = {
  neutral: 'text-ink-soft bg-paper-deep border-line',
  accent: 'text-accent-deep bg-accent-wash border-accent-line',
  success: 'text-success bg-success-wash border-success/25',
  warning: 'text-warning bg-warning-wash border-warning/25',
  danger: 'text-danger bg-danger-wash border-danger/25',
  info: 'text-info bg-info-wash border-info/25',
  /** Highlighter-marked label — the signature tone; use sparingly. */
  marker: 'text-ink bg-marker-soft border-marker',
}

interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  tone?: Tone
  icon?: ReactNode
}

/** Small status chip. One tone system app-wide — never hand-roll pill styles. */
export function Badge({ tone = 'neutral', icon, className, children, ...rest }: BadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 text-[11px] font-medium rounded-md border px-2 py-0.5 whitespace-nowrap',
        '[&>svg]:w-3 [&>svg]:h-3',
        tones[tone],
        className,
      )}
      {...rest}
    >
      {icon}
      {children}
    </span>
  )
}
