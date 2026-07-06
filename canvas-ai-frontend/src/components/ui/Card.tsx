import type { HTMLAttributes } from 'react'
import { cn } from '@/lib/utils'

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  /** Hover lift for clickable cards. */
  interactive?: boolean
  /** Kept for API compat — Paper & Ink cards have no accent hairline. */
  accent?: boolean
  /** Resting elevation. Default 1. */
  elevation?: 0 | 1 | 2 | 3
  padding?: 'none' | 'sm' | 'md' | 'lg'
}

const paddings = {
  none: '',
  sm: 'p-4',
  md: 'p-5',
  lg: 'p-6',
}

const elevations = {
  0: 'shadow-none',
  1: '',
  2: 'elev-2',
  3: 'elev-3',
}

/** White paper sheet: hairline border + the faintest lift. */
export function Card({
  interactive = false,
  // `accent` intentionally consumed and ignored — Paper & Ink cards have no
  // accent hairline, but callers still pass it (API compat).
  accent,
  elevation = 1,
  padding = 'md',
  className,
  children,
  ...rest
}: CardProps) {
  void accent
  return (
    <div
      className={cn(
        'card-surface rounded-xl',
        elevations[elevation],
        interactive && 'card-interactive',
        paddings[padding],
        className,
      )}
      {...rest}
    >
      {children}
    </div>
  )
}

/** Page header: serif display title + optional kicker/subtitle/actions. */
interface PageHeaderProps {
  eyebrow?: string
  title: string
  subtitle?: string
  actions?: React.ReactNode
  className?: string
}

export function PageHeader({ eyebrow, title, subtitle, actions, className }: PageHeaderProps) {
  return (
    <div className={cn('flex items-start justify-between gap-4', className)}>
      <div className="min-w-0">
        {eyebrow && (
          <p className="text-[11px] font-medium uppercase tracking-[0.14em] text-ink-faint mb-1.5">{eyebrow}</p>
        )}
        <h1 className="font-display text-[1.75rem] leading-tight font-semibold text-ink">{title}</h1>
        {subtitle && <p className="text-sm text-ink-soft mt-1.5">{subtitle}</p>}
      </div>
      {actions && <div className="flex items-center gap-2 flex-shrink-0">{actions}</div>}
    </div>
  )
}
