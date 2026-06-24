import type { HTMLAttributes } from 'react'
import { cn } from '@/lib/utils'

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  /** Adds hover lift + cyan border glow (use for clickable cards). */
  interactive?: boolean
  /** Adds a subtle cyan gradient hairline across the top. */
  accent?: boolean
  padding?: 'none' | 'sm' | 'md' | 'lg'
}

const paddings = {
  none: '',
  sm: 'p-4',
  md: 'p-5',
  lg: 'p-6',
}

export function Card({
  interactive = false,
  accent = false,
  padding = 'md',
  className,
  children,
  ...rest
}: CardProps) {
  return (
    <div
      className={cn(
        'card-surface rounded-xl',
        interactive && 'card-interactive cursor-pointer',
        accent && 'accent-top',
        paddings[padding],
        className,
      )}
      {...rest}
    >
      {children}
    </div>
  )
}

/** Consistent page header: gradient eyebrow + title + optional subtitle/actions. */
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
      <div>
        {eyebrow && (
          <p className="text-xs font-semibold uppercase tracking-widest text-gradient-brand mb-1.5">
            {eyebrow}
          </p>
        )}
        <h1 className="text-2xl font-semibold text-zinc-50 tracking-tight">{title}</h1>
        {subtitle && <p className="text-sm text-zinc-500 mt-1">{subtitle}</p>}
      </div>
      {actions && <div className="flex items-center gap-2 flex-shrink-0">{actions}</div>}
    </div>
  )
}
