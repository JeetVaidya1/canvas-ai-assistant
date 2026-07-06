import type { ReactNode } from 'react'
import { AlertTriangle, RefreshCw } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/Button'

interface EmptyStateProps {
  icon?: ReactNode
  title: string
  description?: string
  action?: ReactNode
  className?: string
}

/**
 * Canonical empty state — used when a list/section legitimately has no data
 * yet. Always tells the user what to do next.
 */
export function EmptyState({ icon, title, description, action, className }: EmptyStateProps) {
  return (
    <div className={cn('flex flex-col items-center justify-center text-center py-12 px-6', className)}>
      {icon && (
        <div className="w-12 h-12 rounded-xl bg-paper-deep border border-line flex items-center justify-center mb-4 text-ink-faint [&>svg]:w-5 [&>svg]:h-5">
          {icon}
        </div>
      )}
      <h3 className="text-sm font-semibold text-ink">{title}</h3>
      {description && <p className="text-sm text-ink-soft mt-1.5 max-w-sm leading-relaxed">{description}</p>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  )
}

interface ErrorStateProps {
  title?: string
  /** Friendly explanation — never raw error text from the server. */
  description?: string
  onRetry?: () => void
  retrying?: boolean
  compact?: boolean
  className?: string
}

/**
 * Canonical error state with retry. Failures must never render as a spinner
 * that ends in nothing — use this instead.
 */
export function ErrorState({
  title = 'Something went wrong',
  description = 'We couldn’t load this. Check your connection and try again.',
  onRetry,
  retrying = false,
  compact = false,
  className,
}: ErrorStateProps) {
  if (compact) {
    return (
      <div className={cn('flex items-center gap-3 rounded-lg border border-danger/25 bg-danger-wash px-3.5 py-2.5', className)} role="alert">
        <AlertTriangle className="w-4 h-4 text-danger flex-shrink-0" />
        <p className="text-sm text-ink flex-1 min-w-0">{title}</p>
        {onRetry && (
          <Button variant="ghost" size="sm" onClick={onRetry} loading={retrying} leftIcon={<RefreshCw className="w-3.5 h-3.5" />}>
            Retry
          </Button>
        )}
      </div>
    )
  }
  return (
    <div className={cn('flex flex-col items-center justify-center text-center py-12 px-6', className)} role="alert">
      <div className="w-12 h-12 rounded-xl bg-danger-wash border border-danger/25 flex items-center justify-center mb-4">
        <AlertTriangle className="w-5 h-5 text-danger" />
      </div>
      <h3 className="text-sm font-semibold text-ink">{title}</h3>
      <p className="text-sm text-ink-soft mt-1.5 max-w-sm leading-relaxed">{description}</p>
      {onRetry && (
        <Button variant="secondary" size="sm" onClick={onRetry} loading={retrying} className="mt-4" leftIcon={<RefreshCw className="w-3.5 h-3.5" />}>
          Try again
        </Button>
      )}
    </div>
  )
}
