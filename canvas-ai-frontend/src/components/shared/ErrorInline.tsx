import { AlertTriangle, RotateCcw } from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { cn } from '@/lib/utils'

interface ErrorInlineProps {
  message?: string
  onRetry?: () => void
  className?: string
}

/**
 * Small muted inline error panel with an optional Retry action.
 * Use wherever a query fails instead of silently rendering nothing.
 */
export default function ErrorInline({
  message = 'Something went wrong loading this.',
  onRetry,
  className,
}: ErrorInlineProps) {
  return (
    <div
      role="alert"
      className={cn(
        'flex items-center justify-between gap-3 rounded-xl border border-danger/25 bg-danger-wash px-4 py-3',
        className,
      )}
    >
      <div className="flex min-w-0 items-center gap-2.5">
        <AlertTriangle className="h-4 w-4 flex-shrink-0 text-danger" />
        <p className="text-sm text-ink">{message}</p>
      </div>
      {onRetry && (
        <Button
          variant="secondary"
          size="sm"
          onClick={onRetry}
          leftIcon={<RotateCcw className="h-3.5 w-3.5" />}
          className="flex-shrink-0"
        >
          Retry
        </Button>
      )}
    </div>
  )
}
