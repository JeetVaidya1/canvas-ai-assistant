import { RefreshCw } from 'lucide-react'
import { Select } from '@/components/ui/Select'
import type { SelectOption } from '@/components/ui/Select'
import ErrorInline from '@/components/shared/ErrorInline'
import Skeleton from '@/components/shared/Skeleton'
import { cn } from '@/lib/utils'
import { FieldLabel } from './SetupShell'

interface TopicFieldProps {
  options: SelectOption[]
  value: string
  onChange: (value: string) => void
  /** Any fetch in flight — drives the Refresh spinner and disables the select. */
  loading: boolean
  /** First load with no data yet — show a skeleton instead of the select. */
  pending: boolean
  /** Friendly failure message; renders as an inline error with Retry. */
  error: string | null
  onRetry: () => void
  disabled?: boolean
  /** Muted hint under the select (suppressed while an error is showing). */
  helper?: string | null
  ariaLabel: string
}

/** Topic picker: label + refresh action, skeleton on first load, inline error with retry. */
export function TopicField({
  options,
  value,
  onChange,
  loading,
  pending,
  error,
  onRetry,
  disabled = false,
  helper = null,
  ariaLabel,
}: TopicFieldProps) {
  return (
    <div>
      <FieldLabel
        action={
          <button
            type="button"
            onClick={onRetry}
            disabled={loading}
            className="inline-flex items-center gap-1 text-xs text-accent transition-colors hover:text-accent-deep disabled:opacity-50"
            aria-label="Refresh topics"
          >
            <RefreshCw className={cn('h-3 w-3', loading && 'animate-spin')} />
            Refresh
          </button>
        }
      >
        Topic
      </FieldLabel>
      {pending ? (
        <Skeleton className="h-[42px] w-full" />
      ) : (
        <Select
          value={value}
          onChange={onChange}
          options={options}
          disabled={loading || disabled}
          ariaLabel={ariaLabel}
          placeholder={loading ? 'Loading topics…' : 'Select topic'}
        />
      )}
      {helper && !error && <p className="mt-2 text-center text-xs text-ink-faint">{helper}</p>}
      {error && <ErrorInline message={error} onRetry={onRetry} className="mt-2.5" />}
    </div>
  )
}
