import { useRef } from 'react'
import type { KeyboardEvent } from 'react'
import { cn } from '@/lib/utils'
import type { QuizConfidence } from '@/lib/api'

const OPTIONS: readonly { value: QuizConfidence; label: string }[] = [
  { value: 'sure', label: 'Sure' },
  { value: 'thinkso', label: 'Think so' },
  { value: 'guessing', label: 'Guessing' },
]

interface ConfidenceControlProps {
  value: QuizConfidence | null
  onChange: (value: QuizConfidence | null) => void
  disabled?: boolean
}

/**
 * Compact three-way confidence tap shown between picking an option and
 * submitting. Optional — no default, and tapping the active segment clears it.
 * Radio-group semantics with arrow-key navigation.
 */
export function ConfidenceControl({ value, onChange, disabled }: ConfidenceControlProps) {
  const groupRef = useRef<HTMLDivElement>(null)

  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (disabled) return
    const delta =
      event.key === 'ArrowRight' || event.key === 'ArrowDown'
        ? 1
        : event.key === 'ArrowLeft' || event.key === 'ArrowUp'
          ? -1
          : 0
    if (!delta) return
    event.preventDefault()
    const current = OPTIONS.findIndex((o) => o.value === value)
    const next = OPTIONS[(current + delta + OPTIONS.length) % OPTIONS.length]
    onChange(next.value)
    const buttons = groupRef.current?.querySelectorAll<HTMLButtonElement>('button[role="radio"]')
    buttons?.[OPTIONS.indexOf(next)]?.focus()
  }

  return (
    <div className="flex flex-wrap items-center gap-2.5">
      <span id="confidence-label" className="text-xs text-ink-faint">
        How sure are you? <span className="text-ink-faint/80">(optional)</span>
      </span>
      <div
        ref={groupRef}
        role="radiogroup"
        aria-labelledby="confidence-label"
        onKeyDown={handleKeyDown}
        className="inline-flex rounded-lg border border-line bg-paper-deep p-0.5"
      >
        {OPTIONS.map((option, index) => {
          const active = value === option.value
          // Roving tabindex: the active segment is tabbable; with no selection
          // the first segment takes the tab stop.
          const tabbable = active || (value === null && index === 0)
          return (
            <button
              key={option.value}
              type="button"
              role="radio"
              aria-checked={active}
              tabIndex={tabbable ? 0 : -1}
              disabled={disabled}
              onClick={() => onChange(active ? null : option.value)}
              className={cn(
                'rounded-md px-2.5 py-1 text-xs font-medium transition-colors outline-none',
                'focus-visible:ring-2 focus-visible:ring-accent/60',
                'disabled:cursor-not-allowed disabled:opacity-45',
                active
                  ? 'bg-surface text-accent-deep border border-accent-line shadow-[0_1px_2px_rgba(33,31,26,0.06)]'
                  : 'border border-transparent text-ink-soft hover:text-ink',
              )}
            >
              {option.label}
            </button>
          )
        })}
      </div>
    </div>
  )
}
