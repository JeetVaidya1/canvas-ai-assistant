import { useRef } from 'react'
import type { KeyboardEvent } from 'react'
import { cn } from '@/lib/utils'
import type { DifficultyOption } from './types'

interface DifficultyTilesProps<T extends string> {
  options: readonly DifficultyOption<T>[]
  value: T
  onChange: (value: T) => void
  /** Accessible group label. */
  label?: string
  /** Grid column classes (e.g. "grid-cols-3"). */
  className?: string
}

/**
 * Tactile difficulty tiles with real radiogroup semantics: roving tabindex,
 * arrow-key navigation, Home/End, and visible focus rings.
 */
export function DifficultyTiles<T extends string>({
  options,
  value,
  onChange,
  label = 'Difficulty',
  className,
}: DifficultyTilesProps<T>) {
  const refs = useRef<(HTMLButtonElement | null)[]>([])

  const select = (index: number) => {
    const option = options[index]
    if (!option) return
    onChange(option.value)
    refs.current[index]?.focus()
  }

  const onKeyDown = (e: KeyboardEvent<HTMLButtonElement>, index: number) => {
    switch (e.key) {
      case 'ArrowRight':
      case 'ArrowDown':
        e.preventDefault()
        select((index + 1) % options.length)
        break
      case 'ArrowLeft':
      case 'ArrowUp':
        e.preventDefault()
        select((index - 1 + options.length) % options.length)
        break
      case 'Home':
        e.preventDefault()
        select(0)
        break
      case 'End':
        e.preventDefault()
        select(options.length - 1)
        break
    }
  }

  return (
    <div role="radiogroup" aria-label={label} className={cn('grid gap-2.5', className)}>
      {options.map((d, i) => {
        const active = value === d.value
        return (
          <button
            key={d.value}
            ref={(el) => {
              refs.current[i] = el
            }}
            type="button"
            role="radio"
            aria-checked={active}
            tabIndex={active ? 0 : -1}
            onClick={() => onChange(d.value)}
            onKeyDown={(e) => onKeyDown(e, i)}
            className={cn(
              'rounded-xl border px-3 py-4 text-center transition-all outline-none',
              'focus-visible:ring-2 focus-visible:ring-accent/60 focus-visible:ring-offset-2 focus-visible:ring-offset-paper',
              active
                ? 'border-accent bg-accent-wash ring-1 ring-inset ring-accent/20'
                : 'border-line bg-surface hover:border-line-strong hover:bg-surface-hover',
            )}
          >
            <div className={cn('text-sm font-semibold', active ? 'text-accent-deep' : 'text-ink')}>
              {d.label}
            </div>
            <div className={cn('mt-0.5 text-[11px]', active ? 'text-accent-deep/70' : 'text-ink-faint')}>{d.hint}</div>
          </button>
        )
      })}
    </div>
  )
}
