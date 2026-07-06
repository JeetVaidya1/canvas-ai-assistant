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
              'rounded-2xl border px-3 py-4 text-center transition-all outline-none',
              'focus-visible:ring-2 focus-visible:ring-cyan-400/70 focus-visible:ring-offset-2 focus-visible:ring-offset-[#0a0c14]',
              active
                ? 'border-cyan-400/50 bg-gradient-brand-soft ring-2 ring-cyan-400/25 shadow-[0_8px_24px_-12px_rgba(34,211,238,0.5)]'
                : 'border-white/10 bg-white/[0.03] hover:border-cyan-400/30 hover:bg-white/[0.05]',
            )}
          >
            <div className={cn('text-sm font-semibold', active ? 'text-cyan-200' : 'text-zinc-200')}>
              {d.label}
            </div>
            <div className="mt-0.5 text-[11px] text-zinc-500">{d.hint}</div>
          </button>
        )
      })}
    </div>
  )
}
