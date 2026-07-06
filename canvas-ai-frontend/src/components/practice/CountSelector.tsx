import { useRef } from 'react'
import type { KeyboardEvent } from 'react'
import { cn } from '@/lib/utils'

interface CountSelectorProps {
  counts: readonly number[]
  value: number
  onChange: (value: number) => void
  /** Accessible group label. */
  label: string
}

/**
 * Segmented question/problem-count control with radiogroup semantics
 * (roving tabindex + arrow keys), styled to match the setup tiles.
 */
export function CountSelector({ counts, value, onChange, label }: CountSelectorProps) {
  const refs = useRef<(HTMLButtonElement | null)[]>([])

  const select = (index: number) => {
    const count = counts[index]
    if (count === undefined) return
    onChange(count)
    refs.current[index]?.focus()
  }

  const onKeyDown = (e: KeyboardEvent<HTMLButtonElement>, index: number) => {
    switch (e.key) {
      case 'ArrowRight':
      case 'ArrowDown':
        e.preventDefault()
        select((index + 1) % counts.length)
        break
      case 'ArrowLeft':
      case 'ArrowUp':
        e.preventDefault()
        select((index - 1 + counts.length) % counts.length)
        break
      case 'Home':
        e.preventDefault()
        select(0)
        break
      case 'End':
        e.preventDefault()
        select(counts.length - 1)
        break
    }
  }

  return (
    <div
      role="radiogroup"
      aria-label={label}
      className="flex gap-1.5 rounded-2xl border border-white/10 bg-white/[0.03] p-1.5"
    >
      {counts.map((c, i) => {
        const active = value === c
        return (
          <button
            key={c}
            ref={(el) => {
              refs.current[i] = el
            }}
            type="button"
            role="radio"
            aria-checked={active}
            tabIndex={active ? 0 : -1}
            onClick={() => onChange(c)}
            onKeyDown={(e) => onKeyDown(e, i)}
            className={cn(
              'flex-1 rounded-xl py-2.5 text-sm font-semibold transition-all outline-none',
              'focus-visible:ring-2 focus-visible:ring-cyan-400/70 focus-visible:ring-offset-2 focus-visible:ring-offset-[#0a0c14]',
              active
                ? 'bg-gradient-brand-soft text-cyan-100 ring-1 ring-inset ring-cyan-400/30'
                : 'text-zinc-400 hover:bg-white/[0.05] hover:text-zinc-200',
            )}
          >
            {c}
          </button>
        )
      })}
    </div>
  )
}
