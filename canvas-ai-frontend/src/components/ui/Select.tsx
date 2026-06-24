import { useState, useRef, useEffect } from 'react'
import type { KeyboardEvent } from 'react'
import { motion, AnimatePresence } from 'motion/react'
import { ChevronDown, Check } from 'lucide-react'
import { cn } from '@/lib/utils'

export interface SelectOption {
  value: string
  label: string
  hint?: string
}

interface SelectProps {
  value: string
  options: SelectOption[]
  onChange: (value: string) => void
  placeholder?: string
  className?: string
  disabled?: boolean
  ariaLabel?: string
}

/**
 * Custom select — built from scratch to replace the native <select>. Animated
 * popover, full keyboard navigation (↑/↓/Enter/Esc/Home/End), click-outside to
 * close, and proper listbox semantics.
 */
export function Select({ value, options, onChange, placeholder, className, disabled, ariaLabel }: SelectProps) {
  const [open, setOpen] = useState(false)
  const [active, setActive] = useState(0)
  const ref = useRef<HTMLDivElement>(null)
  const selected = options.find((o) => o.value === value)

  useEffect(() => {
    if (!open) return
    setActive(Math.max(0, options.findIndex((o) => o.value === value)))
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', onDoc)
    return () => document.removeEventListener('mousedown', onDoc)
  }, [open, options, value])

  const choose = (v: string) => {
    onChange(v)
    setOpen(false)
  }

  const onKeyDown = (e: KeyboardEvent<HTMLButtonElement>) => {
    if (disabled) return
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault()
        if (!open) setOpen(true)
        else setActive((a) => Math.min(options.length - 1, a + 1))
        break
      case 'ArrowUp':
        e.preventDefault()
        if (open) setActive((a) => Math.max(0, a - 1))
        break
      case 'Home':
        if (open) { e.preventDefault(); setActive(0) }
        break
      case 'End':
        if (open) { e.preventDefault(); setActive(options.length - 1) }
        break
      case 'Enter':
      case ' ':
        e.preventDefault()
        if (open && options[active]) choose(options[active].value)
        else setOpen(true)
        break
      case 'Escape':
        setOpen(false)
        break
    }
  }

  return (
    <div ref={ref} className={cn('relative', className)}>
      <button
        type="button"
        disabled={disabled}
        onClick={() => !disabled && setOpen((o) => !o)}
        onKeyDown={onKeyDown}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-label={ariaLabel}
        className={cn(
          'w-full flex items-center justify-between gap-2 px-3 py-2.5 rounded-lg text-sm text-left transition-all duration-150',
          'bg-zinc-800/70 border text-zinc-100',
          open ? 'border-cyan-400/60 ring-2 ring-cyan-400/20' : 'border-zinc-700 hover:border-zinc-600',
          disabled && 'opacity-50 cursor-not-allowed',
        )}
      >
        <span className={cn('truncate', !selected && 'text-zinc-500')}>{selected?.label ?? placeholder ?? 'Select…'}</span>
        <ChevronDown className={cn('w-4 h-4 flex-shrink-0 transition-transform duration-200', open ? 'rotate-180 text-cyan-400' : 'text-zinc-500')} />
      </button>

      <AnimatePresence>
        {open && (
          <motion.ul
            role="listbox"
            initial={{ opacity: 0, y: -4, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -4, scale: 0.98 }}
            transition={{ duration: 0.13, ease: 'easeOut' }}
            className="absolute z-50 mt-1.5 w-full max-h-64 overflow-y-auto rounded-lg border border-zinc-700 bg-zinc-900 p-1 shadow-2xl shadow-black/60"
          >
            {options.map((o, i) => {
              const isSel = o.value === value
              return (
                <li key={o.value} role="option" aria-selected={isSel}>
                  <button
                    type="button"
                    onMouseEnter={() => setActive(i)}
                    onClick={() => choose(o.value)}
                    className={cn(
                      'w-full flex items-center justify-between gap-2 px-2.5 py-2 rounded-md text-sm text-left transition-colors',
                      isSel
                        ? 'text-cyan-200 bg-gradient-brand-soft'
                        : i === active
                          ? 'bg-zinc-800 text-zinc-100'
                          : 'text-zinc-300',
                    )}
                  >
                    <span className="min-w-0">
                      <span className="block truncate">{o.label}</span>
                      {o.hint && <span className="block text-xs text-zinc-500 truncate">{o.hint}</span>}
                    </span>
                    {isSel && <Check className="w-4 h-4 text-cyan-400 flex-shrink-0" />}
                  </button>
                </li>
              )
            })}
          </motion.ul>
        )}
      </AnimatePresence>
    </div>
  )
}
