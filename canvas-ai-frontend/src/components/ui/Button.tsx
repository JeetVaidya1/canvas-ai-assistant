import { forwardRef, useState } from 'react'
import type { ButtonHTMLAttributes, ReactNode, PointerEvent } from 'react'
import { Loader2 } from 'lucide-react'
import { cn } from '@/lib/utils'

type Variant = 'primary' | 'secondary' | 'ghost' | 'danger'
type Size = 'sm' | 'md' | 'lg'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant
  size?: Size
  loading?: boolean
  leftIcon?: ReactNode
  rightIcon?: ReactNode
}

interface Ripple {
  id: number
  x: number
  y: number
  size: number
}

const base =
  'relative overflow-hidden inline-flex items-center justify-center gap-2 font-medium rounded-lg select-none whitespace-nowrap ' +
  'transition-[transform,background-color,border-color,box-shadow,filter,color] duration-150 ease-out ' +
  'outline-none focus-visible:ring-2 focus-visible:ring-cyan-400/60 focus-visible:ring-offset-2 focus-visible:ring-offset-zinc-950 ' +
  'active:scale-[0.96] disabled:opacity-50 disabled:cursor-not-allowed disabled:saturate-50 disabled:active:scale-100'

const variants: Record<Variant, string> = {
  primary: 'text-white bg-gradient-brand glow-brand-sm hover:brightness-110 hover:glow-brand',
  secondary: 'text-zinc-200 bg-zinc-800 border border-zinc-700 hover:bg-zinc-700/80 hover:border-zinc-600 hover:text-white',
  ghost: 'text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800/70',
  danger: 'text-white bg-red-600 hover:bg-red-500 shadow-[0_4px_14px_-6px_rgba(239,68,68,0.5)]',
}

const rippleTint: Record<Variant, string> = {
  primary: 'bg-white/30',
  secondary: 'bg-white/10',
  ghost: 'bg-white/10',
  danger: 'bg-white/30',
}

const sizes: Record<Size, string> = {
  sm: 'text-xs px-3 py-1.5',
  md: 'text-sm px-4 py-2',
  lg: 'text-base px-5 py-2.5',
}

let rippleSeq = 0

/**
 * Interactive button — built from scratch with click-point ripple feedback and
 * press physics. Drop-in: same props as before, so every button app-wide
 * inherits the new interaction.
 */
export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = 'primary', size = 'md', loading = false, leftIcon, rightIcon, className, children, disabled, onPointerDown, ...rest },
  ref,
) {
  const [ripples, setRipples] = useState<Ripple[]>([])

  const handlePointerDown = (e: PointerEvent<HTMLButtonElement>) => {
    if (!(disabled || loading)) {
      const rect = e.currentTarget.getBoundingClientRect()
      const size = Math.max(rect.width, rect.height) * 1.1
      const id = ++rippleSeq
      setRipples((r) => [...r, { id, size, x: e.clientX - rect.left - size / 2, y: e.clientY - rect.top - size / 2 }])
      window.setTimeout(() => setRipples((r) => r.filter((p) => p.id !== id)), 600)
    }
    onPointerDown?.(e)
  }

  return (
    <button
      ref={ref}
      disabled={disabled || loading}
      onPointerDown={handlePointerDown}
      className={cn(base, variants[variant], sizes[size], className)}
      {...rest}
    >
      {ripples.map((r) => (
        <span
          key={r.id}
          aria-hidden
          className={cn('pointer-events-none absolute rounded-full', rippleTint[variant])}
          style={{ left: r.x, top: r.y, width: r.size, height: r.size }}
          ref={(el) => {
            if (el && !el.dataset.run) {
              el.dataset.run = '1'
              el.animate(
                [{ transform: 'scale(0)', opacity: 0.55 }, { transform: 'scale(1)', opacity: 0 }],
                { duration: 560, easing: 'cubic-bezier(0.22,1,0.36,1)' },
              )
            }
          }}
        />
      ))}
      <span className="relative inline-flex items-center gap-2">
        {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : leftIcon}
        {children}
        {!loading && rightIcon}
      </span>
    </button>
  )
})
