import { forwardRef } from 'react'
import type { ButtonHTMLAttributes, ReactNode } from 'react'
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

const base =
  'inline-flex items-center justify-center gap-2 font-medium rounded-lg transition-all ' +
  'focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400/60 focus-visible:ring-offset-2 ' +
  'focus-visible:ring-offset-zinc-950 disabled:opacity-50 disabled:cursor-not-allowed disabled:saturate-50 ' +
  'active:scale-[0.98] select-none whitespace-nowrap'

const variants: Record<Variant, string> = {
  // Brand gradient + soft glow — the landing-page DNA, used for the main action on a screen.
  primary:
    'text-white bg-gradient-brand hover:brightness-110 glow-brand-sm hover:glow-brand',
  secondary:
    'text-zinc-200 bg-zinc-800 border border-zinc-700 hover:bg-zinc-700/80 hover:border-zinc-600',
  ghost:
    'text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800/70',
  danger:
    'text-white bg-red-600 hover:bg-red-500',
}

const sizes: Record<Size, string> = {
  sm: 'text-xs px-3 py-1.5',
  md: 'text-sm px-4 py-2',
  lg: 'text-base px-5 py-2.5',
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = 'primary', size = 'md', loading = false, leftIcon, rightIcon, className, children, disabled, ...rest },
  ref,
) {
  return (
    <button
      ref={ref}
      disabled={disabled || loading}
      className={cn(base, variants[variant], sizes[size], className)}
      {...rest}
    >
      {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : leftIcon}
      {children}
      {!loading && rightIcon}
    </button>
  )
})
