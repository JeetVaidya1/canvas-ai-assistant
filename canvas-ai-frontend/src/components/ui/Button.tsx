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
  'inline-flex items-center justify-center gap-2 font-medium rounded-lg select-none whitespace-nowrap ' +
  'transition-[background-color,border-color,box-shadow,color,transform] duration-150 ease-out ' +
  'outline-none focus-visible:ring-2 focus-visible:ring-accent/60 focus-visible:ring-offset-2 focus-visible:ring-offset-paper ' +
  'active:scale-[0.985] disabled:opacity-45 disabled:cursor-not-allowed disabled:active:scale-100'

const variants: Record<Variant, string> = {
  primary: 'text-white bg-accent hover:bg-accent-deep shadow-[0_1px_2px_rgba(33,31,26,0.15)]',
  secondary: 'text-ink bg-surface border border-line hover:border-line-strong hover:bg-surface-hover shadow-[0_1px_2px_rgba(33,31,26,0.05)]',
  ghost: 'text-ink-soft hover:text-ink hover:bg-paper-deep',
  danger: 'text-white bg-danger hover:bg-[#a53a3a] shadow-[0_1px_2px_rgba(33,31,26,0.15)]',
}

const sizes: Record<Size, string> = {
  sm: 'text-xs px-3 py-1.5',
  md: 'text-sm px-4 py-2',
  lg: 'text-[15px] px-5 py-2.5',
}

/**
 * The app's one button. Paper & Ink: solid pen-blue primary, white paper
 * secondary, quiet ghost. No gradients, no glow, no ripple — press physics
 * and clear states carry the interaction.
 */
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
