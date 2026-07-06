import type { HTMLAttributes, ReactNode } from 'react'
import { cn } from '@/lib/utils'

type Tone = 'neutral' | 'accent' | 'success' | 'warning' | 'danger' | 'info'

const tones: Record<Tone, string> = {
  neutral: 'text-zinc-300 bg-white/[0.05] border-white/10',
  accent: 'text-cyan-300 bg-cyan-500/10 border-cyan-400/20',
  success: 'text-emerald-300 bg-emerald-500/10 border-emerald-400/20',
  warning: 'text-amber-300 bg-amber-500/10 border-amber-400/25',
  danger: 'text-rose-300 bg-rose-500/10 border-rose-400/25',
  info: 'text-sky-300 bg-sky-500/10 border-sky-400/20',
}

interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  tone?: Tone
  icon?: ReactNode
}

/** Small status chip. One tone system app-wide — never hand-roll pill styles. */
export function Badge({ tone = 'neutral', icon, className, children, ...rest }: BadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 text-[11px] font-medium rounded-full border px-2.5 py-0.5 whitespace-nowrap',
        '[&>svg]:w-3 [&>svg]:h-3',
        tones[tone],
        className,
      )}
      {...rest}
    >
      {icon}
      {children}
    </span>
  )
}
