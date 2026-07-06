import { motion } from 'motion/react'
import { cn } from '@/lib/utils'
import { scoreTone } from '@/lib/score'

interface ProgressBarProps {
  /** 0–100 */
  value: number
  className?: string
  /** Explicit bar color; defaults to the semantic scoreTone. */
  color?: string
  label?: string
}

export function ProgressBar({ value, className, color, label }: ProgressBarProps) {
  const clamped = Math.max(0, Math.min(100, value))
  const barColor = color ?? scoreTone(clamped).stroke
  return (
    <div
      className={cn('h-1.5 rounded-full bg-paper-deep border border-line/60 overflow-hidden', className)}
      role="progressbar"
      aria-valuenow={Math.round(clamped)}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-label={label}
    >
      <motion.div
        className="h-full rounded-full"
        style={{ backgroundColor: barColor }}
        initial={{ width: 0 }}
        animate={{ width: `${clamped}%` }}
        transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
      />
    </div>
  )
}

interface ProgressRingProps {
  /** 0–100 */
  value: number
  /** Outer diameter in px. */
  size?: number
  strokeWidth?: number
  /** Explicit ring color; defaults to the semantic scoreTone. */
  color?: string
  className?: string
  children?: React.ReactNode
}

/**
 * Circular progress ring (readiness, mastery). Center content via children.
 * Ink on paper: warm track, semantic stroke, no glow filters.
 */
export function ProgressRing({ value, size = 104, strokeWidth = 8, color, className, children }: ProgressRingProps) {
  const clamped = Math.max(0, Math.min(100, value))
  const ringColor = color ?? scoreTone(clamped).stroke
  const r = (size - strokeWidth) / 2
  const circ = 2 * Math.PI * r
  return (
    <div className={cn('relative flex-shrink-0', className)} style={{ width: size, height: size }}>
      <svg className="-rotate-90" width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="#e7e3d9" strokeWidth={strokeWidth} />
        <motion.circle
          cx={size / 2} cy={size / 2} r={r} fill="none"
          stroke={ringColor} strokeWidth={strokeWidth} strokeLinecap="round"
          strokeDasharray={circ}
          initial={{ strokeDashoffset: circ }}
          animate={{ strokeDashoffset: circ * (1 - clamped / 100) }}
          transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
        />
      </svg>
      {children && <div className="absolute inset-0 flex flex-col items-center justify-center">{children}</div>}
    </div>
  )
}
