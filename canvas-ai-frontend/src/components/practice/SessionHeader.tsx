import type { ReactNode } from 'react'
import { ProgressBar } from '@/components/ui/Progress'
import { PROGRESS_ACCENT } from './constants'

interface SessionHeaderProps {
  /** "Question" or "Problem". */
  itemLabel: string
  /** 0-based index of the current item. */
  index: number
  total: number
  /** Muted context after the counter, e.g. "Topic · difficulty". */
  meta?: string
  /** Right-aligned live counters (correct count, timer). */
  right?: ReactNode
  /** 0–100 completion through the session. */
  progress: number
}

/** Slim in-session progress strip: "Question N of M" + meta + counters + ProgressBar. */
export function SessionHeader({ itemLabel, index, total, meta, right, progress }: SessionHeaderProps) {
  return (
    <div className="mb-6">
      <div className="mb-2.5 flex items-center justify-between">
        <div className="flex items-baseline gap-2">
          <span className="text-sm font-semibold text-ink">{itemLabel}</span>
          <span className="section-num tnum">
            {index + 1} of {total}
          </span>
          {meta && <span className="hidden text-xs text-ink-faint sm:inline">· {meta}</span>}
        </div>
        {right && <div className="flex items-center gap-3.5">{right}</div>}
      </div>
      <ProgressBar value={progress} color={PROGRESS_ACCENT} label={`${itemLabel} progress`} />
      <div className="sr-only">{Math.round(progress)}% complete</div>
    </div>
  )
}
