import { Clock } from 'lucide-react'
import { ProgressRing } from '@/components/ui/Progress'
import { scoreTone } from '@/lib/score'
import { cn } from '@/lib/utils'
import { formatTime } from '../format'

interface ScorePanelProps {
  pct: number
  correct: number
  total: number
  timeElapsed: number
}

/** Debrief hero: semantic score ring with the page's one highlighted serif figure. */
export function ScorePanel({ pct, correct, total, timeElapsed }: ScorePanelProps) {
  const tone = scoreTone(pct)
  return (
    <div className="flex flex-col items-center gap-6 sm:flex-row">
      <ProgressRing value={pct} size={132} strokeWidth={10}>
        {/* One highlighter mark per page — behind the headline score figure. */}
        <span className={cn('hl font-display tnum text-3xl font-semibold leading-none', tone.text)}>
          {pct}%
        </span>
        <span className="mt-1 text-xs text-ink-faint">{tone.label}</span>
      </ProgressRing>

      <div className="grid w-full flex-1 grid-cols-2 gap-3">
        <div className="rounded-lg border border-line bg-paper-deep p-4 text-center">
          <div className="mb-0.5 text-2xl font-bold text-success tnum">
            {correct}/{total}
          </div>
          <div className="text-xs text-ink-faint">Correct</div>
        </div>
        <div className="rounded-lg border border-line bg-paper-deep p-4 text-center">
          <div className="mb-0.5 flex items-center justify-center gap-1.5 text-2xl font-bold text-ink tnum">
            <Clock className="h-4 w-4 text-ink-faint" />
            {formatTime(timeElapsed)}
          </div>
          <div className="text-xs text-ink-faint">Time</div>
        </div>
      </div>
    </div>
  )
}
