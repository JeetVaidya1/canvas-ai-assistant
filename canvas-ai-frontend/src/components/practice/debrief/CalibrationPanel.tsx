import { AlertTriangle } from 'lucide-react'
import { ProgressBar } from '@/components/ui/Progress'
import type { QuizCalibration, QuizConfidence } from '@/lib/api'

const BUCKETS: readonly { key: QuizConfidence; label: string }[] = [
  { key: 'sure', label: 'Sure' },
  { key: 'thinkso', label: 'Think so' },
  { key: 'guessing', label: 'Guessing' },
]

interface CalibrationPanelProps {
  calibration: QuizCalibration | undefined
}

/**
 * How well confidence tracked correctness: one bar per tagged bucket, plus a
 * warning callout for confident-wrong answers (they seed the review queue).
 */
export function CalibrationPanel({ calibration }: CalibrationPanelProps) {
  const rows = calibration
    ? BUCKETS.map(({ key, label }) => ({ key, label, ...calibration[key] })).filter((r) => r.n > 0)
    : []

  if (rows.length === 0) {
    return (
      <p className="text-sm text-ink-faint">
        No confidence tags this run. Tap Sure / Think so / Guessing before submitting to see how
        well your gut tracks your score.
      </p>
    )
  }

  const confidentWrong = calibration?.confident_wrong ?? 0

  return (
    <div>
      <div className="space-y-3">
        {rows.map((row) => {
          const pct = Math.round((row.correct / row.n) * 100)
          return (
            <div key={row.key}>
              <div className="mb-1 flex items-center justify-between text-sm">
                <span className="text-ink-soft">{row.label}</span>
                <span className="flex-shrink-0 text-ink-faint tnum">
                  {row.correct}/{row.n} correct &middot; {pct}%
                </span>
              </div>
              <ProgressBar value={pct} className="h-2" label={`${row.label} accuracy`} />
            </div>
          )
        })}
      </div>

      {confidentWrong > 0 && (
        <div className="mt-4 rounded-lg border border-warning/25 bg-warning-wash p-3.5">
          <div className="flex items-start gap-2.5">
            <AlertTriangle className="mt-0.5 h-4 w-4 flex-shrink-0 text-warning" />
            <p className="text-sm text-ink-soft">
              <span className="font-semibold text-warning">
                You were sure about {confidentWrong} {confidentWrong === 1 ? 'answer' : 'answers'} you
                missed
              </span>{' '}
              — these seed your review queue.
            </p>
          </div>
        </div>
      )}
    </div>
  )
}
