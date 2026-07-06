import { ProgressBar } from '@/components/ui/Progress'
import type { QuizTopicScore } from '@/lib/api'

interface TopicBreakdownProps {
  byTopic: QuizTopicScore[]
}

/** Per-topic accuracy bars, weakest first — where the next drill should aim. */
export function TopicBreakdown({ byTopic }: TopicBreakdownProps) {
  const sorted = [...byTopic].sort((a, b) => a.pct - b.pct)
  if (sorted.length === 0) {
    return <p className="text-sm text-ink-faint">No per-topic data for this run.</p>
  }
  return (
    <div className="space-y-2.5">
      {sorted.map((t) => (
        <div key={t.topic}>
          <div className="mb-1 flex items-center justify-between text-sm">
            <span className="truncate pr-3 text-ink-soft">{t.topic}</span>
            <span className="flex-shrink-0 text-ink-faint tnum">
              {t.correct}/{t.total} &middot; {t.pct}%
            </span>
          </div>
          <ProgressBar value={t.pct} className="h-2" label={`${t.topic} score`} />
        </div>
      ))}
    </div>
  )
}
