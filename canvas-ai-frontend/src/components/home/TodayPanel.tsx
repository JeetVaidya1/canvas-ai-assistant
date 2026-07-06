import { Check, ChevronRight } from 'lucide-react'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import ErrorInline from '@/components/shared/ErrorInline'
import type { TodayItem } from './todayItems'

interface TodayPanelProps {
  loading: boolean
  /** True only when every remote source failed — partial data still renders. */
  error: boolean
  onRetry: () => void
  items: TodayItem[]
  onboarding: boolean
  totalMin: number
  onGo: (to: string) => void
  onPrefetch: (to: string) => void
}

/**
 * The opinionated "what to do today" checklist that leads CourseHome.
 * Assembled client-side from due reviews, the weakest topic and recent
 * activity; falls back to an honest onboarding checklist when no data exists.
 */
export default function TodayPanel({
  loading,
  error,
  onRetry,
  items,
  onboarding,
  totalMin,
  onGo,
  onPrefetch,
}: TodayPanelProps) {
  // The single primary "Start" action belongs to the top not-yet-done item.
  const primaryKey = items.find((i) => !i.done)?.key

  return (
    <Card padding="none" elevation={2}>
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1 px-5 pt-5 pb-4 border-b border-line">
        <h2 className="font-display text-xl font-semibold text-ink">Today</h2>
        <p className="text-sm text-ink-soft min-w-0">
          {loading ? (
            'Putting your plan together…'
          ) : error ? (
            'Your plan is unavailable'
          ) : onboarding ? (
            'Three steps to get this course working for you'
          ) : (
            <>
              About <span className="hl tnum">~{totalMin} min</span> of focused work
            </>
          )}
        </p>
      </div>

      {loading ? (
        <TodaySkeleton />
      ) : error ? (
        <div className="p-5">
          <ErrorInline message="Couldn't assemble today's plan." onRetry={onRetry} />
        </div>
      ) : (
        <ul className="divide-y divide-line">
          {items.map((item) => (
            <TodayRow
              key={item.key}
              item={item}
              primary={item.key === primaryKey}
              onGo={onGo}
              onPrefetch={onPrefetch}
            />
          ))}
        </ul>
      )}
    </Card>
  )
}

function TodayRow({
  item,
  primary,
  onGo,
  onPrefetch,
}: {
  item: TodayItem
  primary: boolean
  onGo: (to: string) => void
  onPrefetch: (to: string) => void
}) {
  const done = item.done === true
  return (
    <li>
      <div
        className="group flex items-center gap-3.5 px-5 py-3.5 hover:bg-paper-deep/40 transition-colors cursor-pointer"
        onClick={() => onGo(item.to)}
        onMouseEnter={() => onPrefetch(item.to)}
      >
        {/* Checkbox-style bullet — visual only */}
        <span
          aria-hidden
          className={`w-[18px] h-[18px] rounded-[5px] border flex items-center justify-center flex-shrink-0 ${
            done ? 'bg-success border-success text-white' : 'bg-surface border-line-strong'
          }`}
        >
          {done && <Check className="w-3 h-3" strokeWidth={3} />}
        </span>

        <div className="flex-1 min-w-0">
          <p className={`text-sm font-medium ${done ? 'text-ink-faint line-through' : 'text-ink'}`}>
            {item.label}
          </p>
          {item.detail && <p className="text-xs text-ink-faint truncate mt-0.5">{item.detail}</p>}
        </div>

        <span className="text-xs text-ink-faint tnum flex-shrink-0">~{item.etaMin} min</span>

        {primary ? (
          <Button
            size="sm"
            className="flex-shrink-0"
            onClick={(e) => {
              e.stopPropagation()
              onGo(item.to)
            }}
          >
            Start
          </Button>
        ) : (
          <ChevronRight className="w-4 h-4 text-ink-faint group-hover:text-accent group-hover:translate-x-0.5 transition-all flex-shrink-0" />
        )}
      </div>
    </li>
  )
}

function TodaySkeleton() {
  return (
    <ul className="divide-y divide-line" aria-hidden>
      {[0, 1, 2].map((i) => (
        <li key={i} className="flex items-center gap-3.5 px-5 py-4">
          <div className="w-[18px] h-[18px] rounded-[5px] bg-paper-deep animate-pulse flex-shrink-0" />
          <div className="flex-1 space-y-1.5 min-w-0">
            <div className="h-3.5 w-48 max-w-full rounded bg-paper-deep animate-pulse" />
            <div className="h-3 w-32 max-w-full rounded bg-paper-deep animate-pulse" />
          </div>
          <div className="h-3 w-12 rounded bg-paper-deep animate-pulse flex-shrink-0" />
        </li>
      ))}
    </ul>
  )
}
