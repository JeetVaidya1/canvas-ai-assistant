import { ListTree } from 'lucide-react'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { ProgressBar } from '@/components/ui/Progress'
import { EmptyState } from '@/components/ui/States'
import ErrorInline from '@/components/shared/ErrorInline'
import type { CourseTopic } from '@/lib/api/topics'

interface TopicsSectionProps {
  loading: boolean
  error: boolean
  onRetry: () => void
  topics: CourseTopic[]
  /** Mastery pct for a topic name, or null when unmeasured (no bar rendered). */
  masteryFor: (name: string) => number | null
  onDrill: () => void
  onAsk: () => void
  onPrefetchDrill: () => void
  onPrefetchAsk: () => void
  onRebuild: () => void
  rebuilding: boolean
  rebuildError: boolean
}

/**
 * The Course Brain taxonomy as dense rows on the white sheet: name,
 * one-line description, mastery bar where readiness has measured the topic,
 * and quiet Drill / Ask actions on hover.
 */
export default function TopicsSection({
  loading,
  error,
  onRetry,
  topics,
  masteryFor,
  onDrill,
  onAsk,
  onPrefetchDrill,
  onPrefetchAsk,
  onRebuild,
  rebuilding,
  rebuildError,
}: TopicsSectionProps) {
  return (
    <section>
      <div className="section-head mb-4">
        <span className="section-num">01</span>
        <h2 className="text-sm font-semibold text-ink">Topics</h2>
        {!loading && !error && topics.length > 0 && (
          <span className="ml-auto text-xs text-ink-faint tnum">
            {topics.length} topic{topics.length !== 1 ? 's' : ''}
          </span>
        )}
      </div>

      {loading ? (
        <TopicsSkeleton />
      ) : error ? (
        <ErrorInline message="Couldn't load this course's topics." onRetry={onRetry} />
      ) : topics.length === 0 ? (
        <Card padding="none">
          {rebuildError && (
            <div className="px-5 pt-5">
              <ErrorInline message="Topic generation failed — try again." onRetry={onRebuild} />
            </div>
          )}
          <EmptyState
            icon={<ListTree />}
            title="No topics yet"
            description="Vindexa reads your materials and distills them into a clean topic map — the backbone for drills, mastery and readiness."
            action={
              <Button onClick={onRebuild} loading={rebuilding}>
                Generate topics
              </Button>
            }
          />
        </Card>
      ) : (
        <Card padding="none">
          <ul className="divide-y divide-line">
            {topics.map((topic) => (
              <TopicRow
                key={topic.slug}
                topic={topic}
                mastery={masteryFor(topic.name)}
                onDrill={onDrill}
                onAsk={onAsk}
                onPrefetchDrill={onPrefetchDrill}
                onPrefetchAsk={onPrefetchAsk}
              />
            ))}
          </ul>
        </Card>
      )}
    </section>
  )
}

function TopicRow({
  topic,
  mastery,
  onDrill,
  onAsk,
  onPrefetchDrill,
  onPrefetchAsk,
}: {
  topic: CourseTopic
  mastery: number | null
  onDrill: () => void
  onAsk: () => void
  onPrefetchDrill: () => void
  onPrefetchAsk: () => void
}) {
  return (
    <li className="group flex items-center gap-4 px-5 py-3">
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-ink truncate">{topic.name}</p>
        {topic.description && (
          <p className="text-xs text-ink-faint truncate mt-0.5">{topic.description}</p>
        )}
      </div>

      {mastery !== null && (
        <div className="hidden sm:flex items-center gap-2.5 flex-shrink-0 w-36">
          <ProgressBar value={mastery} className="flex-1" label={`${topic.name} mastery`} />
          <span className="text-xs text-ink-soft tnum w-8 text-right">{mastery}%</span>
        </div>
      )}

      <div className="flex items-center gap-0.5 flex-shrink-0 sm:opacity-0 sm:group-hover:opacity-100 sm:group-focus-within:opacity-100 transition-opacity">
        <Button variant="ghost" size="sm" onClick={onDrill} onMouseEnter={onPrefetchDrill}>
          Drill
        </Button>
        <Button variant="ghost" size="sm" onClick={onAsk} onMouseEnter={onPrefetchAsk}>
          Ask
        </Button>
      </div>
    </li>
  )
}

function TopicsSkeleton() {
  return (
    <Card padding="none" aria-hidden>
      <ul className="divide-y divide-line">
        {[0, 1, 2, 3, 4].map((i) => (
          <li key={i} className="flex items-center gap-4 px-5 py-3.5">
            <div className="flex-1 space-y-1.5 min-w-0">
              <div className="h-3.5 w-44 max-w-full rounded bg-paper-deep animate-pulse" />
              <div className="h-3 w-64 max-w-full rounded bg-paper-deep animate-pulse" />
            </div>
            <div className="h-1.5 w-36 rounded-full bg-paper-deep animate-pulse hidden sm:block" />
          </li>
        ))}
      </ul>
    </Card>
  )
}
