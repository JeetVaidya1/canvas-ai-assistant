import { useState } from 'react'
import { Link } from 'react-router-dom'
import { Brain, ChevronRight } from 'lucide-react'
import type { Readiness } from '@/lib/api'
import type { CourseTopic } from '@/lib/api/topics'
import { Card } from '@/components/ui/Card'
import { ProgressBar } from '@/components/ui/Progress'
import { EmptyState } from '@/components/ui/States'
import { SectionHead } from './SectionHead'
import { matchAnalyticsToTopics } from './topicMatch'

const MAX_LEGACY_TOPICS = 12

interface MasteryBarRowProps {
  label: string
  masteryPct: number
}

function MasteryBarRow({ label, masteryPct }: MasteryBarRowProps) {
  const pct = Math.round(masteryPct)
  return (
    <div>
      <div className="flex items-center justify-between text-sm mb-1.5">
        <span className="text-ink truncate pr-3">{label}</span>
        <span className="text-ink tnum font-semibold">{pct}%</span>
      </div>
      <ProgressBar value={masteryPct} className="h-2.5" label={`${label} mastery`} />
    </div>
  )
}

interface TopicMasteryListProps {
  readiness: Readiness | null
  /** Course Brain taxonomy — when present, mastery is re-keyed onto these clean names. */
  topics?: CourseTopic[]
  courseId: string
}

/**
 * Per-topic mastery keyed on the Course Brain: analytics by_topic rows are
 * matched to course topics (normalized contains-match) and shown under the
 * clean topic names. Unmatched analytics rows stay visible in a collapsed
 * disclosure; topics with no data yet get a quiet "Drill" nudge.
 */
export function TopicMasteryList({ readiness, topics, courseId }: TopicMasteryListProps) {
  const [showOther, setShowOther] = useState(false)
  const byTopic = readiness?.by_topic ?? []
  const hasCourseTopics = !!topics && topics.length > 0
  const { matched, unmatched } = matchAnalyticsToTopics(topics ?? [], byTopic)

  // Legacy fallback: no Course Brain yet — render the raw analytics rows.
  const legacyRows = byTopic.filter((t) => t.has_data).slice(0, MAX_LEGACY_TOPICS)
  const isEmpty = hasCourseTopics ? false : legacyRows.length === 0

  return (
    <Card padding="lg" className="h-full">
      <SectionHead
        num="03"
        title="Topic mastery"
        hint="Your course's topics — where your readiness score comes from"
      />
      {isEmpty ? (
        <EmptyState
          icon={<Brain />}
          title="No topic mastery yet"
          description="Ask questions or take a quiz to start scoring topics."
        />
      ) : hasCourseTopics ? (
        <div className="space-y-3.5">
          {matched.map((row) =>
            row.masteryPct !== null ? (
              <MasteryBarRow key={row.slug} label={row.name} masteryPct={row.masteryPct} />
            ) : (
              <div key={row.slug} className="flex items-center justify-between text-sm">
                <span className="text-ink-faint truncate pr-3">{row.name}</span>
                <span className="flex items-center gap-2.5 flex-shrink-0">
                  <span className="text-xs text-ink-faint">not practiced yet</span>
                  <Link
                    to={`/course/${courseId}/practice`}
                    className="text-xs font-medium text-accent hover:text-accent-deep focus-ring rounded"
                  >
                    Drill
                  </Link>
                </span>
              </div>
            ),
          )}

          {/* Analytics rows that matched no course topic — never hide data. */}
          {unmatched.length > 0 && (
            <div className="pt-3 border-t border-line">
              <button
                type="button"
                onClick={() => setShowOther((v) => !v)}
                aria-expanded={showOther}
                className="flex items-center gap-1.5 text-xs text-ink-soft hover:text-ink transition-colors focus-ring rounded"
              >
                <ChevronRight
                  className={`w-3.5 h-3.5 transition-transform ${showOther ? 'rotate-90' : ''}`}
                />
                Other tracked items ({unmatched.length})
              </button>
              {showOther && (
                <div className="space-y-3.5 mt-3.5">
                  {unmatched.map((t) => (
                    <MasteryBarRow key={t.topic} label={t.topic} masteryPct={t.mastery_pct} />
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      ) : (
        <div className="space-y-3.5">
          {legacyRows.map((t) => (
            <MasteryBarRow key={t.topic} label={t.topic} masteryPct={t.mastery_pct} />
          ))}
        </div>
      )}
    </Card>
  )
}
