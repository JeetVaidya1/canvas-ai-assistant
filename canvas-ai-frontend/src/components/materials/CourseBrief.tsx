import { Map as MapIcon, RefreshCw } from 'lucide-react'
import type { CourseTopic } from '@/lib/api/topics'
import { useCourseTopics, useRebuildCourseTopics } from '@/hooks/useCourseTopics'
import { showError } from '@/lib/toast'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { EmptyState } from '@/components/ui/States'
import ErrorInline from '@/components/shared/ErrorInline'

/** Coverage lines list at most this many source docs before eliding. */
const MAX_COVERAGE_DOCS = 2

function formatPages(pages: [number, number]): string {
  const [start, end] = pages
  return start === end ? `p. ${start}` : `pp. ${start}–${end}`
}

/** "Graphs.pdf · pp. 3–18 · Trees.pdf · pp. 1–4 +2 more" — or null when unknown. */
function formatCoverage(coverage: CourseTopic['doc_coverage']): string | null {
  if (!coverage || coverage.length === 0) return null
  const shown = coverage.slice(0, MAX_COVERAGE_DOCS).map((c) => `${c.doc} · ${formatPages(c.pages)}`)
  const extra = coverage.length - MAX_COVERAGE_DOCS
  return extra > 0 ? `${shown.join('  ·  ')}  +${extra} more` : shown.join('  ·  ')
}

function BriefSkeleton() {
  return (
    <div className="space-y-5" aria-label="Loading course brief">
      {[0, 1, 2, 3].map((i) => (
        <div key={i} className="flex gap-3">
          <div className="h-4 w-6 rounded bg-paper-deep animate-pulse flex-shrink-0" />
          <div className="flex-1 space-y-2">
            <div className="h-4 w-2/5 rounded bg-paper-deep animate-pulse" />
            <div className="h-3 w-4/5 rounded bg-paper-deep animate-pulse" />
          </div>
        </div>
      ))}
    </div>
  )
}

interface CourseBriefProps {
  courseId: string
  /** Whether the course has any indexed files — decides which empty state to show. */
  hasFiles: boolean
}

/**
 * The Course Brief — "What Vindexa understood" after ingest. Renders the
 * Course Brain taxonomy as a numbered syllabus: clean topic names,
 * descriptions, and the doc/page coverage each topic was derived from.
 * This is the trust-building moment right after upload.
 */
export default function CourseBrief({ courseId, hasFiles }: CourseBriefProps) {
  const topicsQuery = useCourseTopics(courseId)
  const rebuild = useRebuildCourseTopics(courseId)

  const topics = topicsQuery.data?.topics ?? []
  const hasTopics = topics.length > 0

  const handleRebuild = () => {
    rebuild.mutate(undefined, {
      onError: () => showError('Couldn’t rebuild the course map. Try again in a moment.'),
    })
  }

  return (
    <Card padding="md" className="space-y-4">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h2 className="text-sm font-semibold text-ink">What Vindexa understood</h2>
          <p className="text-xs text-ink-soft mt-0.5">
            The course map behind every quiz, review and readiness score.
          </p>
        </div>
        {hasTopics && (
          <Button
            variant="ghost"
            size="sm"
            onClick={handleRebuild}
            loading={rebuild.isPending}
            leftIcon={<RefreshCw className="w-3.5 h-3.5" />}
            className="flex-shrink-0"
          >
            Rebuild
          </Button>
        )}
      </div>

      {topicsQuery.isPending ? (
        <BriefSkeleton />
      ) : topicsQuery.isError ? (
        <ErrorInline
          message="Couldn’t load the course brief."
          onRetry={() => void topicsQuery.refetch()}
        />
      ) : !hasTopics && !hasFiles ? (
        <EmptyState
          icon={<MapIcon />}
          title="Nothing to map yet"
          description="Add materials and Vindexa will map your course — topics, what they cover, and how they build on each other."
          className="py-8"
        />
      ) : !hasTopics ? (
        <EmptyState
          icon={<MapIcon />}
          title="Your course map is ready to build"
          description="Vindexa reads your files and distills them into the topics that drive quizzes, mastery and reviews."
          className="py-8"
          action={
            <Button onClick={handleRebuild} loading={rebuild.isPending} leftIcon={<RefreshCw className="w-4 h-4" />}>
              Generate topics
            </Button>
          }
        />
      ) : (
        <ol className="space-y-4">
          {topics.map((topic, i) => {
            const coverage = formatCoverage(topic.doc_coverage)
            return (
              <li key={topic.slug} className="flex gap-3">
                <span className="section-num pt-1 flex-shrink-0 tnum">
                  {String(i + 1).padStart(2, '0')}
                </span>
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-semibold text-ink">{topic.name}</p>
                  {topic.description && (
                    <p className="text-sm text-ink-soft mt-0.5 leading-relaxed">{topic.description}</p>
                  )}
                  {coverage && (
                    <p className="text-xs text-ink-faint mt-1 truncate" title={coverage}>
                      {coverage}
                    </p>
                  )}
                </div>
              </li>
            )
          })}
        </ol>
      )}
    </Card>
  )
}
