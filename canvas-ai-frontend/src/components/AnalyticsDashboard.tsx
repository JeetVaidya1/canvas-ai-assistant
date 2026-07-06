// src/components/AnalyticsDashboard.tsx
import { useNavigate } from 'react-router-dom'
import { RefreshCw, Sparkles } from 'lucide-react'
import type { LearningAnalytics } from '@/lib/api'
import { useLearningAnalytics, useConceptGraph } from '@/hooks/useAnalytics'
import { useReadiness } from '@/hooks/useReadiness'
import { Button } from '@/components/ui/Button'
import { Card } from '@/components/ui/Card'
import { EmptyState, ErrorState } from '@/components/ui/States'
import { AnalyticsSkeleton } from '@/components/progress/AnalyticsSkeleton'
import { ReadinessHero } from '@/components/progress/ReadinessHero'
import { StatStrip } from '@/components/progress/StatStrip'
import { ReviewQueue } from '@/components/progress/ReviewQueue'
import { ConceptGraphMap } from '@/components/progress/ConceptGraphMap'
import { TopicMasteryList } from '@/components/progress/TopicMasteryList'
import { FocusPanel } from '@/components/progress/FocusPanel'
import { TrendCharts } from '@/components/progress/TrendCharts'
import { StudyScheduleCard } from '@/components/progress/StudyScheduleCard'

interface AnalyticsDashboardProps {
  courseId: string
  userId: string
}

/** True once the user has produced any signal worth charting. */
function hasSignal(analytics: LearningAnalytics): boolean {
  return (
    analytics.total_questions > 0 ||
    analytics.topics_progress.length > 0 ||
    analytics.study_time_trend.length > 0 ||
    analytics.study_streak > 0
  )
}

/**
 * Analytics tab of the Progress destination — thin composition over the
 * progress/ section components. Analytics + readiness gate the page; the
 * concept graph loads separately (its first call builds the graph
 * server-side and is slow — non-blocking).
 */
export default function AnalyticsDashboard({ courseId, userId }: AnalyticsDashboardProps) {
  const navigate = useNavigate()
  const analyticsQuery = useLearningAnalytics(courseId, userId)
  const readinessQuery = useReadiness(courseId, userId)
  const graphQuery = useConceptGraph(courseId, userId)

  const analytics = analyticsQuery.data ?? null
  const readiness = readinessQuery.data ?? null
  const graph = graphQuery.data ?? null

  const refreshAll = () => {
    void analyticsQuery.refetch()
    void readinessQuery.refetch()
    void graphQuery.refetch()
  }

  if (analyticsQuery.isPending || readinessQuery.isPending) {
    return <AnalyticsSkeleton />
  }

  if (analyticsQuery.isError) {
    return (
      <div className="max-w-6xl mx-auto p-6">
        <ErrorState
          title="Couldn't load your analytics"
          description="Check your connection and try again — your study data is safe."
          onRetry={refreshAll}
          retrying={analyticsQuery.isRefetching}
        />
      </div>
    )
  }

  // Zero-data: guide to Practice instead of rendering zeroed-out charts.
  if (!analytics || !hasSignal(analytics)) {
    return (
      <div className="max-w-6xl mx-auto p-6">
        <Card accent padding="lg" elevation={2}>
          <EmptyState
            icon={<Sparkles />}
            title="Your analytics build as you study"
            description="Take a quick quiz to start — your readiness score, topic mastery and study trends will grow from there."
            action={
              <Button onClick={() => navigate(`/course/${courseId}/practice`)} leftIcon={<Sparkles className="w-4 h-4" />}>
                Take a quick quiz
              </Button>
            }
          />
        </Card>
      </div>
    )
  }

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      {/* Slim toolbar — the Progress wrapper bar already names the page. */}
      <div className="flex items-center justify-end -mb-2">
        <Button
          variant="ghost"
          size="sm"
          onClick={refreshAll}
          loading={analyticsQuery.isRefetching}
          leftIcon={<RefreshCw className="w-3.5 h-3.5" />}
        >
          Refresh
        </Button>
      </div>

      {/* Hero: exam readiness */}
      {readinessQuery.isError ? (
        <ErrorState
          compact
          title="Couldn't load your exam readiness."
          onRetry={() => void readinessQuery.refetch()}
          retrying={readinessQuery.isRefetching}
        />
      ) : (
        readiness && <ReadinessHero readiness={readiness} />
      )}

      <StatStrip analytics={analytics} />

      {/* Mistake-driven review queue (hidden when nothing is due) */}
      <ReviewQueue courseId={courseId} userId={userId} />

      <ConceptGraphMap
        graph={graph}
        isError={graphQuery.isError}
        onRetry={() => void graphQuery.refetch()}
      />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-stretch">
        <TopicMasteryList readiness={readiness} />
        <FocusPanel analytics={analytics} />
      </div>

      <TrendCharts analytics={analytics} />

      <StudyScheduleCard analytics={analytics} />
    </div>
  )
}
