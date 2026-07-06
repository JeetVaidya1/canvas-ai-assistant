import { useNavigate, useParams } from 'react-router-dom'
import { motion } from 'motion/react'
import { Clock, FileText, Sparkles } from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { useCourses } from '@/hooks/useCourses'
import { useCourseFiles } from '@/hooks/useCourseFiles'
import { useCourseTopics, useRebuildCourseTopics } from '@/hooks/useCourseTopics'
import { useLearningAnalytics } from '@/hooks/useAnalytics'
import { useReadiness } from '@/hooks/useReadiness'
import { useRecentActivity } from '@/hooks/useRecentActivity'
import { useReviewQueue } from '@/hooks/useReviews'
import { useUser } from '@/hooks/useUser'
import { usePrefetch } from '@/hooks/usePrefetch'
import TodayPanel from '@/components/home/TodayPanel'
import TopicsSection from '@/components/home/TopicsSection'
import CourseRail from '@/components/home/CourseRail'
import { buildTodayPlan, masteryForTopic, pageLabel } from '@/components/home/todayItems'

const ENTRANCE = { duration: 0.25, ease: [0.22, 1, 0.36, 1] as const }

/**
 * CourseHome — the course command center. Leads with an opinionated Today
 * checklist and the Course Brain topic map (main column); readiness, streak,
 * materials and the exam CTA live in a sticky right rail. Navigation between
 * modes belongs to the sidebar — this page is about WHAT to do next.
 */
export default function CourseHome() {
  const { courseId } = useParams<{ courseId: string }>()
  const navigate = useNavigate()
  const userId = useUser()

  const { data: courses } = useCourses()
  const filesQ = useCourseFiles(courseId)
  const readinessQ = useReadiness(courseId, userId)
  const reviewsQ = useReviewQueue(courseId, userId)
  const topicsQ = useCourseTopics(courseId)
  const analyticsQ = useLearningAnalytics(courseId, userId)
  const rebuildTopics = useRebuildCourseTopics(courseId)
  const recent = useRecentActivity().filter((e) => e.courseId === courseId)
  const { prefetchLearn, prefetchPractice, prefetchStudyKit, prefetchProgress } = usePrefetch()

  const course = courses?.find((c) => c.course_id === courseId)
  const fileCount = filesQ.data?.length ?? 0
  const hasFiles = fileCount > 0
  const readiness = readinessQ.data ?? null
  const score = readiness ? Math.round(readiness.score_pct) : null
  const lastStudied = recent[0]?.page ?? null

  const go = (to: string) => navigate(`/course/${courseId}${to ? `/${to}` : ''}`)

  // Warm a destination's primary data while the cursor hovers its row.
  const prefetchFor = (to: string) => {
    if (!courseId) return
    if (to === 'learn' || to === 'chat') prefetchLearn()
    else if (to === 'practice' || to === 'quiz') prefetchPractice(courseId)
    else if (to === 'kit' || to === 'notes') prefetchStudyKit(courseId)
    else if (to === 'progress') prefetchProgress(courseId)
  }

  const plan = buildTodayPlan({
    dueCount: reviewsQ.data ? reviewsQ.data.due_count : null,
    readiness,
    topics: topicsQ.data?.topics,
    recentPage: lastStudied,
    hasFiles,
  })
  // Topics have their own section skeleton, so Today only waits on its
  // primary remote sources (reviews + readiness) and the file count.
  const todayLoading = reviewsQ.isPending || readinessQ.isPending || filesQ.isLoading
  const todayError = reviewsQ.isError && readinessQ.isError
  const retryToday = () => {
    if (reviewsQ.isError) void reviewsQ.refetch()
    if (readinessQ.isError) void readinessQ.refetch()
  }

  return (
    <div className="max-w-6xl mx-auto px-6 py-9 space-y-8">
      {/* ── Header ───────────────────────────────────────── */}
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={ENTRANCE}
        className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-5"
      >
        <div className="min-w-0">
          <p className="text-[11px] font-medium uppercase tracking-[0.14em] text-ink-faint mb-2">Course</p>
          <h1 className="font-display text-[2rem] leading-tight font-semibold text-ink tracking-tight truncate">
            {course?.title ?? 'Course'}
          </h1>
          <div className="flex items-center gap-3 mt-2.5 text-sm text-ink-soft">
            <span className="inline-flex items-center gap-1.5">
              <FileText className="w-4 h-4 text-ink-faint" />
              {filesQ.isLoading ? (
                <span className="inline-block h-3.5 w-24 rounded bg-paper-deep animate-pulse align-middle" />
              ) : (
                <span>{hasFiles ? `${fileCount} file${fileCount !== 1 ? 's' : ''} indexed` : 'No materials yet'}</span>
              )}
            </span>
            {lastStudied && (
              <span className="inline-flex items-center gap-1.5 text-ink-faint">
                <Clock className="w-4 h-4" /> Last: <span className="text-ink-soft">{pageLabel(lastStudied)}</span>
              </span>
            )}
          </div>
        </div>
        <Button
          size="lg"
          leftIcon={<Sparkles className="w-4 h-4" />}
          onClick={() => go(lastStudied ?? 'learn')}
          onMouseEnter={() => prefetchFor(lastStudied ?? 'learn')}
          className="flex-shrink-0"
        >
          {lastStudied ? 'Continue studying' : 'Ask your course'}
        </Button>
      </motion.div>

      {/* ── Command center: main column + sticky rail ────── */}
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ ...ENTRANCE, delay: 0.05 }}
        className="grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_296px] gap-6 items-start"
      >
        <div className="space-y-8 min-w-0">
          <TodayPanel
            loading={todayLoading}
            error={todayError}
            onRetry={retryToday}
            items={plan.items}
            onboarding={plan.onboarding}
            totalMin={plan.totalMin}
            onGo={go}
            onPrefetch={prefetchFor}
          />

          <TopicsSection
            loading={topicsQ.isPending}
            error={topicsQ.isError}
            onRetry={() => void topicsQ.refetch()}
            topics={topicsQ.data?.topics ?? []}
            masteryFor={(name) => masteryForTopic(name, readiness)}
            onDrill={() => go('practice')}
            onAsk={() => go('learn')}
            onPrefetchDrill={() => prefetchFor('practice')}
            onPrefetchAsk={() => prefetchFor('learn')}
            onRebuild={() => rebuildTopics.mutate()}
            rebuilding={rebuildTopics.isPending}
            rebuildError={rebuildTopics.isError}
          />
        </div>

        <aside className="lg:sticky lg:top-6">
          <CourseRail
            readinessLoading={readinessQ.isPending}
            readinessError={readinessQ.isError}
            onRetryReadiness={() => void readinessQ.refetch()}
            score={score}
            hasFiles={hasFiles}
            streak={analyticsQ.data ? analyticsQ.data.study_streak : null}
            streakLoading={analyticsQ.isPending}
            streakError={analyticsQ.isError}
            fileCount={fileCount}
            filesLoading={filesQ.isLoading}
            onGo={go}
          />
        </aside>
      </motion.div>
    </div>
  )
}
