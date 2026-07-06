import { ClipboardList, FileText } from 'lucide-react'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { ProgressRing } from '@/components/ui/Progress'
import ErrorInline from '@/components/shared/ErrorInline'
import { scoreTone } from '@/lib/score'

interface CourseRailProps {
  readinessLoading: boolean
  readinessError: boolean
  onRetryReadiness: () => void
  /** Rounded readiness pct, or null when no score exists yet. */
  score: number | null
  hasFiles: boolean
  /** Day streak from analytics, or null when unknown. */
  streak: number | null
  streakLoading: boolean
  streakError: boolean
  fileCount: number
  filesLoading: boolean
  onGo: (to: string) => void
}

/**
 * The sticky right rail of the command center: readiness ring, streak,
 * materials count and the mock-exam CTA.
 */
export default function CourseRail({
  readinessLoading,
  readinessError,
  onRetryReadiness,
  score,
  hasFiles,
  streak,
  streakLoading,
  streakError,
  fileCount,
  filesLoading,
  onGo,
}: CourseRailProps) {
  return (
    <div className="space-y-4">
      <ReadinessCard
        loading={readinessLoading}
        error={readinessError}
        onRetry={onRetryReadiness}
        score={score}
        hasFiles={hasFiles}
        onGo={onGo}
      />

      {/* Streak + materials — dense stat rows */}
      <Card padding="none">
        <div className="divide-y divide-line">
          <div className="flex items-center gap-3 px-4 py-3">
            {streakLoading ? (
              <span className="h-3.5 w-24 rounded bg-paper-deep animate-pulse" aria-hidden />
            ) : streakError ? (
              <span className="text-sm text-ink-faint">Streak unavailable</span>
            ) : streak !== null && streak > 0 ? (
              <span className="text-sm text-ink">
                <span className="font-semibold tnum">{streak}</span>-day streak
              </span>
            ) : (
              <span className="text-sm text-ink-soft">No streak yet — study today</span>
            )}
          </div>
          <button
            type="button"
            onClick={() => onGo('materials')}
            className="w-full flex items-center gap-3 px-4 py-3 hover:bg-paper-deep/40 transition-colors focus-ring rounded-b-xl"
          >
            <FileText className="w-4 h-4 text-ink-faint flex-shrink-0" />
            {filesLoading ? (
              <span className="h-3.5 w-16 rounded bg-paper-deep animate-pulse flex-1 max-w-16 text-left" aria-hidden />
            ) : (
              <span className="text-sm text-ink flex-1 text-left">
                <span className="font-semibold tnum">{fileCount}</span> file{fileCount !== 1 ? 's' : ''}
              </span>
            )}
            <span className="text-xs font-medium text-accent-deep">Materials</span>
          </button>
        </div>
      </Card>

      {/* Exam CTA */}
      <Card padding="md">
        <div className="flex items-center gap-2.5">
          <ClipboardList className="w-4 h-4 text-ink-faint flex-shrink-0" />
          <p className="text-sm font-semibold text-ink">Ready to test yourself?</p>
        </div>
        <p className="text-xs text-ink-soft mt-1.5 leading-relaxed">
          A timed mock exam, graded with a per-concept breakdown.
        </p>
        <Button variant="secondary" size="sm" className="mt-3 w-full" onClick={() => onGo('exam')}>
          Sit a mock exam
        </Button>
      </Card>
    </div>
  )
}

function ReadinessCard({
  loading,
  error,
  onRetry,
  score,
  hasFiles,
  onGo,
}: {
  loading: boolean
  error: boolean
  onRetry: () => void
  score: number | null
  hasFiles: boolean
  onGo: (to: string) => void
}) {
  if (error) {
    return (
      <Card padding="md" elevation={2}>
        <ErrorInline message="Couldn't load readiness." onRetry={onRetry} />
      </Card>
    )
  }
  if (loading) {
    return (
      <Card padding="lg" elevation={2} className="flex flex-col items-center" aria-hidden>
        <div className="w-24 h-24 rounded-full bg-paper-deep animate-pulse" />
        <div className="h-3.5 w-24 rounded bg-paper-deep animate-pulse mt-4" />
        <div className="h-3 w-36 rounded bg-paper-deep animate-pulse mt-2" />
      </Card>
    )
  }
  if (score !== null) {
    const tone = scoreTone(score)
    return (
      <Card padding="lg" elevation={2} className="flex flex-col items-center text-center">
        <ProgressRing value={score} size={96}>
          <span className={`text-2xl font-bold tnum ${tone.text}`}>{score}%</span>
        </ProgressRing>
        <p className={`text-sm font-semibold mt-3 ${tone.text}`}>{tone.label}</p>
        <p className="text-xs text-ink-faint mt-1 leading-relaxed">
          Exam readiness, estimated from your topic mastery.
        </p>
      </Card>
    )
  }
  return (
    <Card padding="lg" elevation={2}>
      <p className="text-sm font-semibold text-ink">No readiness score yet</p>
      <p className="text-xs text-ink-soft mt-1.5 leading-relaxed">
        {hasFiles
          ? 'Answer a few questions and Vindexa starts estimating how ready you are.'
          : 'Add materials, then study — readiness builds from your activity.'}
      </p>
      <Button
        variant="secondary"
        size="sm"
        className="mt-3 w-full"
        onClick={() => onGo(hasFiles ? 'learn' : 'materials')}
      >
        {hasFiles ? 'Start learning' : 'Add materials'}
      </Button>
    </Card>
  )
}
