import { Play, Gauge, Library, RefreshCw } from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { Card } from '@/components/ui/Card'
import { EmptyState } from '@/components/ui/States'
import { SetupShell, FieldLabel } from './SetupShell'
import { DifficultyTiles } from './DifficultyTiles'
import { CountSelector } from './CountSelector'
import { TopicField } from './TopicField'
import { PRACTICE_DIFFICULTIES, PRACTICE_COUNTS } from './constants'
import type { PracticeController } from './usePracticeSession'

const SUBTITLE =
  'Open-ended problems generated fresh from your course. Difficulty adapts to your mastery, problem by problem.'

/** Inline offer to pick an interrupted problem set back up (or let it go). */
function ResumeSnapshotBar({ practice }: { practice: PracticeController }) {
  const snap = practice.snapshot
  if (!snap) return null
  const answered = snap.userAnswers.filter((a) => a !== '').length
  return (
    <Card padding="none" className="mb-7 flex items-center gap-3 pl-4 pr-2 py-2.5 animate-fade-up">
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-medium text-ink">Pick up your unfinished set</p>
        <p className="mt-0.5 text-xs text-ink-faint">
          {snap.topic || 'Practice'} ·{' '}
          <span className="tnum">
            {answered}/{snap.problems.length}
          </span>{' '}
          answered
        </p>
      </div>
      <Button
        size="sm"
        variant="secondary"
        onClick={practice.resumeSnapshot}
        leftIcon={<Play className="h-3.5 w-3.5" />}
        className="flex-shrink-0"
      >
        Resume
      </Button>
      <Button size="sm" variant="ghost" onClick={practice.discardSnapshot} className="flex-shrink-0">
        Discard
      </Button>
    </Card>
  )
}

/** Center-first Problem Set setup: adaptive difficulty tiles, count, topic, one CTA. */
export function ProblemSetSetup({ practice }: { practice: PracticeController }) {
  const { topics } = practice

  // Course is indexed but produced no topics — problems need materials.
  if (topics.empty) {
    return (
      <SetupShell title="Set up your problem set" subtitle={SUBTITLE}>
        <Card padding="lg">
          <EmptyState
            icon={<Library />}
            title="No topics to practice yet"
            description="Upload course materials so Vindexa can index topics and generate problems grounded in your own content."
            action={
              <Button
                variant="secondary"
                size="sm"
                onClick={topics.refetch}
                loading={topics.loading}
                leftIcon={<RefreshCw className="w-3.5 h-3.5" />}
              >
                Check again
              </Button>
            }
          />
        </Card>
      </SetupShell>
    )
  }

  return (
    <SetupShell title="Set up your problem set" subtitle={SUBTITLE}>
      <ResumeSnapshotBar practice={practice} />

      <div className="mb-6">
        <FieldLabel center>Difficulty</FieldLabel>
        <DifficultyTiles
          options={PRACTICE_DIFFICULTIES}
          value={practice.difficulty}
          onChange={practice.setDifficulty}
          label="Problem difficulty"
          className="grid-cols-2 sm:grid-cols-4"
        />
      </div>

      <div className="mb-6">
        <FieldLabel center>Problems</FieldLabel>
        <CountSelector
          counts={PRACTICE_COUNTS}
          value={practice.problemCount}
          onChange={practice.setProblemCount}
          label="Number of problems"
        />
      </div>

      <div className="mb-6">
        <TopicField
          options={topics.options}
          value={practice.selectedTopic}
          onChange={practice.setSelectedTopic}
          loading={topics.loading}
          pending={topics.pending}
          error={topics.error}
          onRetry={topics.refetch}
          disabled={!practice.courseId}
          ariaLabel="Practice topic"
        />
      </div>

      {practice.difficulty === 'adaptive' ? (
        <div className="mb-8 flex items-start gap-2.5 rounded-xl border border-accent-line bg-accent-wash px-3.5 py-3">
          <Gauge className="mt-0.5 h-4 w-4 flex-shrink-0 text-accent" />
          <p className="text-xs text-accent-deep">
            Adaptive mode reads your recent mastery and calibrates each problem's difficulty —
            you'll see the resolved level on every card.
          </p>
        </div>
      ) : (
        <div className="mb-8" />
      )}

      <Button
        onClick={() => void practice.startSession()}
        disabled={practice.loading || topics.loading || !practice.selectedTopic || !practice.courseId}
        loading={practice.loading}
        size="lg"
        leftIcon={<Play className="h-4 w-4" />}
        className="w-full !py-3.5 !text-base"
      >
        {practice.loading ? 'Generating…' : 'Start session'}
      </Button>
    </SetupShell>
  )
}
