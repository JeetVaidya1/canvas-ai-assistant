import { Zap } from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { SetupShell, FieldLabel } from './SetupShell'
import { DifficultyTiles } from './DifficultyTiles'
import { CountSelector } from './CountSelector'
import { TopicField } from './TopicField'
import { QUIZ_DIFFICULTIES, QUIZ_COUNTS, WHOLE_COURSE } from './constants'
import type { QuizController } from './useQuizRun'

/** Center-first Quick Quiz setup: difficulty tiles, count, topic, one CTA. */
export function QuizSetup({ quiz }: { quiz: QuizController }) {
  const { topics } = quiz
  return (
    <SetupShell
      title="Set up your quiz drill"
      subtitle="Rapid multiple-choice, graded the instant you answer. Pick your focus and go."
    >
      <div className="mb-6">
        <FieldLabel center>Difficulty</FieldLabel>
        <DifficultyTiles
          options={QUIZ_DIFFICULTIES}
          value={quiz.difficulty}
          onChange={quiz.setDifficulty}
          label="Quiz difficulty"
          className="grid-cols-3"
        />
      </div>

      <div className="mb-6">
        <FieldLabel center>Questions</FieldLabel>
        <CountSelector
          counts={QUIZ_COUNTS}
          value={quiz.questionCount}
          onChange={quiz.setQuestionCount}
          label="Number of questions"
        />
      </div>

      <div className="mb-8">
        <TopicField
          options={topics.options}
          value={quiz.selectedTopic}
          onChange={quiz.setSelectedTopic}
          loading={topics.loading}
          pending={topics.pending}
          error={topics.error}
          onRetry={topics.refetch}
          disabled={!quiz.courseId}
          ariaLabel="Quiz topic"
          helper={
            quiz.selectedTopic === WHOLE_COURSE
              ? 'Pulls core concepts broadly from across the entire course.'
              : null
          }
        />
      </div>

      <Button
        onClick={() => void quiz.startQuiz()}
        disabled={quiz.loading || topics.loading || !quiz.selectedTopic || !quiz.courseId}
        loading={quiz.loading}
        size="lg"
        leftIcon={<Zap className="h-4 w-4" />}
        className="w-full !py-3.5 !text-base"
      >
        {quiz.loading ? 'Generating your drill…' : 'Start drill'}
      </Button>
      {quiz.loading && (
        <p className="mt-3 text-center text-xs text-ink-faint">
          Retrieving from your materials, reranking, and writing questions — this can take a moment.
        </p>
      )}
    </SetupShell>
  )
}
