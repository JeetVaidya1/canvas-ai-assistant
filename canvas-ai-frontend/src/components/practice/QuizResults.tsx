import { motion } from 'motion/react'
import { useNavigate } from 'react-router-dom'
import { ArrowLeft, CheckCircle, Info, RotateCcw } from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { Card } from '@/components/ui/Card'
import { ScorePanel } from './debrief/ScorePanel'
import { CalibrationPanel } from './debrief/CalibrationPanel'
import { TopicBreakdown } from './debrief/TopicBreakdown'
import { MistakeList } from './debrief/MistakeList'
import type { QuizController } from './useQuizRun'

/** Numbered syllabus-style header for one debrief section. */
function SectionHead({ num, title }: { num: string; title: string }) {
  return (
    <div className="section-head mb-4">
      <span className="section-num">{num}</span>
      <h3 className="text-sm font-semibold text-ink">{title}</h3>
    </div>
  )
}

/**
 * Quiz debrief — a working document, not a trophy screen: score, confidence
 * calibration, per-topic accuracy, every mistake replayed with its source,
 * and one clear next action.
 */
export function QuizResults({ quiz }: { quiz: QuizController }) {
  const navigate = useNavigate()
  const result = quiz.result
  if (!result) return null

  const run = quiz.run
  const mistakes = run?.answers.filter((a) => !a.result.is_correct) ?? []
  const perfect = result.score.correct === result.score.total && result.score.total > 0
  const shortfall =
    run && run.generationStatus === 'partial' && run.questions.length < run.numRequested
      ? run.questions.length
      : null

  // Weakest ground first: backend's weak areas, else the lowest-scoring topic.
  const sortedTopics = [...result.by_topic].sort((a, b) => a.pct - b.pct)
  const weakTopic = result.weak_areas[0] ?? sortedTopics.find((t) => t.pct < 100)?.topic ?? null

  return (
    <div className="mx-auto max-w-3xl px-5 py-8">
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, ease: 'easeOut' }}
        className="space-y-5"
      >
        <div>
          <h2 className="font-display text-2xl font-semibold text-ink">Debrief</h2>
          <p className="mt-0.5 text-sm text-ink-soft">
            {run?.topicLabel ?? quiz.selectedTopic} &middot; {quiz.difficulty}
          </p>
        </div>

        {shortfall !== null && (
          <div className="flex items-start gap-2.5 rounded-lg border border-info/25 bg-info-wash p-3.5">
            <Info className="mt-0.5 h-4 w-4 flex-shrink-0 text-info" />
            <p className="text-sm text-ink-soft">
              We could only write {shortfall} question{shortfall === 1 ? '' : 's'} this time — your
              score covers what you answered.
            </p>
          </div>
        )}

        <Card padding="lg">
          <SectionHead num="01" title="Score" />
          <ScorePanel
            pct={result.score.pct}
            correct={result.score.correct}
            total={result.score.total}
            timeElapsed={quiz.timeElapsed}
          />
        </Card>

        <Card padding="lg">
          <SectionHead num="02" title="Calibration" />
          <CalibrationPanel calibration={result.calibration} />
        </Card>

        {result.by_topic.length > 0 && (
          <Card padding="lg">
            <SectionHead num="03" title="By topic" />
            <TopicBreakdown byTopic={result.by_topic} />
          </Card>
        )}

        {(mistakes.length > 0 || perfect) && (
          <Card padding="lg">
            <SectionHead num="04" title="Mistakes" />
            {perfect ? (
              <p className="flex items-center gap-2 text-sm text-success">
                <CheckCircle className="h-4 w-4 flex-shrink-0" />
                Clean sheet — nothing to review from this run.
              </p>
            ) : (
              <MistakeList mistakes={mistakes} />
            )}
          </Card>
        )}

        <div className="flex flex-wrap gap-3">
          <Button
            onClick={() => void quiz.startQuiz(weakTopic ?? undefined)}
            loading={quiz.loading}
            leftIcon={<RotateCcw className="h-4 w-4" />}
          >
            {quiz.loading ? 'Generating…' : weakTopic ? 'Drill weak areas again' : 'Drill again'}
          </Button>
          <Button
            variant="secondary"
            onClick={() => void navigate(`/course/${quiz.courseId}`)}
            leftIcon={<ArrowLeft className="h-4 w-4" />}
          >
            Back to course
          </Button>
        </div>
      </motion.div>
    </div>
  )
}
