import { AnimatePresence, motion } from 'motion/react'
import { CheckCircle, Clock, Gauge, Globe, Library, Trophy, ArrowRight } from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { Card } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { Markdown } from '@/components/ui/Markdown'
import {
  formatTime,
  formatEstimatedTime,
  resolveDifficultyBadge,
  resolveProblemSource,
  resolveOptionState,
} from './format'
import { SessionHeader } from './SessionHeader'
import { OptionRow } from './OptionRow'
import { FeedbackPanel } from './FeedbackPanel'
import type { PracticeController } from './usePracticeSession'

const letterFor = (index: number) => String.fromCharCode(65 + index)

/** Active problem-set run: progress strip, metadata badges, options, solution reveal. */
export function ProblemCard({ practice }: { practice: PracticeController }) {
  const session = practice.session
  if (!session) return null
  const problem = session.problems[session.currentProblemIndex]
  if (!problem) return null

  const revealed = practice.showExplanation
  const isCorrect = practice.selectedAnswer === problem.correct_answer
  const isLast = session.currentProblemIndex === session.problems.length - 1
  const progress = ((session.currentProblemIndex + 1) / session.problems.length) * 100
  const diff = resolveDifficultyBadge(problem.difficulty)
  const source = resolveProblemSource(problem)

  return (
    <div className="mx-auto max-w-2xl px-5 py-7">
      <SessionHeader
        itemLabel="Problem"
        index={session.currentProblemIndex}
        total={session.problems.length}
        meta={`${practice.selectedTopic} · ${practice.difficulty}`}
        progress={progress}
        right={
          <span className="inline-flex items-center gap-1.5 text-sm text-ink-soft tnum">
            <Clock className="h-4 w-4 text-ink-faint" />
            {formatTime(practice.timeElapsed)}
          </span>
        }
      />

      {/* Problem card — animates in on each question change */}
      <AnimatePresence mode="wait">
        <motion.div
          key={session.currentProblemIndex}
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -16 }}
          transition={{ duration: 0.28, ease: 'easeOut' }}
        >
          <Card accent padding="lg">
            {/* Metadata: resolved difficulty + grounding + estimated time */}
            <div className="flex flex-wrap items-center gap-2 mb-4">
              <Badge tone={diff.tone} icon={<Gauge />}>
                {diff.label}
              </Badge>
              {source === 'materials' ? (
                <Badge tone="accent" icon={<Library />}>
                  From your materials
                </Badge>
              ) : (
                <Badge tone="neutral" icon={<Globe />}>
                  General knowledge
                </Badge>
              )}
              <Badge tone="neutral" icon={<Clock />}>
                {formatEstimatedTime(problem.estimated_time)}
              </Badge>
            </div>

            <div className="text-xl font-medium leading-snug text-ink mb-6">
              <Markdown content={problem.question} />
            </div>

            <div className="space-y-2.5 mb-5">
              {problem.options.map((option, index) => {
                const letter = letterFor(index)
                return (
                  <OptionRow
                    key={index}
                    letter={letter}
                    text={option}
                    state={resolveOptionState(
                      letter,
                      practice.selectedAnswer,
                      revealed,
                      revealed ? problem.correct_answer : null,
                    )}
                    revealed={revealed}
                    onSelect={() => practice.selectAnswer(letter)}
                  />
                )
              })}
            </div>

            <FeedbackPanel show={revealed} correct={isCorrect} explanation={problem.explanation} />

            {!revealed ? (
              <Button
                onClick={practice.submitAnswer}
                disabled={!practice.selectedAnswer}
                size="lg"
                leftIcon={<CheckCircle className="w-4 h-4" />}
                className="w-full"
              >
                Submit Answer
              </Button>
            ) : (
              <Button
                onClick={practice.nextProblem}
                size="lg"
                rightIcon={isLast ? <Trophy className="w-4 h-4" /> : <ArrowRight className="w-4 h-4" />}
                className="w-full"
              >
                {isLast ? 'View Results' : 'Next Question'}
              </Button>
            )}
          </Card>
        </motion.div>
      </AnimatePresence>
    </div>
  )
}
