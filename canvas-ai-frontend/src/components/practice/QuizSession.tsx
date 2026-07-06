import { AnimatePresence, motion } from 'motion/react'
import { CheckCircle, Clock, Target, Trophy, ArrowRight } from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { Card } from '@/components/ui/Card'
import { Markdown } from '@/components/ui/Markdown'
import { LETTERS } from './constants'
import { formatTime, resolveOptionState } from './format'
import { SessionHeader } from './SessionHeader'
import { OptionRow } from './OptionRow'
import { FeedbackPanel } from './FeedbackPanel'
import { SourceTag, ConceptTag } from './Tags'
import type { QuizController } from './useQuizRun'

/** Active quiz run: progress strip, animated question card, graded options, grounded feedback. */
export function QuizSession({ quiz }: { quiz: QuizController }) {
  if (!quiz.run) return null
  const run = quiz.run
  const question = run.questions[run.currentIndex]
  const feedback = run.feedback
  const progress = ((run.currentIndex + (feedback ? 1 : 0)) / run.questions.length) * 100
  const isLast = run.currentIndex === run.questions.length - 1

  return (
    <div className="mx-auto max-w-2xl px-5 py-7">
      <SessionHeader
        itemLabel="Question"
        index={run.currentIndex}
        total={run.questions.length}
        meta={`${quiz.selectedTopic} · ${quiz.difficulty}`}
        progress={progress}
        right={
          <>
            <span className="inline-flex items-center gap-1.5 text-sm text-success tnum">
              <CheckCircle className="h-4 w-4" />
              {run.correctCount}
            </span>
            <span className="inline-flex items-center gap-1.5 text-sm text-ink-soft tnum">
              <Clock className="h-4 w-4 text-ink-faint" />
              {formatTime(quiz.timeElapsed)}
            </span>
          </>
        }
      />

      {/* Question card — animated per-question */}
      <AnimatePresence mode="wait">
        <motion.div
          key={run.currentIndex}
          initial={{ opacity: 0, x: 24 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -24 }}
          transition={{ duration: 0.28, ease: 'easeOut' }}
        >
          <Card accent={!!feedback} padding="lg">
            <div className="flex flex-wrap items-center gap-2 mb-4">
              <ConceptTag concept={question.concept} />
              <SourceTag source={question.source} />
            </div>

            <div className="text-xl font-medium leading-snug text-ink mb-6">
              <Markdown content={question.question} />
            </div>

            <div className="space-y-2.5 mb-5">
              {question.options.map((option, index) => {
                const letter = LETTERS[index] ?? String.fromCharCode(65 + index)
                return (
                  <OptionRow
                    key={index}
                    letter={letter}
                    // Options already carry their "A) " prefix from the backend.
                    text={option.replace(/^[A-D]\)\s*/, '')}
                    state={resolveOptionState(
                      letter,
                      run.selectedLetter,
                      !!feedback,
                      feedback?.correct_answer ?? null,
                    )}
                    revealed={!!feedback}
                    onSelect={() => quiz.selectLetter(letter)}
                  />
                )
              })}
            </div>

            <FeedbackPanel
              show={!!feedback}
              correct={feedback?.is_correct ?? false}
              explanation={feedback?.explanation ?? ''}
            >
              {feedback && (
                <>
                  {/* Cited mistake explanation — grounded in the user's own pages. */}
                  {!feedback.is_correct && feedback.mistake_explanation && (
                    <div className="mt-3 rounded-lg bg-warning-wash border border-warning/25 p-3">
                      <div className="text-xs font-semibold text-warning mb-1 flex items-center gap-1.5">
                        <Target className="w-3.5 h-3.5" />
                        Why you missed this
                      </div>
                      <div className="text-sm text-ink-soft">
                        <Markdown content={feedback.mistake_explanation} />
                      </div>
                      {feedback.mistake_source?.doc_name && (
                        <div className="mt-2">
                          <SourceTag source={feedback.mistake_source} label="From" />
                        </div>
                      )}
                    </div>
                  )}
                  {feedback.source?.doc_name && (
                    <div className="mt-3 flex flex-wrap gap-2">
                      <ConceptTag concept={feedback.concept} />
                      <SourceTag source={feedback.source} label="Source" />
                    </div>
                  )}
                </>
              )}
            </FeedbackPanel>

            {!feedback ? (
              <Button
                onClick={() => void quiz.submitAnswer()}
                disabled={!run.selectedLetter || quiz.submitting}
                loading={quiz.submitting}
                size="lg"
                leftIcon={<CheckCircle className="w-4 h-4" />}
                className="w-full"
              >
                {quiz.submitting ? 'Checking…' : 'Submit Answer'}
              </Button>
            ) : (
              <Button
                onClick={() => void quiz.nextQuestion()}
                disabled={quiz.submitting}
                loading={quiz.submitting}
                size="lg"
                rightIcon={isLast ? <Trophy className="w-4 h-4" /> : <ArrowRight className="w-4 h-4" />}
                className="w-full"
              >
                {quiz.submitting ? 'Scoring…' : isLast ? 'See Your Results' : 'Next Question'}
              </Button>
            )}
          </Card>
        </motion.div>
      </AnimatePresence>
    </div>
  )
}
