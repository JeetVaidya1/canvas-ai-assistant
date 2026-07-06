import { AnimatePresence, motion } from 'motion/react'
import { CheckCircle, Clock, PenLine, Trophy, ArrowRight } from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { Card } from '@/components/ui/Card'
import { Markdown } from '@/components/ui/Markdown'
import { LETTERS } from './constants'
import { formatTime, resolveOptionState } from './format'
import { SessionHeader } from './SessionHeader'
import { OptionRow } from './OptionRow'
import { FeedbackPanel } from './FeedbackPanel'
import { ConfidenceControl } from './ConfidenceControl'
import { SourceChip } from './SourceChip'
import { ConceptTag } from './Tags'
import type { QuizController } from './useQuizRun'

/** Honest waiting card shown only when the user outruns background generation. */
function AwaitingQuestion({ quiz }: { quiz: QuizController }) {
  const generating = quiz.run?.generationStatus === 'generating'
  return (
    <Card padding="lg">
      <div className="flex items-center gap-3 mb-4">
        <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-xl border border-accent-line bg-accent-wash">
          <PenLine className="h-4.5 w-4.5 text-accent" />
        </div>
        <div>
          <h3 className="text-[15px] font-semibold text-ink">
            {generating ? 'Writing your next question…' : 'Wrapping up your quiz…'}
          </h3>
          <p className="text-sm text-ink-soft">
            {generating
              ? 'You answered faster than we could write — it lands in a few seconds.'
              : 'Generation has finished; scoring what you answered.'}
          </p>
        </div>
      </div>
      <div
        role="status"
        aria-label={generating ? 'Writing your next question' : 'Scoring your quiz'}
        className="h-1.5 overflow-hidden rounded-full border border-line/60 bg-paper-deep"
      >
        <div className="h-full w-1/4 rounded-full bg-accent animate-indeterminate" />
      </div>
      {!generating && !quiz.submitting && (
        <Button onClick={() => void quiz.finishNow()} size="lg" className="mt-5 w-full">
          Score my quiz
        </Button>
      )}
    </Card>
  )
}

/**
 * Active quiz run: honest progress strip (available vs requested), animated
 * question card, graded options, optional confidence tap, cited feedback.
 * Starts on the first ~3 questions while the rest stream in behind it.
 */
export function QuizSession({ quiz }: { quiz: QuizController }) {
  if (!quiz.run) return null
  const run = quiz.run
  const question = run.questions[run.currentIndex] as (typeof run.questions)[number] | undefined
  const feedback = run.feedback
  const generating = run.generationStatus === 'generating'
  // While generating, the requested count is the honest denominator; once
  // terminal, what exists is all there will be.
  const totalPlanned = generating ? run.numRequested : Math.max(run.questions.length, 1)
  const answered = run.currentIndex + (feedback ? 1 : 0)
  const progress = (answered / totalPlanned) * 100
  const isLast =
    run.currentIndex + 1 >= run.numRequested ||
    (!generating && run.currentIndex + 1 >= run.questions.length)

  return (
    <div className="mx-auto max-w-2xl px-5 py-7">
      <SessionHeader
        itemLabel="Question"
        index={run.currentIndex}
        total={totalPlanned}
        availability={generating ? `${run.questions.length} ready` : null}
        meta={`${run.topicLabel} · ${quiz.difficulty}`}
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

      {/* Question card — animated per-question; waiting card if we outran generation */}
      <AnimatePresence mode="wait">
        <motion.div
          key={question ? question.id : `awaiting-${run.currentIndex}`}
          initial={{ opacity: 0, x: 24 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -24 }}
          transition={{ duration: 0.28, ease: 'easeOut' }}
        >
          {!question ? (
            <AwaitingQuestion quiz={quiz} />
          ) : (
            <Card accent={!!feedback} padding="lg">
              <div className="flex flex-wrap items-center gap-2 mb-4">
                <ConceptTag concept={question.concept} />
                <SourceChip source={question.source} />
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

              {/* Confidence tap — appears once an option is picked, before Submit */}
              {!feedback && run.selectedLetter && (
                <div className="mb-5 animate-fade-up">
                  <ConfidenceControl
                    value={run.confidence}
                    onChange={quiz.setConfidence}
                    disabled={quiz.submitting}
                  />
                </div>
              )}

              <FeedbackPanel
                show={!!feedback}
                correct={feedback?.is_correct ?? false}
                explanation={feedback?.explanation ?? ''}
              >
                {feedback?.source?.doc_name && (
                  <div className="mt-3 flex flex-wrap items-center gap-2">
                    <ConceptTag concept={feedback.concept} />
                    <SourceChip source={feedback.source} />
                  </div>
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
                  {quiz.submitting ? 'Scoring…' : isLast ? 'See Your Debrief' : 'Next Question'}
                </Button>
              )}
            </Card>
          )}
        </motion.div>
      </AnimatePresence>
    </div>
  )
}
