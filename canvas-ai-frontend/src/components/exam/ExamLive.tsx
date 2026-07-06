// Live exam screen: sticky paper timer bar (solid paper + hairline, mono
// digits), focused question column with MCQ / free-response inputs, hint &
// solution panes, bottom navigation, the submit-confirm Modal, and the
// slide-over QuestionNavigator.
import { useState } from 'react'
import { motion, AnimatePresence } from 'motion/react'
import {
  ChevronLeft,
  ChevronRight,
  Eye,
  Flag,
  LayoutGrid,
  Pause,
  Play,
  Timer,
} from 'lucide-react'
import { Badge } from '@/components/ui/Badge'
import { Button } from '@/components/ui/Button'
import { Textarea } from '@/components/ui/Input'
import { Markdown } from '@/components/ui/Markdown'
import { Modal } from '@/components/ui/Modal'
import { formatTime } from './examMeta'
import { QuestionNavigator } from './QuestionNavigator'
import type { ExamController } from './useExamSession'
import type { SolveJSON } from './types'

interface ExamLiveProps {
  exam: ExamController
}

/** Steps array → a markdown bullet list (keeps math/code rendering intact). */
function stepsToMarkdown(s: SolveJSON): string {
  return (s.steps ?? []).map((step) => `- ${step}`).join('\n')
}

export function ExamLive({ exam }: ExamLiveProps) {
  const session = exam.examSession
  // Question navigator lives in a slide-over panel rather than a competing card.
  const [navOpen, setNavOpen] = useState(false)
  const [confirmOpen, setConfirmOpen] = useState(false)

  if (!session) return null

  const currentQ = session.questions[session.currentQuestion]
  const progress = ((session.currentQuestion + 1) / session.questions.length) * 100
  const answeredCount = Object.keys(session.userAnswers).filter((k) => session.userAnswers[k]).length
  const unansweredCount = session.questions.length - answeredCount
  const timeLow = exam.timeRemaining < 300
  const timeCritical = exam.timeRemaining < 60
  const timerTone = timeCritical
    ? 'border-danger/25 bg-danger-wash text-danger'
    : timeLow
    ? 'border-warning/25 bg-warning-wash text-warning'
    : 'border-line bg-surface text-ink'
  const isLast = session.currentQuestion === session.questions.length - 1

  const hint = exam.hints[currentQ.id]
  const solution = exam.solutions[currentQ.id]

  const slideVariants = {
    enter: (dir: number) => ({ opacity: 0, x: dir > 0 ? 28 : -28 }),
    center: { opacity: 1, x: 0 },
    exit: (dir: number) => ({ opacity: 0, x: dir > 0 ? -28 : 28 }),
  }

  const confirmSubmit = async () => {
    await exam.submitExam()
    setConfirmOpen(false)
  }

  return (
    <div className="relative flex min-h-full flex-col">
      {/* Slim sticky top bar — solid paper strip with a hairline, like the app's .top-bar */}
      <div className="sticky top-0 z-20 top-bar">
        <div className="mx-auto flex max-w-3xl items-center justify-between gap-4 px-5 py-2.5">
          <div className="min-w-0">
            <h2 className="text-sm font-semibold text-ink truncate">{session.examName}</h2>
            <p className="section-num mt-0.5 truncate">
              Question {String(session.currentQuestion + 1).padStart(2, '0')} of {String(session.questions.length).padStart(2, '0')} &middot; {answeredCount} answered
            </p>
          </div>
          <div className="flex items-center gap-2 flex-shrink-0">
            <motion.div
              animate={timeLow && !session.isPaused ? { scale: [1, 1.04, 1] } : { scale: 1 }}
              transition={timeLow ? { repeat: Infinity, duration: 1.4 } : { duration: 0.2 }}
              className={`flex items-center gap-2 px-3.5 py-2 rounded-lg border ${timerTone}`}
            >
              <Timer className="w-4 h-4" />
              <span className="font-mono tnum text-lg font-semibold tracking-tight">{formatTime(exam.timeRemaining)}</span>
            </motion.div>
            <button
              onClick={() => setNavOpen(true)}
              className="inline-flex items-center gap-1.5 rounded-lg border border-line bg-surface px-2.5 py-2 text-[13px] text-ink-soft transition-colors hover:bg-surface-hover hover:border-line-strong hover:text-ink"
              aria-label="Open question navigator"
            >
              <LayoutGrid className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">Questions</span>
            </button>
            <Button
              variant="secondary"
              size="sm"
              onClick={exam.pauseExam}
              leftIcon={session.isPaused ? <Play className="w-3.5 h-3.5" /> : <Pause className="w-3.5 h-3.5" />}
            >
              {session.isPaused ? 'Resume' : 'Pause'}
            </Button>
          </div>
        </div>
        {/* Thin progress bar */}
        <div className="h-0.5 w-full bg-line/60">
          <motion.div
            className="h-0.5 bg-accent"
            animate={{ width: `${progress}%` }}
            transition={{ duration: 0.35, ease: 'easeOut' }}
          />
        </div>
      </div>

      {/* Focused reading column */}
      <div className="mx-auto w-full max-w-2xl flex-1 px-5 py-7">
        {/* Paused overlay banner */}
        <AnimatePresence>
          {session.isPaused && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="mb-5 overflow-hidden"
            >
              <div className="flex items-center gap-2.5 rounded-xl border border-warning/25 bg-warning-wash px-4 py-3">
                <Pause className="w-4 h-4 text-warning flex-shrink-0" />
                <span className="text-sm text-warning">Exam paused — the timer is frozen. Resume when you're ready.</span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Distraction-free question view with slide transition */}
        <div className="relative" style={{ minHeight: 320 }}>
          <AnimatePresence mode="wait" custom={exam.navDirection}>
            <motion.div
              key={currentQ.id}
              custom={exam.navDirection}
              variants={slideVariants}
              initial="enter"
              animate="center"
              exit="exit"
              transition={{ duration: 0.28, ease: 'easeOut' }}
            >
              {/* Meta tags — one Badge system, no hand-rolled pills */}
              <div className="flex items-center gap-2 mb-5">
                <Badge tone="neutral">{currentQ.points} pt{currentQ.points !== 1 ? 's' : ''}</Badge>
                <Badge tone="neutral">{currentQ.topic}</Badge>
                <Badge tone="neutral" className="capitalize">{currentQ.difficulty}</Badge>
              </div>

              {/* Large readable question text */}
              <div className="text-lg sm:text-xl font-medium text-ink leading-relaxed mb-7">
                <Markdown content={currentQ.question} />
              </div>

              {/* Answer area — prominent */}
              {currentQ.type === 'multiple_choice' && currentQ.options ? (
                <div className="space-y-2.5 mb-6">
                  {currentQ.options.map((option, index) => {
                    const letter = String.fromCharCode(65 + index)
                    const isSelected = exam.currentAnswer === letter
                    return (
                      <motion.button
                        key={index}
                        whileTap={{ scale: 0.99 }}
                        onClick={() => exam.setCurrentAnswer(letter)}
                        aria-pressed={isSelected}
                        className={`w-full p-4 border rounded-xl text-left transition-all text-[15px] ${
                          isSelected
                            ? 'border-accent bg-accent-wash ring-2 ring-accent/20'
                            : 'card-surface hover:border-accent-line hover:bg-surface-hover'
                        }`}
                      >
                        <div className="flex items-center gap-3.5">
                          <div className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0 transition-colors ${
                            isSelected ? 'bg-accent text-white' : 'bg-paper-deep text-ink-soft'
                          }`}>
                            {letter}
                          </div>
                          <span className="flex-1 text-ink">{option}</span>
                        </div>
                      </motion.button>
                    )
                  })}
                </div>
              ) : (
                <div className="mb-6">
                  <Textarea
                    value={exam.currentAnswer}
                    onChange={(e) => exam.setCurrentAnswer(e.target.value)}
                    placeholder="Write your answer here…"
                    rows={6}
                    aria-label="Your answer"
                    className="!text-[15px] leading-relaxed resize-none p-4"
                  />
                </div>
              )}

              {/* Hint / Solution — subtle secondary actions */}
              <div className="flex items-center gap-3 mb-5">
                <button
                  onClick={exam.requestHint}
                  disabled={exam.hinting || !exam.courseId}
                  className="inline-flex items-center gap-1.5 text-xs font-medium text-ink-soft hover:text-warning disabled:opacity-50 transition-colors"
                >
                  {exam.hinting ? 'Getting hint…' : 'Get hint'}
                </button>
                <span className="text-ink-faint">·</span>
                <button
                  onClick={exam.requestSolution}
                  disabled={exam.solving || !exam.courseId}
                  className="inline-flex items-center gap-1.5 text-xs font-medium text-ink-soft hover:text-success disabled:opacity-50 transition-colors"
                >
                  <Eye className="w-3.5 h-3.5" />
                  {exam.solving ? 'Solving…' : 'Show solution'}
                </button>
              </div>

              {/* Hint / Solution panes */}
              <AnimatePresence>
                {hint && (
                  <motion.div
                    initial={{ opacity: 0, y: -6 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                    className="mb-4 rounded-xl border border-warning/25 bg-warning-wash p-3.5"
                  >
                    <div className="text-xs font-medium text-warning mb-1.5">Hint</div>
                    <Markdown content={stepsToMarkdown(hint)} className="text-xs" />
                  </motion.div>
                )}

                {solution && (
                  <motion.div
                    initial={{ opacity: 0, y: -6 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                    className="mb-4 rounded-xl border border-success/25 bg-success-wash p-3.5"
                  >
                    <div className="text-xs font-medium text-success mb-1.5">Solution</div>
                    {solution.choice && (
                      <div className="mb-1 text-xs text-success">Choice: <strong>{solution.choice}</strong></div>
                    )}
                    <div className="mb-1.5 text-xs text-success">
                      Answer: <strong>{solution.final_answer}</strong> {solution.units ? `[${solution.units}]` : ''}
                    </div>
                    <Markdown content={stepsToMarkdown(solution)} className="text-xs" />
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          </AnimatePresence>
        </div>

        {/* Bottom navigation — Previous / Next, Submit prominent only on the last question */}
        <div className="mt-6 flex items-center justify-between border-t border-line pt-5">
          <Button
            variant="secondary"
            onClick={exam.previousQuestion}
            disabled={session.currentQuestion === 0}
            leftIcon={<ChevronLeft className="w-4 h-4" />}
          >
            Previous
          </Button>

          <div className="flex items-center gap-2">
            {!isLast && (
              <button
                onClick={() => setConfirmOpen(true)}
                className="text-xs text-ink-faint transition-colors hover:text-ink"
              >
                Finish early
              </button>
            )}
            {isLast ? (
              <Button
                onClick={() => setConfirmOpen(true)}
                leftIcon={<Flag className="w-4 h-4" />}
              >
                Submit exam
              </Button>
            ) : (
              <Button
                onClick={exam.nextQuestion}
                rightIcon={<ChevronRight className="w-4 h-4" />}
              >
                Next
              </Button>
            )}
          </div>
        </div>
      </div>

      {/* Submit confirmation — the timer's auto-submit at 0:00 bypasses this */}
      <Modal
        open={confirmOpen}
        onClose={() => setConfirmOpen(false)}
        title="Submit exam?"
        description="Your answers will be sent to the AI judge for grading. You can't change them afterwards."
        size="sm"
        locked={exam.submitting}
        footer={
          <>
            <Button variant="secondary" onClick={() => setConfirmOpen(false)} disabled={exam.submitting}>
              Keep working
            </Button>
            <Button
              onClick={confirmSubmit}
              loading={exam.submitting}
              leftIcon={<Flag className="w-3.5 h-3.5" />}
            >
              {exam.submitting ? 'Grading…' : 'Submit for grading'}
            </Button>
          </>
        }
      >
        {unansweredCount > 0 ? (
          <p className="text-sm text-warning">
            {unansweredCount} of {session.questions.length} questions {unansweredCount === 1 ? 'is' : 'are'} still
            unanswered — they'll score zero points.
          </p>
        ) : (
          <p className="text-sm text-ink-soft">
            All {session.questions.length} questions answered with <span className="tnum">{formatTime(exam.timeRemaining)}</span> left on the clock.
          </p>
        )}
      </Modal>

      {/* Question navigator — slide-over panel */}
      <QuestionNavigator
        open={navOpen}
        onClose={() => setNavOpen(false)}
        session={session}
        onGoToQuestion={exam.goToQuestion}
        onRequestSubmit={() => setConfirmOpen(true)}
      />
    </div>
  )
}
