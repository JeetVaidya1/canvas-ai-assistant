import { useState, useEffect, useMemo } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import {
  CheckCircle,
  XCircle,
  Clock,
  Target,
  Trophy,
  RotateCcw,
  Brain,
  Zap,
  ArrowRight,
  BookOpen,
  RefreshCw,
  FileText,
  Sparkles,
} from 'lucide-react'
import { BrandMark } from '@/components/ui/BrandMark'

import { Markdown } from '@/components/ui/Markdown'
import { Button } from '@/components/ui/Button'
import { Card } from '@/components/ui/Card'
import { Select } from '@/components/ui/Select'
import {
  generateQuiz,
  submitQuizAnswer,
  submitQuiz,
  type QuizQuestion,
  type QuizAnswerResult,
  type QuizResult,
  type QuizSource,
} from '../lib/api'
import { showError } from '../lib/toast'
import { usePracticeTopics } from '@/hooks/useTopics'
import { useInvalidateProgress } from '@/hooks/useInvalidateProgress'

interface QuizModeProps {
  courseId: string
  userId: string
  onModeChange?: (mode: 'chat' | 'quiz' | 'notes' | 'practice' | 'analytics') => void
}

interface QuizRun {
  quizId: string
  questions: QuizQuestion[]
  currentIndex: number
  selectedLetter: string
  feedback: QuizAnswerResult | null
  questionStart: number
  correctCount: number
}

type Difficulty = 'easy' | 'medium' | 'hard'

const LETTERS = ['A', 'B', 'C', 'D'] as const
// Sentinel for "quiz the entire course" — sends a null topic so the backend does
// broad whole-course retrieval (core concepts) instead of one narrow topic.
const WHOLE_COURSE = 'Whole course'

const DIFFICULTIES: { value: Difficulty; label: string; hint: string }[] = [
  { value: 'easy', label: 'Easy', hint: 'Warm-up' },
  { value: 'medium', label: 'Medium', hint: 'Balanced' },
  { value: 'hard', label: 'Hard', hint: 'Exam-grade' },
]
const COUNTS = [5, 10, 15, 20] as const

// ── Small presentational helpers ─────────────────────────────────────────────

/** Pill that cites a source document + page from the user's own materials. */
function SourceTag({ source, label }: { source: QuizSource; label?: string }) {
  if (!source?.doc_name) return null
  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-zinc-400 bg-zinc-800/80 border border-zinc-700 rounded-full px-2.5 py-1">
      <FileText className="w-3 h-3 text-cyan-400/80" />
      {label && <span className="text-zinc-500">{label}</span>}
      <span className="truncate max-w-[16rem]">{source.doc_name}</span>
      {source.page ? <span className="text-zinc-500">p.{source.page}</span> : null}
    </span>
  )
}

/** Pill that surfaces the concept a grounded question is testing. */
function ConceptTag({ concept }: { concept?: string }) {
  if (!concept) return null
  return (
    <span className="inline-flex items-center gap-1.5 text-xs font-medium text-cyan-300 bg-gradient-brand-soft border border-cyan-400/20 rounded-full px-2.5 py-1">
      <Sparkles className="w-3 h-3" />
      <span className="truncate max-w-[14rem]">{concept}</span>
    </span>
  )
}

export default function QuizMode({ courseId, userId, onModeChange }: QuizModeProps) {
  const [run, setRun] = useState<QuizRun | null>(null)
  const [result, setResult] = useState<QuizResult | null>(null)
  const [selectedTopic, setSelectedTopic] = useState(WHOLE_COURSE)
  const [difficulty, setDifficulty] = useState<Difficulty>('medium')
  const [questionCount, setQuestionCount] = useState(10)
  const [loading, setLoading] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [timeElapsed, setTimeElapsed] = useState(0)

  // "Whole course" is always available and the default — specific topics just
  // let the user narrow the focus. The shared hook attaches the auth token.
  const topicsQuery = usePracticeTopics(courseId)
  const invalidateProgress = useInvalidateProgress(courseId)

  const availableTopics = useMemo(
    () => [WHOLE_COURSE, ...(topicsQuery.data?.topics?.filter(Boolean) ?? [])],
    [topicsQuery.data],
  )
  // isFetching (not isPending) so the Refresh action also shows as loading.
  const topicsLoading = !!courseId && topicsQuery.isFetching
  const topicsError = topicsQuery.isError
    ? 'Could not load specific topics — you can still quiz the whole course.'
    : topicsQuery.data?.error ?? null

  // Keep the selected topic valid as the list loads/refreshes.
  useEffect(() => {
    if (!availableTopics.includes(selectedTopic)) setSelectedTopic(WHOLE_COURSE)
  }, [availableTopics, selectedTopic])

  // Whole-quiz timer (for the summary screen).
  useEffect(() => {
    let interval: number | undefined
    if (run && !result) {
      interval = window.setInterval(() => setTimeElapsed((prev) => prev + 1), 1000)
    }
    return () => {
      if (interval) window.clearInterval(interval)
    }
  }, [run, result])

  const startQuiz = async () => {
    if (!courseId) return
    setLoading(true)
    try {
      // Null topic => backend retrieves broadly across the whole course.
      const topicArg = selectedTopic === WHOLE_COURSE ? null : selectedTopic
      const quiz = await generateQuiz(courseId, topicArg, difficulty, questionCount)
      if (!quiz.questions.length) {
        showError('No questions could be generated. Try another topic.')
        return
      }
      setRun({
        quizId: quiz.quiz_id,
        questions: quiz.questions,
        currentIndex: 0,
        selectedLetter: '',
        feedback: null,
        questionStart: Date.now(),
        correctCount: 0,
      })
      setResult(null)
      setTimeElapsed(0)
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Failed to generate quiz'
      showError(msg)
    } finally {
      setLoading(false)
    }
  }

  const submitAnswer = async () => {
    if (!run || !run.selectedLetter || run.feedback) return
    setSubmitting(true)
    try {
      const question = run.questions[run.currentIndex]
      const timeTaken = (Date.now() - run.questionStart) / 1000
      const feedback = await submitQuizAnswer(
        run.quizId,
        question.id,
        run.selectedLetter,
        timeTaken,
        userId,
      )
      setRun({
        ...run,
        feedback,
        correctCount: run.correctCount + (feedback.is_correct ? 1 : 0),
      })
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Failed to submit answer'
      showError(msg)
    } finally {
      setSubmitting(false)
    }
  }

  const nextQuestion = async () => {
    if (!run) return
    if (run.currentIndex < run.questions.length - 1) {
      setRun({
        ...run,
        currentIndex: run.currentIndex + 1,
        selectedLetter: '',
        feedback: null,
        questionStart: Date.now(),
      })
    } else {
      await finishQuiz()
    }
  }

  const finishQuiz = async () => {
    if (!run) return
    setSubmitting(true)
    try {
      const finalResult = await submitQuiz(run.quizId, userId)
      setResult(finalResult)
      // Scoring the quiz changed mastery server-side — refresh progress views.
      invalidateProgress()
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Failed to score quiz'
      showError(msg)
    } finally {
      setSubmitting(false)
    }
  }

  const resetQuiz = () => {
    setRun(null)
    setResult(null)
    setTimeElapsed(0)
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  // ── Setup screen — center-first, focused, tactile ───────────────────────────
  if (!run) {
    const topicOptions = availableTopics.map((t) => ({
      value: t,
      label: t === WHOLE_COURSE ? 'Whole course' : t,
      hint: t === WHOLE_COURSE ? 'Broad — core concepts from everywhere' : undefined,
    }))

    return (
      <div className="flex min-h-full flex-col items-center justify-center px-4 py-10">
        <motion.div
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
          className="w-full max-w-xl"
        >
          {/* Identity */}
          <div className="mb-8 text-center">
            <BrandMark className="mx-auto mb-5 h-14 w-14 glow-brand-sm" />
            <h1 className="text-[28px] font-semibold tracking-tight text-zinc-50">
              Set up your quiz drill
            </h1>
            <p className="mx-auto mt-2 max-w-md text-sm text-zinc-400">
              Rapid multiple-choice, graded the instant you answer. Pick your focus and go.
            </p>
          </div>

          {/* Difficulty — 3 big selectable tiles */}
          <div className="mb-6">
            <div className="mb-2.5 flex items-center justify-center text-[11px] font-semibold uppercase tracking-[0.16em] text-zinc-500">
              Difficulty
            </div>
            <div className="grid grid-cols-3 gap-2.5">
              {DIFFICULTIES.map((d) => {
                const active = difficulty === d.value
                return (
                  <button
                    key={d.value}
                    onClick={() => setDifficulty(d.value)}
                    className={`group rounded-2xl border px-4 py-5 text-center transition-all ${
                      active
                        ? 'border-cyan-400/50 bg-gradient-brand-soft ring-2 ring-cyan-400/25 shadow-[0_8px_24px_-12px_rgba(34,211,238,0.5)]'
                        : 'border-white/10 bg-white/[0.03] hover:border-cyan-400/30 hover:bg-white/[0.05]'
                    }`}
                  >
                    <div className={`text-base font-semibold ${active ? 'text-cyan-200' : 'text-zinc-200'}`}>
                      {d.label}
                    </div>
                    <div className="mt-0.5 text-[11px] text-zinc-500">{d.hint}</div>
                  </button>
                )
              })}
            </div>
          </div>

          {/* Count — segmented control */}
          <div className="mb-6">
            <div className="mb-2.5 flex items-center justify-center text-[11px] font-semibold uppercase tracking-[0.16em] text-zinc-500">
              Questions
            </div>
            <div className="flex gap-1.5 rounded-2xl border border-white/10 bg-white/[0.03] p-1.5">
              {COUNTS.map((c) => {
                const active = questionCount === c
                return (
                  <button
                    key={c}
                    onClick={() => setQuestionCount(c)}
                    className={`flex-1 rounded-xl py-2.5 text-sm font-semibold transition-all ${
                      active
                        ? 'bg-gradient-brand text-white shadow-[0_6px_18px_-8px_rgba(34,211,238,0.5)]'
                        : 'text-zinc-400 hover:bg-white/[0.05] hover:text-zinc-200'
                    }`}
                  >
                    {c}
                  </button>
                )
              })}
            </div>
          </div>

          {/* Topic — Select primitive */}
          <div className="mb-8">
            <div className="mb-2.5 flex items-center justify-between">
              <span className="text-[11px] font-semibold uppercase tracking-[0.16em] text-zinc-500">Topic</span>
              <button
                onClick={() => void topicsQuery.refetch()}
                disabled={topicsLoading}
                className="inline-flex items-center gap-1 text-xs text-cyan-300 transition-colors hover:text-cyan-200 disabled:opacity-50"
                aria-label="Refresh topics"
              >
                <RefreshCw className={`h-3 w-3 ${topicsLoading ? 'animate-spin' : ''}`} />
                Refresh
              </button>
            </div>
            <Select
              value={selectedTopic}
              onChange={setSelectedTopic}
              options={topicOptions}
              disabled={topicsLoading || !courseId}
              ariaLabel="Quiz topic"
              placeholder={topicsLoading ? 'Loading topics…' : 'Select topic'}
            />
            {selectedTopic === WHOLE_COURSE && (
              <p className="mt-2 text-center text-xs text-zinc-500">
                Pulls core concepts broadly from across the entire course.
              </p>
            )}
            {topicsError && <p className="mt-2 text-center text-xs text-amber-500">{topicsError}</p>}
          </div>

          {/* One prominent CTA */}
          <Button
            onClick={() => void startQuiz()}
            disabled={loading || topicsLoading || !selectedTopic || !courseId}
            loading={loading}
            size="lg"
            leftIcon={<Zap className="h-4 w-4" />}
            className="w-full !py-3.5 !text-base"
          >
            {loading ? 'Generating your drill…' : 'Start drill'}
          </Button>
          {loading && (
            <p className="mt-3 text-center text-xs text-zinc-500">
              Retrieving from your materials, reranking, and writing questions — this can take a moment.
            </p>
          )}
        </motion.div>
      </div>
    )
  }

  // ── Results screen ───────────────────────────────────────────────────────────
  if (result) {
    const scorePct = result.score.pct
    const sortedTopics = [...result.by_topic].sort((a, b) => a.pct - b.pct)
    const headline =
      scorePct >= 85 ? 'Outstanding work' : scorePct >= 60 ? 'Solid effort' : 'Good start — keep going'
    const ringColor = scorePct >= 70 ? '#10b981' : scorePct >= 40 ? '#f59e0b' : '#ef4444'

    return (
      <div className="max-w-3xl mx-auto px-5 py-8">
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, ease: 'easeOut' }}
        >
          <Card accent>
            <div className="flex items-center gap-3 mb-6">
              <div className="w-11 h-11 rounded-xl bg-gradient-brand-soft border border-cyan-400/15 flex items-center justify-center flex-shrink-0 glow-brand-sm">
                <Trophy className="w-5 h-5 text-cyan-300" />
              </div>
              <div>
                <h2 className="text-xl font-semibold text-gradient-brand mb-0.5">{headline}</h2>
                <p className="text-sm text-zinc-500">
                  {selectedTopic} &middot; {difficulty}
                </p>
              </div>
            </div>

            {/* Big score ring + stats */}
            <div className="flex flex-col sm:flex-row items-center gap-6 mb-6">
              <motion.div
                initial={{ scale: 0.85, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ delay: 0.1, type: 'spring', stiffness: 180, damping: 16 }}
                className="relative flex-shrink-0"
              >
                <svg width="132" height="132" viewBox="0 0 132 132" className="-rotate-90">
                  <circle cx="66" cy="66" r="58" fill="none" stroke="#27272a" strokeWidth="10" />
                  <motion.circle
                    cx="66"
                    cy="66"
                    r="58"
                    fill="none"
                    stroke={ringColor}
                    strokeWidth="10"
                    strokeLinecap="round"
                    strokeDasharray={2 * Math.PI * 58}
                    initial={{ strokeDashoffset: 2 * Math.PI * 58 }}
                    animate={{ strokeDashoffset: 2 * Math.PI * 58 * (1 - scorePct / 100) }}
                    transition={{ delay: 0.25, duration: 0.9, ease: 'easeOut' }}
                  />
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                  <span className="text-3xl font-bold text-gradient-brand leading-none">{scorePct}%</span>
                  <span className="text-xs text-zinc-500 mt-1">Score</span>
                </div>
              </motion.div>

              <div className="grid grid-cols-2 gap-3 flex-1 w-full">
                <div className="bg-zinc-800/70 border border-zinc-700/50 rounded-lg p-4 text-center">
                  <div className="text-2xl font-bold mb-0.5 text-emerald-400">
                    {result.score.correct}/{result.score.total}
                  </div>
                  <div className="text-xs text-zinc-500">Correct</div>
                </div>
                <div className="bg-zinc-800/70 border border-zinc-700/50 rounded-lg p-4 text-center">
                  <div className="text-2xl font-bold mb-0.5 text-zinc-100 flex items-center justify-center gap-1.5">
                    <Clock className="w-4 h-4 text-cyan-300" />
                    {formatTime(timeElapsed)}
                  </div>
                  <div className="text-xs text-zinc-500">Time</div>
                </div>
              </div>
            </div>

            {/* By-topic breakdown */}
            {sortedTopics.length > 0 && (
              <div className="mb-6">
                <h3 className="text-sm font-medium text-zinc-300 mb-3">Breakdown by topic</h3>
                <div className="space-y-2.5">
                  {sortedTopics.map((t, i) => {
                    const barColor =
                      t.pct >= 70 ? '#10b981' : t.pct >= 40 ? '#f59e0b' : '#ef4444'
                    return (
                      <div key={t.topic}>
                        <div className="flex items-center justify-between mb-1 text-sm">
                          <span className="text-zinc-300 truncate pr-3">{t.topic}</span>
                          <span className="text-zinc-500 flex-shrink-0">
                            {t.correct}/{t.total} &middot; {t.pct}%
                          </span>
                        </div>
                        <div className="w-full bg-zinc-800 rounded-full h-2 overflow-hidden">
                          <motion.div
                            className="h-2 rounded-full"
                            style={{ background: barColor }}
                            initial={{ width: 0 }}
                            animate={{ width: `${t.pct}%` }}
                            transition={{ delay: 0.15 + i * 0.06, duration: 0.6, ease: 'easeOut' }}
                          />
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}

            {/* Weak areas */}
            {result.weak_areas.length > 0 && (
              <div className="mb-6 bg-amber-500/10 border border-amber-500/20 rounded-lg p-4">
                <div className="flex items-start gap-2.5">
                  <Target className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
                  <div>
                    <h4 className="text-sm font-medium text-amber-400 mb-1.5">Worth another look</h4>
                    <div className="flex flex-wrap gap-1.5">
                      {result.weak_areas.map((area) => (
                        <span
                          key={area}
                          className="text-xs text-amber-300 bg-amber-500/10 border border-amber-500/20 rounded-full px-2.5 py-1"
                        >
                          {area}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )}

            <div className="flex flex-wrap gap-3">
              <Button onClick={resetQuiz} leftIcon={<RotateCcw className="w-4 h-4" />}>
                New Quiz
              </Button>
              <Button
                variant="secondary"
                onClick={() =>
                  onModeChange
                    ? onModeChange('practice')
                    : window.dispatchEvent(new CustomEvent('navigateToPractice'))
                }
                leftIcon={<BookOpen className="w-4 h-4" />}
              >
                Practice Weak Areas
              </Button>
              <Button
                variant="secondary"
                onClick={() =>
                  onModeChange
                    ? onModeChange('analytics')
                    : window.dispatchEvent(new CustomEvent('navigateToAnalytics'))
                }
                leftIcon={<Brain className="w-4 h-4" />}
              >
                View Analytics
              </Button>
            </div>
          </Card>
        </motion.div>
      </div>
    )
  }

  // ── Active quiz ─────────────────────────────────────────────────────────────
  const question = run.questions[run.currentIndex]
  const feedback = run.feedback
  const progress = ((run.currentIndex + (feedback ? 1 : 0)) / run.questions.length) * 100
  const isLast = run.currentIndex === run.questions.length - 1

  return (
    <div className="mx-auto max-w-2xl px-5 py-7">
      {/* Slim progress strip — question N of M + meta + live counters */}
      <div className="mb-6">
        <div className="mb-2.5 flex items-center justify-between">
          <div className="flex items-baseline gap-2">
            <span className="text-sm font-semibold text-zinc-100">
              Question {run.currentIndex + 1}
              <span className="text-zinc-500"> of {run.questions.length}</span>
            </span>
            <span className="hidden text-xs text-zinc-500 sm:inline">
              · {selectedTopic} · {difficulty}
            </span>
          </div>
          <div className="flex items-center gap-3.5">
            <span className="inline-flex items-center gap-1.5 text-sm text-emerald-400/90">
              <CheckCircle className="h-4 w-4" />
              {run.correctCount}
            </span>
            <span className="inline-flex items-center gap-1.5 text-sm text-zinc-400">
              <Clock className="h-4 w-4 text-cyan-300" />
              {formatTime(timeElapsed)}
            </span>
          </div>
        </div>
        {/* Segmented progress */}
        <div className="flex gap-1">
          {run.questions.map((_, i) => {
            const done = i < run.currentIndex || (i === run.currentIndex && !!feedback)
            const current = i === run.currentIndex && !feedback
            return (
              <div
                key={i}
                className={`h-1.5 flex-1 rounded-full transition-colors duration-500 ${
                  done ? 'bg-gradient-brand' : current ? 'bg-cyan-400/40' : 'bg-white/[0.08]'
                }`}
              />
            )
          })}
        </div>
        <div className="sr-only">{Math.round(progress)}% complete</div>
      </div>

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
            {/* Concept + source tags */}
            <div className="flex flex-wrap items-center gap-2 mb-4">
              <ConceptTag concept={question.concept} />
              <SourceTag source={question.source} />
            </div>

            <div className="text-xl font-medium leading-snug text-zinc-50 mb-6">
              <Markdown content={question.question} />
            </div>

            <div className="space-y-2.5 mb-5">
              {question.options.map((option, index) => {
                const letter = LETTERS[index] ?? String.fromCharCode(65 + index)
                const isSelected = run.selectedLetter === letter
                const isCorrectOption = feedback ? letter === feedback.correct_answer : false

                let klass =
                  'w-full p-4 border rounded-xl text-left transition-all text-[15px] disabled:cursor-default '
                if (feedback) {
                  if (isCorrectOption)
                    klass += 'border-emerald-500/70 bg-emerald-500/10 text-emerald-300'
                  else if (isSelected) klass += 'border-red-500/70 bg-red-500/10 text-red-300'
                  else klass += 'border-zinc-700/60 bg-zinc-800/40 text-zinc-500'
                } else if (isSelected) {
                  klass +=
                    'border-transparent bg-gradient-brand-soft text-cyan-200 ring-2 ring-cyan-400/40'
                } else {
                  klass +=
                    'border-zinc-700 text-zinc-200 hover:border-cyan-400/40 hover:bg-cyan-400/5'
                }

                return (
                  <motion.button
                    key={index}
                    whileTap={!feedback ? { scale: 0.99 } : undefined}
                    onClick={() => !feedback && setRun({ ...run, selectedLetter: letter })}
                    disabled={!!feedback}
                    className={klass}
                  >
                    <div className="flex items-center gap-3">
                      <div
                        className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0 ${
                          feedback && isCorrectOption
                            ? 'bg-emerald-500 text-white'
                            : feedback && isSelected && !isCorrectOption
                              ? 'bg-red-500 text-white'
                              : isSelected
                                ? 'bg-gradient-brand text-white'
                                : 'bg-zinc-700 text-zinc-400'
                        }`}
                      >
                        {feedback && isCorrectOption ? (
                          <CheckCircle className="w-3.5 h-3.5" />
                        ) : feedback && isSelected && !isCorrectOption ? (
                          <XCircle className="w-3.5 h-3.5" />
                        ) : (
                          letter
                        )}
                      </div>
                      {/* Options already carry their "A) " prefix from the backend. */}
                      <span className="flex-1">{option.replace(/^[A-D]\)\s*/, '')}</span>
                    </div>
                  </motion.button>
                )
              })}
            </div>

            <AnimatePresence>
              {feedback && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  transition={{ duration: 0.3, ease: 'easeOut' }}
                  className="overflow-hidden"
                >
                  <div
                    className={`rounded-xl p-4 mb-5 ${
                      feedback.is_correct
                        ? 'bg-emerald-500/10 border border-emerald-500/25'
                        : 'bg-red-500/10 border border-red-500/25'
                    }`}
                  >
                    <div className="flex items-start gap-2.5">
                      <motion.div
                        initial={{ scale: 0.6, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        transition={{ type: 'spring', stiffness: 220, damping: 14 }}
                        className="flex-shrink-0 mt-0.5"
                      >
                        {feedback.is_correct ? (
                          <CheckCircle className="w-5 h-5 text-emerald-400" />
                        ) : (
                          <XCircle className="w-5 h-5 text-red-400" />
                        )}
                      </motion.div>
                      <div className="min-w-0 flex-1">
                        <h4
                          className={`text-sm font-semibold mb-1 ${
                            feedback.is_correct ? 'text-emerald-400' : 'text-red-400'
                          }`}
                        >
                          {feedback.is_correct ? 'Correct!' : 'Not quite right'}
                        </h4>
                        <div
                          className={`text-sm ${
                            feedback.is_correct ? 'text-emerald-300/90' : 'text-red-300/90'
                          }`}
                        >
                          <Markdown content={feedback.explanation} />
                        </div>

                        {/* Cited mistake explanation — grounded in the user's own pages. */}
                        {!feedback.is_correct && feedback.mistake_explanation && (
                          <div className="mt-3 rounded-lg bg-zinc-900/70 border border-zinc-700 p-3">
                            <div className="text-xs font-semibold text-amber-400 mb-1 flex items-center gap-1.5">
                              <Target className="w-3.5 h-3.5" />
                              Why you missed this
                            </div>
                            <div className="text-sm text-zinc-300">
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
                      </div>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            <div>
              {!feedback ? (
                <Button
                  onClick={() => void submitAnswer()}
                  disabled={!run.selectedLetter || submitting}
                  loading={submitting}
                  size="lg"
                  leftIcon={<CheckCircle className="w-4 h-4" />}
                  className="w-full"
                >
                  {submitting ? 'Checking…' : 'Submit Answer'}
                </Button>
              ) : (
                <Button
                  onClick={() => void nextQuestion()}
                  disabled={submitting}
                  loading={submitting}
                  size="lg"
                  rightIcon={
                    isLast ? <Trophy className="w-4 h-4" /> : <ArrowRight className="w-4 h-4" />
                  }
                  className="w-full"
                >
                  {submitting ? 'Scoring…' : isLast ? 'See Your Results' : 'Next Question'}
                </Button>
              )}
            </div>
          </Card>
        </motion.div>
      </AnimatePresence>
    </div>
  )
}
