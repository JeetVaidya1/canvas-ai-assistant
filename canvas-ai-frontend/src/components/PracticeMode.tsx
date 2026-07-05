import { useState, useEffect, useMemo } from 'react'
import { motion, AnimatePresence } from 'motion/react'
import { Markdown } from '@/components/ui/Markdown'
import { Button } from '@/components/ui/Button'
import { Card, PageHeader } from '@/components/ui/Card'
import { Select } from '@/components/ui/Select'
import {
  Play,
  CheckCircle,
  XCircle,
  Clock,
  Target,
  Trophy,
  RotateCcw,
  Brain,
  ArrowRight,
  BookOpen,
  RefreshCw,
  Library,
  Globe,
  Gauge,
  Eye
} from 'lucide-react'
import { BrandMark } from '@/components/ui/BrandMark'

import {
  generatePracticeProblems,
  trackPracticeSession as apiTrackPracticeSession,
  type PracticeProblem
} from '../lib/api'
import { showError } from '../lib/toast'
import { usePracticeTopics } from '@/hooks/useTopics'
import { useInvalidateProgress } from '@/hooks/useInvalidateProgress'

interface PracticeSession {
  problems: PracticeProblem[]
  currentProblemIndex: number
  userAnswers: string[]
  startTime: Date
  isComplete: boolean
  score: number
}

interface PracticeModeProps {
  courseId: string
  userId: string
  onModeChange?: (mode: 'chat' | 'quiz' | 'notes' | 'practice' | 'analytics') => void
}

type DifficultyLevel = 'adaptive' | 'easy' | 'medium' | 'hard'

/** Backend may attach a grounding hint; read it defensively without altering the data layer. */
type ProblemSource = 'materials' | 'general'

const DIFFICULTY_STYLES: Record<string, { label: string; cls: string }> = {
  easy: { label: 'Easy', cls: 'text-emerald-300 bg-emerald-500/10 border-emerald-500/25' },
  medium: { label: 'Medium', cls: 'text-amber-300 bg-amber-500/10 border-amber-500/25' },
  hard: { label: 'Hard', cls: 'text-rose-300 bg-rose-500/10 border-rose-500/25' },
}

function resolveDifficulty(raw: string): { label: string; cls: string } {
  const key = (raw || '').trim().toLowerCase()
  return DIFFICULTY_STYLES[key] ?? { label: raw || 'Adaptive', cls: 'text-cyan-300 bg-gradient-brand-soft border-cyan-400/25' }
}

/** Infer whether a problem is grounded in course materials vs. general knowledge. */
function resolveSource(problem: PracticeProblem): ProblemSource {
  const probe = problem as PracticeProblem & {
    source?: string
    grounded?: boolean
    from_materials?: boolean
  }
  if (typeof probe.grounded === 'boolean') return probe.grounded ? 'materials' : 'general'
  if (typeof probe.from_materials === 'boolean') return probe.from_materials ? 'materials' : 'general'
  const s = (probe.source || '').toLowerCase()
  if (s.includes('general')) return 'general'
  if (s.includes('material') || s.includes('course') || s.includes('document') || s.includes('retriev')) return 'materials'
  // Default: the practice backend grounds in retrieved course content.
  return 'materials'
}

function formatEstimatedTime(raw: string): string {
  const trimmed = (raw || '').trim()
  if (!trimmed) return '~1 min'
  if (/\d/.test(trimmed) && !/min|sec/i.test(trimmed)) return `~${trimmed} min`
  return trimmed.startsWith('~') ? trimmed : `~${trimmed}`
}

export default function PracticeMode({ courseId, userId, onModeChange }: PracticeModeProps) {
  const [session, setSession] = useState<PracticeSession | null>(null)
  const [selectedTopic, setSelectedTopic] = useState('')
  const [difficulty, setDifficulty] = useState<DifficultyLevel>('adaptive')
  const [problemCount, setProblemCount] = useState(5)
  const [loading, setLoading] = useState(false)
  const [selectedAnswer, setSelectedAnswer] = useState('')
  const [showExplanation, setShowExplanation] = useState(false)
  const [timeElapsed, setTimeElapsed] = useState(0)

  const topicsQuery = usePracticeTopics(courseId)
  const invalidateProgress = useInvalidateProgress(courseId)

  const availableTopics = useMemo(() => {
    if (!courseId) return ['General Topics']
    const topics = topicsQuery.data?.topics
    return topics && topics.length > 0 ? topics : ['Course Content', 'General Review']
  }, [courseId, topicsQuery.data])

  // isFetching (not isPending) so the Refresh action also shows as loading.
  const topicsLoading = !!courseId && topicsQuery.isFetching
  const topicsError = !courseId
    ? 'Please select a course first'
    : topicsQuery.isError
      ? 'Failed to load topics. Please try again.'
      : topicsQuery.data?.error ?? null

  // Keep the selected topic valid as the list loads/refreshes.
  useEffect(() => {
    setSelectedTopic((prev) =>
      prev && availableTopics.includes(prev) ? prev : availableTopics[0] ?? '',
    )
  }, [availableTopics])

  useEffect(() => {
    let interval: number | undefined
    if (session && !session.isComplete) {
      interval = window.setInterval(() => setTimeElapsed(prev => prev + 1), 1000)
    }
    return () => {
      if (interval) window.clearInterval(interval)
    }
  }, [session])

  const startPracticeSession = async () => {
    if (!courseId || !selectedTopic) return
    setLoading(true)
    try {
      const problems = await generatePracticeProblems(courseId, selectedTopic, difficulty, problemCount, userId)
      setSession({
        problems,
        currentProblemIndex: 0,
        userAnswers: new Array(problems.length).fill(''),
        startTime: new Date(),
        isComplete: false,
        score: 0
      })
      setTimeElapsed(0)
      setSelectedAnswer('')
      setShowExplanation(false)
    } catch (e) {
      console.error('Failed to generate practice problems:', e)
      showError('Failed to generate practice problems. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const submitAnswer = () => {
    if (!session || !selectedAnswer) return
    const userAnswers = [...session.userAnswers]
    userAnswers[session.currentProblemIndex] = selectedAnswer
    setSession({ ...session, userAnswers })
    setShowExplanation(true)
  }

  const nextProblem = () => {
    if (!session) return
    if (session.currentProblemIndex < session.problems.length - 1) {
      setSession({ ...session, currentProblemIndex: session.currentProblemIndex + 1 })
      setSelectedAnswer('')
      setShowExplanation(false)
    } else {
      completeSession()
    }
  }

  const completeSession = () => {
    if (!session) return
    const correct = session.userAnswers.filter((a, i) => a === session.problems[i].correct_answer).length
    const score = Math.round((correct / session.problems.length) * 100)
    setSession({ ...session, isComplete: true, score })
    void trackPractice(correct, session.problems.length)
  }

  const trackPractice = async (correct: number, total: number) => {
    try {
      await apiTrackPracticeSession(
        userId,
        courseId,
        selectedTopic,
        total,
        correct,
        Math.max(1, Math.round(timeElapsed / 60)),
        difficulty
      )
      // The session changed mastery server-side — refresh progress views.
      invalidateProgress()
    } catch (e) {
      console.warn('Practice tracking failed (non-blocking):', e)
    }
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const getCurrentProblem = () => session?.problems[session.currentProblemIndex]
  const getAnswerLabel = (i: number) => String.fromCharCode(65 + i)

  // ===== Setup screen — center-first, focused, tactile =====
  if (!session) {
    const topicOptions = availableTopics.map((t) => ({ value: t, label: t }))
    const DIFF_TILES: { value: DifficultyLevel; label: string; hint: string }[] = [
      { value: 'adaptive', label: 'Adaptive', hint: 'Matches your mastery' },
      { value: 'easy', label: 'Easy', hint: 'Warm-up' },
      { value: 'medium', label: 'Medium', hint: 'Balanced' },
      { value: 'hard', label: 'Hard', hint: 'Push yourself' },
    ]
    const PROBLEM_COUNTS = [3, 5, 10, 15] as const

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
            <BrandMark className="mx-auto mb-5 h-14 w-14 glow-brand" />
            <h1 className="text-[28px] font-semibold tracking-tight text-zinc-50">
              Set up your problem set
            </h1>
            <p className="mx-auto mt-2 max-w-md text-sm text-zinc-400">
              Open-ended problems generated fresh from your course. Difficulty adapts to your mastery, problem by problem.
            </p>
          </div>

          {/* Difficulty — 4 tactile tiles */}
          <div className="mb-6">
            <div className="mb-2.5 text-center text-[11px] font-semibold uppercase tracking-[0.16em] text-zinc-500">
              Difficulty
            </div>
            <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-4">
              {DIFF_TILES.map((d) => {
                const active = difficulty === d.value
                return (
                  <button
                    key={d.value}
                    onClick={() => setDifficulty(d.value)}
                    className={`rounded-2xl border px-3 py-4 text-center transition-all ${
                      active
                        ? 'border-cyan-400/50 bg-gradient-brand-soft ring-2 ring-cyan-400/25 shadow-[0_8px_24px_-12px_rgba(34,211,238,0.5)]'
                        : 'border-white/10 bg-white/[0.03] hover:border-cyan-400/30 hover:bg-white/[0.05]'
                    }`}
                  >
                    <div className={`text-sm font-semibold ${active ? 'text-cyan-200' : 'text-zinc-200'}`}>
                      {d.label}
                    </div>
                    <div className="mt-0.5 text-[11px] text-zinc-500">{d.hint}</div>
                  </button>
                )
              })}
            </div>
          </div>

          {/* Problem count — segmented control */}
          <div className="mb-6">
            <div className="mb-2.5 text-center text-[11px] font-semibold uppercase tracking-[0.16em] text-zinc-500">
              Problems
            </div>
            <div className="flex gap-1.5 rounded-2xl border border-white/10 bg-white/[0.03] p-1.5">
              {PROBLEM_COUNTS.map((c) => {
                const active = problemCount === c
                return (
                  <button
                    key={c}
                    onClick={() => setProblemCount(c)}
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
          <div className="mb-6">
            <div className="mb-2.5 flex items-center justify-between">
              <span className="text-[11px] font-semibold uppercase tracking-[0.16em] text-zinc-500">Topic</span>
              <button
                onClick={() => void topicsQuery.refetch()}
                disabled={topicsLoading}
                className="inline-flex items-center gap-1 text-xs text-cyan-300 transition-colors hover:text-cyan-200 disabled:opacity-50"
                aria-label="Reload topics"
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
              ariaLabel="Practice topic"
              placeholder={topicsLoading ? 'Loading…' : 'Select topic'}
            />
            {topicsError && <p className="mt-2 text-center text-xs text-rose-400">{topicsError}</p>}
          </div>

          {difficulty === 'adaptive' && (
            <div className="mb-8 flex items-start gap-2.5 rounded-xl border border-cyan-400/15 bg-gradient-brand-soft px-3.5 py-3">
              <Gauge className="mt-0.5 h-4 w-4 flex-shrink-0 text-cyan-300" />
              <p className="text-xs text-cyan-200/80">
                Adaptive mode reads your recent mastery and calibrates each problem's difficulty — you'll see the resolved level on every card.
              </p>
            </div>
          )}
          {difficulty !== 'adaptive' && <div className="mb-8" />}

          {/* One prominent CTA */}
          <Button
            onClick={startPracticeSession}
            disabled={loading || topicsLoading || !selectedTopic || !courseId}
            loading={loading}
            size="lg"
            leftIcon={<Play className="h-4 w-4" />}
            className="w-full !py-3.5 !text-base"
          >
            {loading ? 'Generating…' : 'Start session'}
          </Button>
        </motion.div>
      </div>
    )
  }

  // ===== Results screen =====
  if (session.isComplete) {
    const correctCount = session.userAnswers.filter((a, i) => a === session.problems[i].correct_answer).length
    return (
      <div className="max-w-3xl mx-auto px-5 py-5 space-y-6">
        <PageHeader
          eyebrow="Session complete"
          title="Practice Complete"
          subtitle="Here's how you did"
        />

        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }}>
          <Card accent padding="lg" className="space-y-5">
            <div className="grid grid-cols-3 gap-4">
              <div className="bg-gradient-brand-soft border border-cyan-400/15 rounded-xl p-4 text-center">
                <div className="text-2xl font-bold mb-0.5 text-gradient-brand">{session.score}%</div>
                <div className="text-xs text-zinc-500">Score</div>
              </div>
              <div className="bg-zinc-800/70 border border-zinc-700/50 rounded-xl p-4 text-center">
                <div className="text-2xl font-bold mb-0.5 text-emerald-400">{correctCount}/{session.problems.length}</div>
                <div className="text-xs text-zinc-500">Correct</div>
              </div>
              <div className="bg-zinc-800/70 border border-zinc-700/50 rounded-xl p-4 text-center">
                <div className="text-2xl font-bold mb-0.5 text-zinc-100">{formatTime(timeElapsed)}</div>
                <div className="text-xs text-zinc-500">Time</div>
              </div>
            </div>

            {/* Performance summary */}
            {session.score >= 80 ? (
              <div className="bg-emerald-500/10 border border-emerald-500/20 rounded-lg p-3 flex items-center gap-3">
                <CheckCircle className="w-5 h-5 text-emerald-400 flex-shrink-0" />
                <p className="text-sm text-emerald-400">Strong mastery of {selectedTopic}. Try harder difficulty or new topics.</p>
              </div>
            ) : session.score >= 60 ? (
              <div className="bg-amber-500/10 border border-amber-500/20 rounded-lg p-3 flex items-center gap-3">
                <Target className="w-5 h-5 text-amber-400 flex-shrink-0" />
                <p className="text-sm text-amber-400">Good progress on {selectedTopic}. A bit more practice will help.</p>
              </div>
            ) : (
              <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-3 flex items-center gap-3">
                <BookOpen className="w-5 h-5 text-red-400 flex-shrink-0" />
                <p className="text-sm text-red-400">Review {selectedTopic} and try easier problems first.</p>
              </div>
            )}

            <div className="flex gap-3">
              <Button
                onClick={() => setSession(null)}
                leftIcon={<RotateCcw className="w-4 h-4" />}
              >
                Practice Again
              </Button>
              <Button
                variant="secondary"
                onClick={() => onModeChange ? onModeChange('analytics') : window.dispatchEvent(new CustomEvent('navigateToAnalytics'))}
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

  // ===== Active session =====
  const currentProblem = getCurrentProblem()
  if (!currentProblem) return null

  const isCorrect = selectedAnswer === currentProblem.correct_answer
  const progress = ((session.currentProblemIndex + 1) / session.problems.length) * 100
  const diff = resolveDifficulty(currentProblem.difficulty)
  const source = resolveSource(currentProblem)
  const estTime = formatEstimatedTime(currentProblem.estimated_time)

  return (
    <div className="mx-auto max-w-2xl px-5 py-7">
      {/* Slim progress strip — problem N of M + meta + timer */}
      <div className="mb-6">
        <div className="mb-2.5 flex items-center justify-between">
          <div className="flex items-baseline gap-2">
            <span className="text-sm font-semibold text-zinc-100">
              Problem {session.currentProblemIndex + 1}
              <span className="text-zinc-500"> of {session.problems.length}</span>
            </span>
            <span className="hidden text-xs capitalize text-zinc-500 sm:inline">
              · {selectedTopic} · {difficulty}
            </span>
          </div>
          <span className="inline-flex items-center gap-1.5 text-sm text-zinc-400">
            <Clock className="h-4 w-4 text-cyan-300" />
            {formatTime(timeElapsed)}
          </span>
        </div>
        <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/[0.08]">
          <motion.div
            className="h-1.5 rounded-full bg-gradient-brand"
            initial={false}
            animate={{ width: `${progress}%` }}
            transition={{ duration: 0.5, ease: 'easeOut' }}
          />
        </div>
      </div>

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
            {/* Metadata badges: difficulty + source + estimated time */}
            <div className="flex flex-wrap items-center gap-2 mb-4">
              <span className={`inline-flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide px-2.5 py-1 rounded-full border ${diff.cls}`}>
                <Gauge className="w-3 h-3" />
                {diff.label}
              </span>
              {source === 'materials' ? (
                <span className="inline-flex items-center gap-1.5 text-[11px] font-medium px-2.5 py-1 rounded-full border border-cyan-400/25 bg-gradient-brand-soft text-cyan-300">
                  <Library className="w-3 h-3" />
                  From your materials
                </span>
              ) : (
                <span className="inline-flex items-center gap-1.5 text-[11px] font-medium px-2.5 py-1 rounded-full border border-zinc-700 bg-zinc-800/70 text-zinc-400">
                  <Globe className="w-3 h-3" />
                  General knowledge
                </span>
              )}
              <span className="inline-flex items-center gap-1.5 text-[11px] font-medium px-2.5 py-1 rounded-full border border-zinc-700 bg-zinc-800/70 text-zinc-400">
                <Clock className="w-3 h-3" />
                {estTime}
              </span>
            </div>

            <div className="mb-6">
              <span className="mb-2 block text-xs font-semibold uppercase tracking-widest text-gradient-brand">
                Question {session.currentProblemIndex + 1}
              </span>
              <div className="text-xl font-medium leading-snug text-zinc-50">
                <Markdown content={currentProblem.question} />
              </div>
            </div>

            {/* Options */}
            <div className="space-y-2.5 mb-5">
              {currentProblem.options.map((option, index) => {
                const letter = getAnswerLabel(index)
                const isSelected = selectedAnswer === letter
                const isCorrectOption = letter === currentProblem.correct_answer

                let klass = 'w-full p-4 border rounded-xl text-left transition-all text-[15px] '
                if (showExplanation) {
                  if (isCorrectOption) klass += 'border-emerald-500/70 bg-emerald-500/10 text-emerald-300'
                  else if (isSelected) klass += 'border-rose-500/70 bg-rose-500/10 text-rose-300'
                  else klass += 'border-zinc-700/60 bg-zinc-800/40 text-zinc-500'
                } else if (isSelected) {
                  klass += 'border-transparent bg-gradient-brand-soft text-cyan-200 ring-2 ring-cyan-400/40'
                } else {
                  klass += 'border-zinc-700 text-zinc-200 hover:border-cyan-400/40 hover:bg-cyan-400/5'
                }

                return (
                  <motion.button
                    key={index}
                    onClick={() => !showExplanation && setSelectedAnswer(letter)}
                    disabled={showExplanation}
                    className={klass}
                    whileTap={!showExplanation ? { scale: 0.99 } : undefined}
                  >
                    <div className="flex items-center gap-3">
                      <div className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0 ${
                        showExplanation && isCorrectOption
                          ? 'bg-emerald-500 text-white'
                          : showExplanation && isSelected && !isCorrectOption
                          ? 'bg-rose-500 text-white'
                          : isSelected
                          ? 'bg-gradient-brand text-white'
                          : 'bg-zinc-700 text-zinc-400'
                      }`}>
                        {showExplanation && isCorrectOption ? (
                          <CheckCircle className="w-3.5 h-3.5" />
                        ) : showExplanation && isSelected && !isCorrectOption ? (
                          <XCircle className="w-3.5 h-3.5" />
                        ) : (
                          letter
                        )}
                      </div>
                      <span className="flex-1">{option}</span>
                    </div>
                  </motion.button>
                )
              })}
            </div>

            {/* Solution reveal */}
            <AnimatePresence>
              {showExplanation && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  transition={{ duration: 0.3, ease: 'easeOut' }}
                  className="overflow-hidden"
                >
                  <div className={`rounded-xl p-4 mb-5 ${isCorrect ? 'bg-emerald-500/10 border border-emerald-500/25' : 'bg-rose-500/10 border border-rose-500/25'}`}>
                    <div className="flex items-start gap-2.5">
                      <motion.div
                        initial={{ scale: 0.6, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        transition={{ delay: 0.1, type: 'spring', stiffness: 300, damping: 18 }}
                      >
                        {isCorrect ? (
                          <CheckCircle className="w-5 h-5 text-emerald-400 flex-shrink-0 mt-0.5" />
                        ) : (
                          <XCircle className="w-5 h-5 text-rose-400 flex-shrink-0 mt-0.5" />
                        )}
                      </motion.div>
                      <div>
                        <h4 className={`text-sm font-semibold mb-1 flex items-center gap-1.5 ${isCorrect ? 'text-emerald-400' : 'text-rose-400'}`}>
                          <Eye className="w-3.5 h-3.5" />
                          {isCorrect ? 'Correct!' : 'Not quite right'}
                        </h4>
                        <div className={`text-sm ${isCorrect ? 'text-emerald-300/90' : 'text-rose-300/90'}`}>
                          <Markdown content={currentProblem.explanation} />
                        </div>
                      </div>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Actions */}
            <div>
              {!showExplanation ? (
                <Button
                  onClick={submitAnswer}
                  disabled={!selectedAnswer}
                  size="lg"
                  leftIcon={<CheckCircle className="w-4 h-4" />}
                  className="w-full"
                >
                  Submit Answer
                </Button>
              ) : session.currentProblemIndex < session.problems.length - 1 ? (
                <Button onClick={nextProblem} size="lg" rightIcon={<ArrowRight className="w-4 h-4" />} className="w-full">
                  Next Question
                </Button>
              ) : (
                <Button onClick={nextProblem} size="lg" rightIcon={<Trophy className="w-4 h-4" />} className="w-full">
                  View Results
                </Button>
              )}
            </div>
          </Card>
        </motion.div>
      </AnimatePresence>
    </div>
  )
}
