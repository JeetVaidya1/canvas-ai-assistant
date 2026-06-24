import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'motion/react'
import { Markdown } from '@/components/ui/Markdown'
import { Button } from '@/components/ui/Button'
import { Card, PageHeader } from '@/components/ui/Card'
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
  ChevronDown,
  RefreshCw,
  Sparkles,
  Library,
  Globe,
  Gauge,
  Eye
} from 'lucide-react'

import {
  generatePracticeProblems,
  trackPracticeSession as apiTrackPracticeSession,
  type PracticeProblem
} from '../lib/api'
import { showError } from '../lib/toast'

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

interface TopicsResponse {
  topics?: string[]
  error?: string
  status?: string
  course_files_count?: number
  extraction_method?: string
  fallback?: boolean
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
  return DIFFICULTY_STYLES[key] ?? { label: raw || 'Adaptive', cls: 'text-cyan-300 bg-gradient-brand-soft border-cyan-500/25' }
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
  const [availableTopics, setAvailableTopics] = useState<string[]>([])
  const [topicsLoading, setTopicsLoading] = useState(false)
  const [topicsError, setTopicsError] = useState<string | null>(null)
  const [topicDropdownOpen, setTopicDropdownOpen] = useState(false)

  useEffect(() => {
    let interval: number | undefined
    if (session && !session.isComplete) {
      interval = window.setInterval(() => setTimeElapsed(prev => prev + 1), 1000)
    }
    return () => {
      if (interval) window.clearInterval(interval)
    }
  }, [session])

  useEffect(() => {
    loadTopics()
  }, [courseId])

  const getPracticeTopics = async (cId: string): Promise<TopicsResponse> => {
    try {
      const response = await fetch(`${import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'}/practice-topics/${cId}`)
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }
      const data = await response.json()
      return data as TopicsResponse
    } catch (error) {
      console.error('Failed to fetch practice topics:', error)
      throw error
    }
  }

  const loadTopics = async () => {
    if (!courseId) {
      setAvailableTopics(['General Topics'])
      setSelectedTopic('General Topics')
      setTopicsError('Please select a course first')
      return
    }

    setTopicsLoading(true)
    setTopicsError(null)

    try {
      const response: TopicsResponse = await getPracticeTopics(courseId)

      if (response.topics && Array.isArray(response.topics)) {
        setAvailableTopics(response.topics)
        setSelectedTopic(response.topics[0] || 'Course Content')
        if (response.error) setTopicsError(response.error)
      } else {
        setAvailableTopics(['Course Content', 'General Review'])
        setSelectedTopic('Course Content')
        setTopicsError(response.error || 'No topics found in response')
      }
    } catch {
      setAvailableTopics(['Course Content', 'General Review'])
      setSelectedTopic('Course Content')
      setTopicsError('Failed to load topics. Please try again.')
    } finally {
      setTopicsLoading(false)
    }
  }

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

  const selectClass =
    'w-full px-3 py-2.5 bg-zinc-800/70 border border-zinc-700 rounded-lg text-zinc-100 ' +
    'focus:border-cyan-500/60 focus:ring-2 focus:ring-cyan-500/20 outline-none text-sm transition-colors'

  // ===== Setup screen =====
  if (!session) {
    return (
      <div className="max-w-3xl mx-auto px-5 py-5 space-y-6">
        <PageHeader
          eyebrow="Practice"
          title="Start a practice session"
          subtitle="Pick a topic and difficulty — Vindexa generates fresh problems from your course."
        />

        <Card accent padding="lg" className="space-y-5">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-brand-soft border border-cyan-500/15 flex items-center justify-center flex-shrink-0">
              <Sparkles className="w-5 h-5 text-cyan-300" />
            </div>
            <div>
              <div className="text-sm font-medium text-zinc-100">Adaptive practice</div>
              <div className="text-xs text-zinc-500">Difficulty adjusts to your mastery, problem by problem.</div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Topic */}
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label className="text-xs font-medium text-zinc-400">Topic</label>
                <button
                  onClick={() => loadTopics()}
                  disabled={topicsLoading}
                  className="text-cyan-300 hover:text-cyan-200 text-xs flex items-center gap-1 transition-colors"
                  aria-label="Reload topics"
                >
                  <RefreshCw className={`w-3 h-3 ${topicsLoading ? 'animate-spin' : ''}`} />
                </button>
              </div>
              <div className="relative">
                <button
                  onClick={() => setTopicDropdownOpen(!topicDropdownOpen)}
                  disabled={topicsLoading || !courseId}
                  className="w-full px-3 py-2.5 border border-zinc-700 rounded-lg bg-zinc-800/70 text-zinc-100 text-sm text-left flex items-center justify-between disabled:opacity-50 focus:border-cyan-500/60 focus:ring-2 focus:ring-cyan-500/20 outline-none transition-colors"
                >
                  <span className="truncate">
                    {topicsLoading ? 'Loading...' : selectedTopic || 'Select topic'}
                  </span>
                  <ChevronDown className={`w-3.5 h-3.5 text-zinc-500 transition-transform ${topicDropdownOpen ? 'rotate-180' : ''}`} />
                </button>
                <AnimatePresence>
                  {topicDropdownOpen && !topicsLoading && (
                    <motion.div
                      initial={{ opacity: 0, y: -4 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -4 }}
                      transition={{ duration: 0.15 }}
                      className="absolute z-10 w-full mt-1 bg-zinc-800 border border-zinc-700 rounded-lg shadow-lg max-h-48 overflow-y-auto"
                    >
                      {availableTopics.map((topic, index) => (
                        <button
                          key={topic}
                          onClick={() => {
                            setSelectedTopic(topic)
                            setTopicDropdownOpen(false)
                          }}
                          className={`w-full px-3 py-2 text-left text-sm transition-colors ${
                            selectedTopic === topic
                              ? 'bg-gradient-brand-soft text-cyan-300 font-medium'
                              : 'text-zinc-400 hover:bg-zinc-700'
                          } ${index === 0 ? 'rounded-t-lg' : ''} ${index === availableTopics.length - 1 ? 'rounded-b-lg' : ''}`}
                        >
                          <div className="truncate">{topic}</div>
                        </button>
                      ))}
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
              {topicsError && <p className="text-xs text-red-400 mt-1">{topicsError}</p>}
            </div>

            {/* Difficulty */}
            <div>
              <label className="block text-xs font-medium text-zinc-400 mb-1.5">Difficulty</label>
              <select
                value={difficulty}
                onChange={(e) => setDifficulty(e.target.value as DifficultyLevel)}
                className={selectClass}
              >
                <option value="adaptive">Adaptive</option>
                <option value="easy">Easy</option>
                <option value="medium">Medium</option>
                <option value="hard">Hard</option>
              </select>
            </div>

            {/* Problem count */}
            <div>
              <label className="block text-xs font-medium text-zinc-400 mb-1.5">Problems</label>
              <select
                value={problemCount}
                onChange={(e) => setProblemCount(Number(e.target.value))}
                className={selectClass}
              >
                <option value={3}>3</option>
                <option value={5}>5</option>
                <option value={10}>10</option>
                <option value={15}>15</option>
              </select>
            </div>
          </div>

          {difficulty === 'adaptive' && (
            <div className="flex items-start gap-2.5 rounded-lg border border-cyan-500/15 bg-gradient-brand-soft px-3 py-2.5">
              <Gauge className="w-4 h-4 text-cyan-300 flex-shrink-0 mt-0.5" />
              <p className="text-xs text-cyan-200/80">
                Adaptive mode reads your recent mastery and calibrates each problem's difficulty — you'll see the resolved level on every card.
              </p>
            </div>
          )}

          <Button
            onClick={startPracticeSession}
            disabled={loading || topicsLoading || !selectedTopic || !courseId}
            loading={loading}
            leftIcon={<Play className="w-4 h-4" />}
          >
            {loading ? 'Generating...' : 'Start Session'}
          </Button>
        </Card>
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
              <div className="bg-gradient-brand-soft border border-cyan-500/15 rounded-xl p-4 text-center">
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
    <div className="max-w-3xl mx-auto p-5 space-y-5">
      {/* Header */}
      <Card padding="md">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-brand-soft border border-cyan-500/15 flex items-center justify-center flex-shrink-0">
              <Target className="w-4 h-4 text-cyan-300" />
            </div>
            <div>
              <h2 className="text-base font-medium text-zinc-100">Practice Session</h2>
              <p className="text-xs text-zinc-500 capitalize">{selectedTopic} &middot; {difficulty}</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-1.5 text-sm text-zinc-400">
              <Clock className="w-4 h-4" />
              {formatTime(timeElapsed)}
            </div>
            <div className="text-sm text-zinc-400">
              {session.currentProblemIndex + 1}/{session.problems.length}
            </div>
          </div>
        </div>
        <div className="w-full bg-zinc-800 rounded-full h-1.5 overflow-hidden">
          <motion.div
            className="bg-gradient-brand h-1.5 rounded-full"
            initial={false}
            animate={{ width: `${progress}%` }}
            transition={{ duration: 0.5, ease: 'easeOut' }}
          />
        </div>
      </Card>

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
                <span className="inline-flex items-center gap-1.5 text-[11px] font-medium px-2.5 py-1 rounded-full border border-cyan-500/25 bg-gradient-brand-soft text-cyan-300">
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

            <div className="mb-5">
              <span className="text-xs font-semibold uppercase tracking-widest text-gradient-brand mb-2 block">
                Question {session.currentProblemIndex + 1}
              </span>
              <div className="text-base font-medium text-zinc-100">
                <Markdown content={currentProblem.question} />
              </div>
            </div>

            {/* Options */}
            <div className="space-y-2 mb-5">
              {currentProblem.options.map((option, index) => {
                const letter = getAnswerLabel(index)
                const isSelected = selectedAnswer === letter
                const isCorrectOption = letter === currentProblem.correct_answer

                let klass = 'w-full p-3 border rounded-lg text-left transition-all text-sm '
                if (showExplanation) {
                  if (isCorrectOption) klass += 'border-emerald-500 bg-emerald-500/10 text-emerald-400'
                  else if (isSelected) klass += 'border-red-500 bg-red-500/10 text-red-400'
                  else klass += 'border-zinc-700 bg-zinc-800 text-zinc-400'
                } else if (isSelected) {
                  klass += 'border-cyan-500/40 bg-gradient-brand-soft text-cyan-300 ring-1 ring-cyan-500/30'
                } else {
                  klass += 'border-zinc-700 text-zinc-300 hover:border-zinc-600 hover:bg-zinc-800'
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
                          ? 'bg-red-500 text-white'
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
                  <div className={`rounded-lg p-4 mb-5 ${isCorrect ? 'bg-emerald-500/10 border border-emerald-500/20' : 'bg-red-500/10 border border-red-500/20'}`}>
                    <div className="flex items-start gap-2">
                      <motion.div
                        initial={{ scale: 0.6, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        transition={{ delay: 0.1, type: 'spring', stiffness: 300, damping: 18 }}
                      >
                        {isCorrect ? (
                          <CheckCircle className="w-5 h-5 text-emerald-400 flex-shrink-0 mt-0.5" />
                        ) : (
                          <XCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
                        )}
                      </motion.div>
                      <div>
                        <h4 className={`text-sm font-medium mb-1 flex items-center gap-1.5 ${isCorrect ? 'text-emerald-400' : 'text-red-400'}`}>
                          <Eye className="w-3.5 h-3.5" />
                          {isCorrect ? 'Correct!' : 'Not quite right'}
                        </h4>
                        <div className={`text-sm ${isCorrect ? 'text-emerald-400/80' : 'text-red-400/80'}`}>
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
                  leftIcon={<CheckCircle className="w-4 h-4" />}
                >
                  Submit Answer
                </Button>
              ) : session.currentProblemIndex < session.problems.length - 1 ? (
                <Button onClick={nextProblem} rightIcon={<ArrowRight className="w-4 h-4" />}>
                  Next Question
                </Button>
              ) : (
                <Button onClick={nextProblem} rightIcon={<Trophy className="w-4 h-4" />}>
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
