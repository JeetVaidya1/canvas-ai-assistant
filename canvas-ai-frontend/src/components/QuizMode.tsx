import { useState, useEffect } from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts'
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
  FileText,
} from 'lucide-react'

import { Markdown } from '@/components/ui/Markdown'
import { Button } from '@/components/ui/Button'
import { Card, PageHeader } from '@/components/ui/Card'
import {
  generateQuiz,
  submitQuizAnswer,
  submitQuiz,
  type QuizQuestion,
  type QuizAnswerResult,
  type QuizResult,
} from '../lib/api'
import { apiFetch } from '../lib/api/client'
import { showError } from '../lib/toast'

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
}

const LETTERS = ['A', 'B', 'C', 'D'] as const
// Sentinel for "quiz the entire course" — sends a null topic so the backend does
// broad whole-course retrieval (core concepts) instead of one narrow topic.
const WHOLE_COURSE = 'Whole course'

export default function QuizMode({ courseId, userId, onModeChange }: QuizModeProps) {
  const [run, setRun] = useState<QuizRun | null>(null)
  const [result, setResult] = useState<QuizResult | null>(null)
  const [selectedTopic, setSelectedTopic] = useState(WHOLE_COURSE)
  const [difficulty, setDifficulty] = useState<'easy' | 'medium' | 'hard'>('medium')
  const [questionCount, setQuestionCount] = useState(10)
  const [loading, setLoading] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [timeElapsed, setTimeElapsed] = useState(0)
  const [availableTopics, setAvailableTopics] = useState<string[]>([])
  const [topicsLoading, setTopicsLoading] = useState(false)
  const [topicsError, setTopicsError] = useState<string | null>(null)
  const [topicDropdownOpen, setTopicDropdownOpen] = useState(false)

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

  useEffect(() => {
    void loadTopics()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [courseId])

  const loadTopics = async () => {
    // "Whole course" is always available and the default — specific topics just
    // let the user narrow the focus. Routed through apiFetch so the auth token
    // is attached (the endpoint is auth-scoped).
    if (!courseId) {
      setAvailableTopics([WHOLE_COURSE])
      setSelectedTopic(WHOLE_COURSE)
      return
    }
    setTopicsLoading(true)
    setTopicsError(null)
    try {
      const data = (await apiFetch(`/practice-topics/${encodeURIComponent(courseId)}`)) as {
        topics?: string[]
        error?: string
      }
      const topics = data.topics?.filter(Boolean) ?? []
      setAvailableTopics([WHOLE_COURSE, ...topics])
      if (data.error) setTopicsError(data.error)
    } catch {
      setAvailableTopics([WHOLE_COURSE])
      setTopicsError('Could not load specific topics — you can still quiz the whole course.')
    } finally {
      setTopicsLoading(false)
    }
  }

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
      setRun({ ...run, feedback })
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

  // ── Setup screen ───────────────────────────────────────────────────────────
  if (!run) {
    return (
      <div className="max-w-3xl mx-auto px-5 py-5">
        <Card>
          <div className="flex items-center gap-3 mb-5">
            <div className="w-10 h-10 rounded-xl bg-gradient-brand-soft border border-cyan-500/15 flex items-center justify-center flex-shrink-0">
              <Brain className="w-5 h-5 text-cyan-300" />
            </div>
            <PageHeader eyebrow="Quiz" title="Start a quiz" subtitle="Pick a topic, difficulty, and length" />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-5">
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label className="text-xs font-medium text-zinc-400">Topic</label>
                <button
                  onClick={() => void loadTopics()}
                  disabled={topicsLoading}
                  className="text-cyan-400 hover:text-cyan-300 text-xs flex items-center gap-1 transition-colors"
                  aria-label="Refresh topics"
                >
                  <RefreshCw className={`w-3 h-3 ${topicsLoading ? 'animate-spin' : ''}`} />
                </button>
              </div>
              <div className="relative">
                <button
                  onClick={() => setTopicDropdownOpen(!topicDropdownOpen)}
                  disabled={topicsLoading || !courseId}
                  className="w-full px-3 py-2 bg-zinc-800/70 border border-zinc-700 rounded-lg focus:border-cyan-500/60 focus:ring-2 focus:ring-cyan-500/20 outline-none text-zinc-100 text-sm text-left flex items-center justify-between disabled:opacity-50 transition-colors"
                >
                  <span className="truncate">
                    {topicsLoading ? 'Loading...' : selectedTopic || 'Select topic'}
                  </span>
                  <ChevronDown className={`w-3.5 h-3.5 transition-transform ${topicDropdownOpen ? 'rotate-180' : ''}`} />
                </button>
                {topicDropdownOpen && !topicsLoading && (
                  <div className="absolute z-10 w-full mt-1 bg-zinc-800 border border-zinc-700 rounded-lg shadow-lg max-h-48 overflow-y-auto">
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
                            : 'text-zinc-400 hover:bg-zinc-700 hover:text-zinc-200'
                        } ${index === 0 ? 'rounded-t-lg' : ''} ${index === availableTopics.length - 1 ? 'rounded-b-lg' : ''}`}
                      >
                        <div className="truncate">{topic}</div>
                      </button>
                    ))}
                  </div>
                )}
              </div>
              {topicsError && <p className="text-xs text-amber-500 mt-1">{topicsError}</p>}
            </div>

            <div>
              <label className="block text-xs font-medium text-zinc-400 mb-1.5">Difficulty</label>
              <select
                value={difficulty}
                onChange={(e) => setDifficulty(e.target.value as 'easy' | 'medium' | 'hard')}
                className="w-full px-3 py-2 bg-zinc-800/70 border border-zinc-700 rounded-lg focus:border-cyan-500/60 focus:ring-2 focus:ring-cyan-500/20 outline-none text-zinc-100 text-sm transition-colors"
              >
                <option value="easy">Easy</option>
                <option value="medium">Medium</option>
                <option value="hard">Hard</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-medium text-zinc-400 mb-1.5">Questions</label>
              <select
                value={questionCount}
                onChange={(e) => setQuestionCount(Number(e.target.value))}
                className="w-full px-3 py-2 bg-zinc-800/70 border border-zinc-700 rounded-lg focus:border-cyan-500/60 focus:ring-2 focus:ring-cyan-500/20 outline-none text-zinc-100 text-sm transition-colors"
              >
                <option value={5}>5</option>
                <option value={10}>10</option>
                <option value={15}>15</option>
                <option value={20}>20</option>
              </select>
            </div>
          </div>

          <Button
            onClick={() => void startQuiz()}
            disabled={loading || topicsLoading || !selectedTopic || !courseId}
            loading={loading}
            leftIcon={<Play className="w-4 h-4" />}
          >
            {loading ? 'Generating quiz...' : 'Start Quiz'}
          </Button>
        </Card>
      </div>
    )
  }

  // ── Results screen ───────────────────────────────────────────────────────────
  if (result) {
    const chartData = result.by_topic.map((t) => ({ topic: t.topic, pct: t.pct }))
    const scorePct = result.score.pct
    return (
      <div className="max-w-3xl mx-auto px-5 py-5">
        <Card>
          <div className="flex items-center gap-3 mb-5">
            <div className="w-10 h-10 rounded-xl bg-gradient-brand-soft border border-cyan-500/15 flex items-center justify-center flex-shrink-0">
              <Trophy className="w-5 h-5 text-cyan-300" />
            </div>
            <div>
              <h2 className="text-xl font-semibold text-gradient-brand mb-0.5">Quiz Complete</h2>
              <p className="text-sm text-zinc-500">Here's how you did</p>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-4 mb-5">
            <div className="bg-gradient-brand-soft border border-cyan-500/15 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold mb-0.5 text-gradient-brand">{scorePct}%</div>
              <div className="text-xs text-zinc-500">Score</div>
            </div>
            <div className="bg-zinc-800/70 border border-zinc-700/50 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold mb-0.5 text-emerald-400">
                {result.score.correct}/{result.score.total}
              </div>
              <div className="text-xs text-zinc-500">Correct</div>
            </div>
            <div className="bg-zinc-800/70 border border-zinc-700/50 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold mb-0.5 text-zinc-100">{formatTime(timeElapsed)}</div>
              <div className="text-xs text-zinc-500">Time</div>
            </div>
          </div>

          {chartData.length > 0 && (
            <div className="mb-5">
              <h3 className="text-sm font-medium text-zinc-300 mb-3">Performance by topic</h3>
              <div className="bg-zinc-800/70 border border-zinc-700/50 rounded-lg p-4" style={{ height: Math.max(140, chartData.length * 44) }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} layout="vertical" margin={{ left: 8, right: 24, top: 4, bottom: 4 }}>
                    <XAxis type="number" domain={[0, 100]} tick={{ fill: '#a1a1aa', fontSize: 11 }} stroke="#3f3f46" />
                    <YAxis
                      type="category"
                      dataKey="topic"
                      width={120}
                      tick={{ fill: '#a1a1aa', fontSize: 11 }}
                      stroke="#3f3f46"
                    />
                    <Tooltip
                      cursor={{ fill: '#27272a' }}
                      contentStyle={{ background: '#18181b', border: '1px solid #3f3f46', borderRadius: 8, fontSize: 12 }}
                      formatter={(value) => [`${value ?? 0}%`, 'Score'] as [string, string]}
                    />
                    <Bar dataKey="pct" radius={[0, 4, 4, 0]}>
                      {chartData.map((entry) => (
                        <Cell
                          key={entry.topic}
                          fill={entry.pct >= 70 ? '#10b981' : entry.pct >= 40 ? '#f59e0b' : '#ef4444'}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {result.weak_areas.length > 0 && (
            <div className="mb-5 bg-amber-500/10 border border-amber-500/20 rounded-lg p-3">
              <div className="flex items-start gap-2">
                <Target className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
                <div>
                  <h4 className="text-sm font-medium text-amber-400 mb-1">Weak areas to review</h4>
                  <p className="text-sm text-amber-400/80">{result.weak_areas.join(', ')}</p>
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
                onModeChange ? onModeChange('practice') : window.dispatchEvent(new CustomEvent('navigateToPractice'))
              }
              leftIcon={<BookOpen className="w-4 h-4" />}
            >
              Practice Weak Areas
            </Button>
            <Button
              variant="secondary"
              onClick={() =>
                onModeChange ? onModeChange('analytics') : window.dispatchEvent(new CustomEvent('navigateToAnalytics'))
              }
              leftIcon={<Brain className="w-4 h-4" />}
            >
              View Analytics
            </Button>
          </div>
        </Card>
      </div>
    )
  }

  // ── Active quiz ─────────────────────────────────────────────────────────────
  const question = run.questions[run.currentIndex]
  const feedback = run.feedback
  const progress = ((run.currentIndex + 1) / run.questions.length) * 100
  const isLast = run.currentIndex === run.questions.length - 1

  return (
    <div className="max-w-3xl mx-auto p-5">
      <Card className="mb-5">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-brand-soft border border-cyan-500/15 flex items-center justify-center flex-shrink-0">
              <Brain className="w-5 h-5 text-cyan-300" />
            </div>
            <div>
              <h2 className="text-base font-medium text-zinc-100">Quiz</h2>
              <p className="text-xs text-zinc-500">{selectedTopic} &middot; {difficulty}</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-1.5 text-sm text-zinc-400">
              <Clock className="w-4 h-4 text-cyan-300" />
              {formatTime(timeElapsed)}
            </div>
            <div className="text-sm text-zinc-400">
              {run.currentIndex + 1}/{run.questions.length}
            </div>
          </div>
        </div>
        <div className="w-full bg-zinc-700/60 rounded-full h-1.5">
          <div className="bg-gradient-brand h-1.5 rounded-full transition-all duration-500" style={{ width: `${progress}%` }} />
        </div>
      </Card>

      <Card>
        <div className="mb-5">
          <span className="text-xs font-semibold uppercase tracking-widest text-gradient-brand mb-2 block">Question {run.currentIndex + 1}</span>
          <div className="text-base font-medium text-zinc-100">
            <Markdown content={question.question} />
          </div>
        </div>

        <div className="space-y-2 mb-5">
          {question.options.map((option, index) => {
            const letter = LETTERS[index] ?? String.fromCharCode(65 + index)
            const isSelected = run.selectedLetter === letter
            const isCorrectOption = feedback ? letter === feedback.correct_answer : false

            let klass = 'w-full p-3 border rounded-lg text-left transition-all text-sm '
            if (feedback) {
              if (isCorrectOption) klass += 'border-emerald-500 bg-emerald-500/10 text-emerald-400'
              else if (isSelected) klass += 'border-red-500 bg-red-500/10 text-red-400'
              else klass += 'border-zinc-700 bg-zinc-800 text-zinc-400'
            } else if (isSelected) {
              klass += 'border-transparent bg-gradient-brand-soft text-cyan-300 ring-2 ring-cyan-500/30'
            } else {
              klass += 'border-zinc-700 hover:border-cyan-500/40 hover:bg-cyan-500/5'
            }

            return (
              <button
                key={index}
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
              </button>
            )
          })}
        </div>

        {feedback && (
          <div
            className={`rounded-lg p-4 mb-5 ${
              feedback.is_correct ? 'bg-emerald-500/10 border border-emerald-500/20' : 'bg-red-500/10 border border-red-500/20'
            }`}
          >
            <div className="flex items-start gap-2">
              {feedback.is_correct ? (
                <CheckCircle className="w-5 h-5 text-emerald-400 flex-shrink-0 mt-0.5" />
              ) : (
                <XCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
              )}
              <div className="min-w-0">
                <h4 className={`text-sm font-medium mb-1 ${feedback.is_correct ? 'text-emerald-400' : 'text-red-400'}`}>
                  {feedback.is_correct ? 'Correct!' : 'Not quite right'}
                </h4>
                <div className={`text-sm ${feedback.is_correct ? 'text-emerald-400/80' : 'text-red-400/80'}`}>
                  <Markdown content={feedback.explanation} />
                </div>
                {!feedback.is_correct && feedback.mistake_explanation && (
                  <div className="mt-2 rounded-lg bg-zinc-800/70 border border-zinc-700 p-2.5">
                    <div className="text-xs font-medium text-amber-400 mb-0.5">Why you missed this</div>
                    <div className="text-sm text-zinc-300"><Markdown content={feedback.mistake_explanation} /></div>
                    {feedback.mistake_source?.doc_name && (
                      <div className="mt-1.5 inline-flex items-center gap-1 text-xs text-zinc-500">
                        <FileText className="w-3 h-3" />
                        {feedback.mistake_source.doc_name}
                        {feedback.mistake_source.page ? `, p.${feedback.mistake_source.page}` : ''}
                      </div>
                    )}
                  </div>
                )}
                {feedback.source?.doc_name && (
                  <div className="mt-2 inline-flex items-center gap-1.5 text-xs text-zinc-400 bg-zinc-800 border border-zinc-700 rounded-full px-2.5 py-1">
                    <FileText className="w-3 h-3" />
                    {feedback.source.doc_name}
                    {feedback.source.page ? `, p.${feedback.source.page}` : ''}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        <div>
          {!feedback ? (
            <Button
              onClick={() => void submitAnswer()}
              disabled={!run.selectedLetter || submitting}
              loading={submitting}
              leftIcon={<CheckCircle className="w-4 h-4" />}
            >
              {submitting ? 'Checking...' : 'Submit Answer'}
            </Button>
          ) : (
            <Button
              onClick={() => void nextQuestion()}
              disabled={submitting}
              loading={submitting}
              rightIcon={isLast ? <Trophy className="w-4 h-4" /> : <ArrowRight className="w-4 h-4" />}
            >
              {submitting ? 'Scoring...' : isLast ? 'View Results' : 'Next Question'}
            </Button>
          )}
        </div>
      </Card>
    </div>
  )
}
