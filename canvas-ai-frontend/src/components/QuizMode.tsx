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
import {
  generateQuiz,
  submitQuizAnswer,
  submitQuiz,
  type QuizQuestion,
  type QuizAnswerResult,
  type QuizResult,
} from '../lib/api'
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
const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

export default function QuizMode({ courseId, userId, onModeChange }: QuizModeProps) {
  const [run, setRun] = useState<QuizRun | null>(null)
  const [result, setResult] = useState<QuizResult | null>(null)
  const [selectedTopic, setSelectedTopic] = useState('')
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
    if (!courseId) {
      setAvailableTopics(['General Topics'])
      setSelectedTopic('General Topics')
      setTopicsError('Please select a course first')
      return
    }
    setTopicsLoading(true)
    setTopicsError(null)
    try {
      const resp = await fetch(`${API_BASE}/practice-topics/${encodeURIComponent(courseId)}`)
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
      const data = (await resp.json()) as { topics?: string[]; error?: string }
      if (data.topics && data.topics.length > 0) {
        setAvailableTopics(data.topics)
        setSelectedTopic(data.topics[0])
        if (data.error) setTopicsError(data.error)
      } else {
        setAvailableTopics(['Course Content', 'General Review'])
        setSelectedTopic('Course Content')
      }
    } catch {
      setAvailableTopics(['Course Content', 'General Review'])
      setSelectedTopic('Course Content')
      setTopicsError('Failed to load topics. Using defaults.')
    } finally {
      setTopicsLoading(false)
    }
  }

  const startQuiz = async () => {
    if (!courseId || !selectedTopic) return
    setLoading(true)
    try {
      const quiz = await generateQuiz(courseId, selectedTopic, difficulty, questionCount)
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
        <div className="bg-zinc-800/60 border border-zinc-700/40 rounded-xl p-5">
          <h2 className="text-base font-medium text-zinc-100 mb-4">Start a quiz</h2>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label className="text-xs font-medium text-zinc-500">Topic</label>
                <button
                  onClick={() => void loadTopics()}
                  disabled={topicsLoading}
                  className="text-cyan-400 hover:text-cyan-300 text-xs flex items-center gap-1"
                >
                  <RefreshCw className={`w-3 h-3 ${topicsLoading ? 'animate-spin' : ''}`} />
                </button>
              </div>
              <div className="relative">
                <button
                  onClick={() => setTopicDropdownOpen(!topicDropdownOpen)}
                  disabled={topicsLoading || !courseId}
                  className="w-full px-3 py-2 border border-zinc-700 rounded-lg bg-zinc-800 text-zinc-100 text-sm text-left flex items-center justify-between disabled:opacity-50"
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
                        className={`w-full px-3 py-2 text-left text-sm hover:bg-zinc-700 transition-colors ${
                          selectedTopic === topic ? 'text-cyan-400 font-medium' : 'text-zinc-400'
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
              <label className="block text-xs font-medium text-zinc-500 mb-1.5">Difficulty</label>
              <select
                value={difficulty}
                onChange={(e) => setDifficulty(e.target.value as 'easy' | 'medium' | 'hard')}
                className="w-full px-3 py-2 border border-zinc-700 rounded-lg bg-zinc-800 text-zinc-100 text-sm"
              >
                <option value="easy">Easy</option>
                <option value="medium">Medium</option>
                <option value="hard">Hard</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-medium text-zinc-500 mb-1.5">Questions</label>
              <select
                value={questionCount}
                onChange={(e) => setQuestionCount(Number(e.target.value))}
                className="w-full px-3 py-2 border border-zinc-700 rounded-lg bg-zinc-800 text-zinc-100 text-sm"
              >
                <option value={5}>5</option>
                <option value={10}>10</option>
                <option value={15}>15</option>
                <option value={20}>20</option>
              </select>
            </div>
          </div>

          <button
            onClick={() => void startQuiz()}
            disabled={loading || topicsLoading || !selectedTopic || !courseId}
            className="bg-cyan-600 text-white px-4 py-2 rounded-lg hover:bg-cyan-500 disabled:opacity-50 disabled:cursor-not-allowed text-sm font-medium flex items-center gap-2 transition-colors"
          >
            {loading ? (
              <>
                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                Generating quiz...
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                Start Quiz
              </>
            )}
          </button>
        </div>
      </div>
    )
  }

  // ── Results screen ───────────────────────────────────────────────────────────
  if (result) {
    const chartData = result.by_topic.map((t) => ({ topic: t.topic, pct: t.pct }))
    const scorePct = result.score.pct
    return (
      <div className="max-w-3xl mx-auto px-5 py-5">
        <div className="bg-zinc-800/60 border border-zinc-700/40 rounded-xl p-5">
          <div className="mb-5">
            <h2 className="text-xl font-semibold text-zinc-100 mb-1">Quiz Complete</h2>
            <p className="text-sm text-zinc-500">Here's how you did</p>
          </div>

          <div className="grid grid-cols-3 gap-4 mb-5">
            <div className="bg-zinc-800 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold mb-0.5 text-cyan-400">{scorePct}%</div>
              <div className="text-xs text-zinc-500">Score</div>
            </div>
            <div className="bg-zinc-800 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold mb-0.5 text-emerald-400">
                {result.score.correct}/{result.score.total}
              </div>
              <div className="text-xs text-zinc-500">Correct</div>
            </div>
            <div className="bg-zinc-800 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold mb-0.5 text-zinc-100">{formatTime(timeElapsed)}</div>
              <div className="text-xs text-zinc-500">Time</div>
            </div>
          </div>

          {chartData.length > 0 && (
            <div className="mb-5">
              <h3 className="text-sm font-medium text-zinc-300 mb-3">Performance by topic</h3>
              <div className="bg-zinc-800 rounded-lg p-4" style={{ height: Math.max(140, chartData.length * 44) }}>
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
            <button
              onClick={resetQuiz}
              className="bg-cyan-600 text-white px-4 py-2 rounded-lg hover:bg-cyan-500 text-sm font-medium flex items-center gap-2 transition-colors"
            >
              <RotateCcw className="w-4 h-4" />
              New Quiz
            </button>
            <button
              onClick={() =>
                onModeChange ? onModeChange('practice') : window.dispatchEvent(new CustomEvent('navigateToPractice'))
              }
              className="bg-zinc-800 border border-zinc-700 text-zinc-400 px-4 py-2 rounded-lg hover:bg-zinc-700 text-sm font-medium flex items-center gap-2 transition-colors"
            >
              <BookOpen className="w-4 h-4" />
              Practice Weak Areas
            </button>
            <button
              onClick={() =>
                onModeChange ? onModeChange('analytics') : window.dispatchEvent(new CustomEvent('navigateToAnalytics'))
              }
              className="bg-zinc-800 border border-zinc-700 text-zinc-400 px-4 py-2 rounded-lg hover:bg-zinc-700 text-sm font-medium flex items-center gap-2 transition-colors"
            >
              <Brain className="w-4 h-4" />
              View Analytics
            </button>
          </div>
        </div>
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
      <div className="bg-zinc-800/60 border border-zinc-700/40 rounded-xl p-5 mb-5">
        <div className="flex items-center justify-between mb-3">
          <div>
            <h2 className="text-base font-medium text-zinc-100">Quiz</h2>
            <p className="text-xs text-zinc-500">{selectedTopic} &middot; {difficulty}</p>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-1.5 text-sm text-zinc-400">
              <Clock className="w-4 h-4" />
              {formatTime(timeElapsed)}
            </div>
            <div className="text-sm text-zinc-400">
              {run.currentIndex + 1}/{run.questions.length}
            </div>
          </div>
        </div>
        <div className="w-full bg-zinc-700 rounded-full h-1.5">
          <div className="bg-cyan-600 h-1.5 rounded-full transition-all duration-500" style={{ width: `${progress}%` }} />
        </div>
      </div>

      <div className="bg-zinc-800/60 border border-zinc-700/40 rounded-xl p-5">
        <div className="mb-5">
          <span className="text-xs font-medium text-cyan-400 mb-2 block">Question {run.currentIndex + 1}</span>
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
              klass += 'border-cyan-500/30 bg-cyan-500/10 text-cyan-400'
            } else {
              klass += 'border-zinc-700 hover:border-zinc-600 hover:bg-zinc-800'
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
                        ? 'bg-cyan-600 text-white'
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
            <button
              onClick={() => void submitAnswer()}
              disabled={!run.selectedLetter || submitting}
              className="bg-cyan-600 text-white px-4 py-2 rounded-lg hover:bg-cyan-500 disabled:opacity-50 disabled:cursor-not-allowed text-sm font-medium flex items-center gap-2 transition-colors"
            >
              {submitting ? (
                <>
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  Checking...
                </>
              ) : (
                <>
                  <CheckCircle className="w-4 h-4" />
                  Submit Answer
                </>
              )}
            </button>
          ) : (
            <button
              onClick={() => void nextQuestion()}
              disabled={submitting}
              className="bg-cyan-600 text-white px-4 py-2 rounded-lg hover:bg-cyan-500 disabled:opacity-50 text-sm font-medium flex items-center gap-2 transition-colors"
            >
              {submitting ? (
                <>
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  Scoring...
                </>
              ) : isLast ? (
                <>
                  View Results
                  <Trophy className="w-4 h-4" />
                </>
              ) : (
                <>
                  Next Question
                  <ArrowRight className="w-4 h-4" />
                </>
              )}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
