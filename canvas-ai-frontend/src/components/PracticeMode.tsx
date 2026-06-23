import { useState, useEffect } from 'react'
import { Markdown } from '@/components/ui/Markdown'
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
  RefreshCw
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

export default function PracticeMode({ courseId, userId, onModeChange }: PracticeModeProps) {
  const [session, setSession] = useState<PracticeSession | null>(null)
  const [selectedTopic, setSelectedTopic] = useState('')
  const [difficulty, setDifficulty] = useState<'easy' | 'medium' | 'hard'>('medium')
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
      const problems = await generatePracticeProblems(courseId, selectedTopic, difficulty, problemCount)
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

  // Setup screen
  if (!session) {
    return (
      <div className="max-w-3xl mx-auto px-5 py-5">
        <div className="bg-zinc-800/60 border border-zinc-700/40 rounded-xl p-5">
          <h2 className="text-base font-medium text-zinc-100 mb-4">Start a practice session</h2>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
            {/* Topic */}
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label className="text-xs font-medium text-zinc-500">Topic</label>
                <button
                  onClick={() => loadTopics()}
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
              {topicsError && <p className="text-xs text-red-500 mt-1">{topicsError}</p>}
            </div>

            {/* Difficulty */}
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

            {/* Problem count */}
            <div>
              <label className="block text-xs font-medium text-zinc-500 mb-1.5">Problems</label>
              <select
                value={problemCount}
                onChange={(e) => setProblemCount(Number(e.target.value))}
                className="w-full px-3 py-2 border border-zinc-700 rounded-lg bg-zinc-800 text-zinc-100 text-sm"
              >
                <option value={3}>3</option>
                <option value={5}>5</option>
                <option value={10}>10</option>
                <option value={15}>15</option>
              </select>
            </div>
          </div>

          <button
            onClick={startPracticeSession}
            disabled={loading || topicsLoading || !selectedTopic || !courseId}
            className="bg-cyan-600 text-white px-4 py-2 rounded-lg hover:bg-cyan-500 disabled:opacity-50 disabled:cursor-not-allowed text-sm font-medium flex items-center gap-2 transition-colors"
          >
            {loading ? (
              <>
                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                Generating...
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                Start Session
              </>
            )}
          </button>
        </div>
      </div>
    )
  }

  // Results screen
  if (session.isComplete) {
    const correctCount = session.userAnswers.filter((a, i) => a === session.problems[i].correct_answer).length
    return (
      <div className="max-w-3xl mx-auto px-5 py-5">
        <div className="bg-zinc-800/60 border border-zinc-700/40 rounded-xl p-5">
          <div className="mb-5">
            <h2 className="text-xl font-semibold text-zinc-100 mb-1">Practice Complete</h2>
            <p className="text-sm text-zinc-500">Here's how you did</p>
          </div>

          <div className="grid grid-cols-3 gap-4 mb-5">
            <div className="bg-zinc-800 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold mb-0.5 text-cyan-400">{session.score}%</div>
              <div className="text-xs text-zinc-500">Score</div>
            </div>
            <div className="bg-zinc-800 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold mb-0.5 text-emerald-400">{correctCount}/{session.problems.length}</div>
              <div className="text-xs text-zinc-500">Correct</div>
            </div>
            <div className="bg-zinc-800 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold mb-0.5 text-zinc-100">{formatTime(timeElapsed)}</div>
              <div className="text-xs text-zinc-500">Time</div>
            </div>
          </div>

          {/* Performance summary */}
          <div className="mb-5">
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
          </div>

          <div className="flex gap-3">
            <button
              onClick={() => setSession(null)}
              className="bg-cyan-600 text-white px-4 py-2 rounded-lg hover:bg-cyan-500 text-sm font-medium flex items-center gap-2 transition-colors"
            >
              <RotateCcw className="w-4 h-4" />
              Practice Again
            </button>
            <button
              onClick={() => onModeChange ? onModeChange('analytics') : window.dispatchEvent(new CustomEvent('navigateToAnalytics'))}
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

  // Active session
  const currentProblem = getCurrentProblem()
  if (!currentProblem) return null

  const isCorrect = selectedAnswer === currentProblem.correct_answer
  const progress = ((session.currentProblemIndex + 1) / session.problems.length) * 100

  return (
    <div className="max-w-3xl mx-auto p-5">
      {/* Header */}
      <div className="bg-zinc-800/60 border border-zinc-700/40 rounded-xl p-5 mb-5">
        <div className="flex items-center justify-between mb-3">
          <div>
            <h2 className="text-base font-medium text-zinc-100">Practice Session</h2>
            <p className="text-xs text-zinc-500">{selectedTopic} &middot; {difficulty}</p>
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
        <div className="w-full bg-zinc-700 rounded-full h-1.5">
          <div
            className="bg-cyan-600 h-1.5 rounded-full transition-all duration-500"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      {/* Question */}
      <div className="bg-zinc-800/60 border border-zinc-700/40 rounded-xl p-5">
        <div className="mb-5">
          <span className="text-xs font-medium text-cyan-400 mb-2 block">
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
              klass += 'border-cyan-500/30 bg-cyan-500/10 text-cyan-400'
            } else {
              klass += 'border-zinc-700 hover:border-zinc-600 hover:bg-zinc-800'
            }

            return (
              <button
                key={index}
                onClick={() => !showExplanation && setSelectedAnswer(letter)}
                disabled={showExplanation}
                className={klass}
              >
                <div className="flex items-center gap-3">
                  <div className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold ${
                    showExplanation && isCorrectOption
                      ? 'bg-emerald-500 text-white'
                      : showExplanation && isSelected && !isCorrectOption
                      ? 'bg-red-500 text-white'
                      : isSelected
                      ? 'bg-cyan-600 text-white'
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
              </button>
            )
          })}
        </div>

        {/* Explanation */}
        {showExplanation && (
          <div className={`rounded-lg p-4 mb-5 ${isCorrect ? 'bg-emerald-500/10 border border-emerald-500/20' : 'bg-red-500/10 border border-red-500/20'}`}>
            <div className="flex items-start gap-2">
              {isCorrect ? (
                <CheckCircle className="w-5 h-5 text-emerald-400 flex-shrink-0 mt-0.5" />
              ) : (
                <XCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
              )}
              <div>
                <h4 className={`text-sm font-medium mb-1 ${isCorrect ? 'text-emerald-400' : 'text-red-400'}`}>
                  {isCorrect ? 'Correct!' : 'Not quite right'}
                </h4>
                <div className={`text-sm ${isCorrect ? 'text-emerald-400/80' : 'text-red-400/80'}`}>
                  <Markdown content={currentProblem.explanation} />
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Actions */}
        <div>
          {!showExplanation ? (
            <button
              onClick={submitAnswer}
              disabled={!selectedAnswer}
              className="bg-cyan-600 text-white px-4 py-2 rounded-lg hover:bg-cyan-500 disabled:opacity-50 disabled:cursor-not-allowed text-sm font-medium flex items-center gap-2 transition-colors"
            >
              <CheckCircle className="w-4 h-4" />
              Submit Answer
            </button>
          ) : (
            <button
              onClick={nextProblem}
              className="bg-cyan-600 text-white px-4 py-2 rounded-lg hover:bg-cyan-500 text-sm font-medium flex items-center gap-2 transition-colors"
            >
              {session.currentProblemIndex < session.problems.length - 1 ? (
                <>
                  Next Question
                  <ArrowRight className="w-4 h-4" />
                </>
              ) : (
                <>
                  View Results
                  <Trophy className="w-4 h-4" />
                </>
              )}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
