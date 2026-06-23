import { useState, useEffect } from 'react'
import {
  Upload,
  Timer,
  Play,
  Pause,
  RotateCcw,
  CheckCircle,
  Brain,
  Eye
} from 'lucide-react'

import {
  generatePracticeExam,
  createExamSession,
  startExamSession,
  pauseExamSession,
  saveExamAnswer,
  navigateExamQuestion,
  submitExamSession as submitExamApi,
  solveExamQuestion,
  uploadPastPaper
} from '../lib/api'
import { showError } from '../lib/toast'

type QType = 'multiple_choice' | 'calculation' | 'short_answer' | 'essay' | 'diagram' | 'proof'
type Diff = 'easy' | 'medium' | 'hard'

interface ExamQuestion {
  id: string
  type: QType
  question: string
  options?: string[]
  points: number
  topic: string
  difficulty: Diff
  answer?: string
  solution?: string
  timeEstimate: number
}

interface ExamSession {
  id: string
  examName: string
  questions: ExamQuestion[]
  timeLimit: number
  startTime?: Date
  endTime?: Date
  userAnswers: Record<string, string>
  currentQuestion: number
  isActive: boolean
  isPaused: boolean
}

interface ExamModeProps {
  courseId: string
  userId: string
}

type SolveJSON = {
  final_answer: string
  steps: string[]
  choice?: string | null
  units?: string | null
}

export default function ExamMode({ courseId, userId }: ExamModeProps) {
  const [examSession, setExamSession] = useState<ExamSession | null>(null)
  const [loading, setLoading] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [analysisSummary, setAnalysisSummary] = useState<any>(null)

  const [timeRemaining, setTimeRemaining] = useState(0)
  const [currentAnswer, setCurrentAnswer] = useState('')
  const [showResults, setShowResults] = useState(false)
  const [examResults, setExamResults] = useState<any>(null)

  const [sessionId, setSessionId] = useState<string | null>(null)

  const [hinting, setHinting] = useState(false)
  const [solving, setSolving] = useState(false)
  const [hints, setHints] = useState<Record<string, SolveJSON>>({})
  const [solutions, setSolutions] = useState<Record<string, SolveJSON>>({})

  useEffect(() => {
    let interval: number | undefined
    if (examSession?.isActive && !examSession.isPaused && timeRemaining > 0) {
      interval = window.setInterval(() => {
        setTimeRemaining(prev => {
          if (prev <= 1) {
            submitExam()
            return 0
          }
          return prev - 1
        })
      }, 1000)
    }
    return () => {
      if (interval) window.clearInterval(interval)
    }
  }, [examSession?.isActive, examSession?.isPaused, timeRemaining])

  const formatTime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    const secs = seconds % 60
    if (hours > 0) return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
    return `${minutes}:${secs.toString().padStart(2, '0')}`
  }

  const generateExamFromPastPaper = async () => {
    if (!courseId) return
    setLoading(true)
    try {
      const result = await generatePracticeExam({
        courseId,
        examType: 'practice',
        difficulty: 'mixed' as any,
        questionCount: 12,
        timeLimit: 120,
        userId: userId || 'anonymous',
        questionTypes: ['multiple_choice','calculation','short_answer']
      })

      const examData = result.exam
      const mapped: ExamQuestion[] = (examData.questions || []).map((q: any) => ({
        id: q.id,
        type: q.type,
        question: q.question,
        options: q.options ?? undefined,
        points: q.points ?? 2,
        topic: q.topic ?? 'General',
        difficulty: (q.difficulty ?? 'medium') as Diff,
        answer: q.correct_answer ?? '',
        solution: q.explanation ?? '',
        timeEstimate: q.time_estimate ?? 3
      }))

      const newSession: ExamSession = {
        id: examData.id,
        examName: examData.name,
        questions: mapped,
        timeLimit: examData.time_limit ?? 120,
        userAnswers: {},
        currentQuestion: 0,
        isActive: false,
        isPaused: false
      }
      setExamSession(newSession)

      try {
        const created = await createExamSession(courseId, userId || 'anonymous', examData)
        if (created?.session?.id) setSessionId(created.session.id)
      } catch (e) {
        console.warn('create session failed (non-fatal):', e)
      }
    } catch (error: any) {
      console.error('Failed to generate exam:', error)
      showError(`Could not generate exam: ${error.message || error}`)
    } finally {
      setLoading(false)
    }
  }

  const startExam = async () => {
    if (!examSession) return
    setExamSession({
      ...examSession,
      isActive: true,
      startTime: new Date()
    })
    setTimeRemaining((examSession.timeLimit ?? 120) * 60)
    setCurrentAnswer('')
    if (sessionId) {
      try { await startExamSession(sessionId) } catch (e) { console.warn(e) }
    }
  }

  const pauseExam = async () => {
    if (!examSession) return
    const newPaused = !examSession.isPaused
    setExamSession({ ...examSession, isPaused: newPaused })
    if (sessionId) {
      try { await pauseExamSession(sessionId) } catch (e) { console.warn(e) }
    }
  }

  const saveAnswer = async () => {
    if (!examSession) return
    const currentQ = examSession.questions[examSession.currentQuestion]
    setExamSession({
      ...examSession,
      userAnswers: { ...examSession.userAnswers, [currentQ.id]: currentAnswer }
    })
    if (sessionId) {
      try { await saveExamAnswer(sessionId, currentQ.id, currentAnswer) } catch (e) { console.warn(e) }
    }
  }

  const goToQuestion = async (index: number) => {
    if (!examSession) return
    await saveAnswer()
    setExamSession({ ...examSession, currentQuestion: index })
    setCurrentAnswer(examSession.userAnswers[examSession.questions[index].id] || '')
    if (sessionId) {
      try { await navigateExamQuestion(sessionId, index) } catch (e) { console.warn(e) }
    }
  }

  const nextQuestion = () => {
    if (!examSession) return
    saveAnswer()
    if (examSession.currentQuestion < examSession.questions.length - 1) {
      const nextIndex = examSession.currentQuestion + 1
      setExamSession({ ...examSession, currentQuestion: nextIndex })
      const nextQ = examSession.questions[nextIndex]
      setCurrentAnswer(examSession.userAnswers[nextQ.id] || '')
    }
  }

  const previousQuestion = () => {
    if (!examSession) return
    saveAnswer()
    if (examSession.currentQuestion > 0) {
      const prevIndex = examSession.currentQuestion - 1
      setExamSession({ ...examSession, currentQuestion: prevIndex })
      const prevQ = examSession.questions[prevIndex]
      setCurrentAnswer(examSession.userAnswers[prevQ.id] || '')
    }
  }

  const submitExam = async () => {
    if (!examSession) return
    await saveAnswer()
    const finalSession = { ...examSession, isActive: false, endTime: new Date() }
    setExamSession(finalSession)

    if (sessionId) {
      try {
        const res = await submitExamApi(sessionId)
        const results = res.results || res.final_score || {}
        setExamResults({
          totalQuestions: results.total_questions ?? finalSession.questions.length,
          correctAnswers: results.correct_answers ?? 0,
          totalPoints: results.total_points ?? finalSession.questions.reduce((a: number, q: ExamQuestion) => a + (q.points ?? 0), 0),
          earnedPoints: results.earned_points ?? 0,
          percentage: results.percentage ?? 0,
          timeSpent: results.time_metrics?.time_used_minutes ?? (finalSession.timeLimit - Math.floor(timeRemaining / 60)),
          breakdown: results.question_results?.map((r: any) => ({
            question: r.question,
            userAnswer: r.user_answer,
            correctAnswer: r.correct_answer,
            points: r.points_possible,
            topic: r.topic
          })) ?? []
        })
        setShowResults(true)
        return
      } catch (e) {
        console.warn('Server submit failed; using local scoring.', e)
      }
    }

    const results = calculateResults(finalSession)
    setExamResults(results)
    setShowResults(true)
  }

  const calculateResults = (session: ExamSession) => {
    let totalPoints = 0
    let earnedPoints = 0
    let correctAnswers = 0

    session.questions.forEach(question => {
      totalPoints += question.points
      const userAnswer = session.userAnswers[question.id]
      if (userAnswer && question.answer) {
        const isCorrect = userAnswer.toLowerCase().trim() === question.answer.toLowerCase().trim()
        if (isCorrect) {
          earnedPoints += question.points
          correctAnswers++
        }
      }
    })

    return {
      totalQuestions: session.questions.length,
      correctAnswers,
      totalPoints,
      earnedPoints,
      percentage: Math.round((earnedPoints / Math.max(1, totalPoints)) * 100),
      timeSpent: session.timeLimit - Math.floor(timeRemaining / 60),
      breakdown: session.questions.map(q => ({
        question: q.question,
        userAnswer: session.userAnswers[q.id],
        correctAnswer: q.answer,
        points: q.points,
        topic: q.topic
      }))
    }
  }

  const onUploadPastPaper = async (file: File) => {
    if (!file || !courseId) return
    setUploading(true)
    try {
      const data = await uploadPastPaper(courseId, file, userId || 'anonymous')
      setAnalysisSummary(data)
    } catch (e: any) {
      console.error(e)
      showError(`Upload failed: ${e.message || e}`)
    } finally {
      setUploading(false)
    }
  }

  const requestHint = async () => {
    if (!examSession || !courseId) return
    setHinting(true)
    try {
      const q = examSession.questions[examSession.currentQuestion]
      const res = await solveExamQuestion({
        courseId,
        questionText: q.question,
        wantHint: false
      })
      const s: SolveJSON = res.solution
      setHints(prev => ({ ...prev, [q.id]: s }))
    } catch (e: any) {
      console.error(e)
      showError(`Hint failed: ${e.message || e}`)
    } finally {
      setHinting(false)
    }
  }

  const requestSolution = async () => {
    if (!examSession || !courseId) return
    setSolving(true)
    try {
      const q = examSession.questions[examSession.currentQuestion]
      const res = await solveExamQuestion({
        courseId,
        questionText: q.question,
        wantHint: false
      } as any)
      const s: SolveJSON = res.solution
      setSolutions(prev => ({ ...prev, [q.id]: s }))
    } catch (e: any) {
      console.error(e)
      showError(`Solve failed: ${e.message || e}`)
    } finally {
      setSolving(false)
    }
  }

  const sampleExam: ExamSession = {
    id: 'sample',
    examName: 'Physics 122 Practice Exam',
    timeLimit: 120,
    currentQuestion: 0,
    isActive: false,
    isPaused: false,
    userAnswers: {},
    questions: [
      {
        id: '1', type: 'multiple_choice',
        question: 'If your heart is beating at 76.0 beats per minute, what is the frequency in hertz?',
        options: ['1.27 Hz', '0.79 Hz', '1.33 Hz', '0.76 Hz'],
        points: 2, topic: 'Oscillations', difficulty: 'easy', answer: 'A', timeEstimate: 3
      },
      {
        id: '2', type: 'calculation',
        question: 'A 4.0-g string is 0.36 m long. It vibrates at 500 Hz in its third harmonic. What is the wavelength?',
        points: 3, topic: 'Waves', difficulty: 'medium', answer: '0.24 m',
        solution: 'Third harmonic: L = 3\u03bb/2 \u2192 \u03bb = 2L/3 = 0.24 m', timeEstimate: 5
      },
      {
        id: '3', type: 'short_answer',
        question: 'Three point charges are located on the x-axis. Calculate the magnitude of the electric force on the middle charge.',
        points: 4, topic: 'Electrostatics', difficulty: 'hard', timeEstimate: 8
      }
    ]
  }

  // Result screen
  if (showResults && examResults) {
    return (
      <div className="max-w-3xl mx-auto p-5">
        <div className="bg-zinc-800/60 border border-zinc-700/40 rounded-xl p-5">
          <div className="mb-5">
            <h2 className="text-xl font-semibold text-zinc-100 mb-1">Exam Complete</h2>
            <p className="text-sm text-zinc-500">Here are your results</p>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-5">
            <div className="bg-zinc-800 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold mb-0.5 text-cyan-400">{examResults.percentage}%</div>
              <div className="text-xs text-zinc-500">Score</div>
            </div>
            <div className="bg-zinc-800 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold mb-0.5 text-emerald-400">{examResults.correctAnswers}/{examResults.totalQuestions}</div>
              <div className="text-xs text-zinc-500">Correct</div>
            </div>
            <div className="bg-zinc-800 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold mb-0.5 text-zinc-100">{examResults.earnedPoints}/{examResults.totalPoints}</div>
              <div className="text-xs text-zinc-500">Points</div>
            </div>
            <div className="bg-zinc-800 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold mb-0.5 text-amber-400">{examResults.timeSpent}m</div>
              <div className="text-xs text-zinc-500">Time</div>
            </div>
          </div>

          <div className="flex gap-3">
            <button
              onClick={() => {
                setShowResults(false)
                setExamSession(null)
                setExamResults(null)
                setSessionId(null)
                setHints({})
                setSolutions({})
              }}
              className="bg-cyan-600 text-white px-4 py-2 rounded-lg hover:bg-cyan-500 text-sm font-medium flex items-center gap-2 transition-colors"
            >
              <RotateCcw className="w-4 h-4" />
              New Exam
            </button>
            <button
              onClick={() => {/* could toggle a review mode */}}
              className="bg-zinc-800 border border-zinc-700 text-zinc-400 px-4 py-2 rounded-lg hover:bg-zinc-700 text-sm font-medium flex items-center gap-2 transition-colors"
            >
              <Eye className="w-4 h-4" />
              Review Answers
            </button>
          </div>
        </div>
      </div>
    )
  }

  // Landing — no session yet
  if (!examSession) {
    return (
      <div className="max-w-3xl mx-auto px-5 py-5">
        <div className="bg-zinc-800/60 border border-zinc-700/40 rounded-xl p-5">
          <h2 className="text-base font-medium text-zinc-100 mb-4">Start an exam</h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            {/* Generate */}
            <div className="bg-zinc-800 rounded-lg p-4">
              <h3 className="text-sm font-medium text-zinc-200 mb-1">Generate from course materials</h3>
              <p className="text-xs text-zinc-500 mb-3">AI creates a practice exam from your uploaded files</p>
              <button
                onClick={generateExamFromPastPaper}
                disabled={loading || !courseId}
                className="bg-cyan-600 text-white px-4 py-2 rounded-lg hover:bg-cyan-500 disabled:opacity-50 text-sm font-medium flex items-center gap-2 transition-colors"
              >
                {loading ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <Brain className="w-4 h-4" />
                    Generate Exam
                  </>
                )}
              </button>
            </div>

            {/* Upload */}
            <div className="bg-zinc-800 rounded-lg p-4">
              <h3 className="text-sm font-medium text-zinc-200 mb-1">Upload a past paper</h3>
              <p className="text-xs text-zinc-500 mb-3">We'll analyze it to create similar practice questions</p>
              <label className={`inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium cursor-pointer transition-colors ${
                courseId ? 'bg-zinc-700 hover:bg-zinc-600 text-white' : 'bg-zinc-700 text-zinc-500 cursor-not-allowed'
              }`}>
                <Upload className="w-4 h-4" />
                {uploading ? 'Uploading...' : 'Choose File'}
                <input
                  type="file"
                  accept=".pdf"
                  className="hidden"
                  onChange={(e) => {
                    const f = e.target.files?.[0]
                    if (f && courseId) onUploadPastPaper(f)
                  }}
                  disabled={!courseId}
                />
              </label>
              {analysisSummary && (
                <div className="text-xs text-zinc-500 mt-2">
                  {analysisSummary.status === 'success' ? (
                    <span>{analysisSummary.questions_found} questions found</span>
                  ) : (
                    <span className="text-red-400">{analysisSummary.message ?? 'Upload failed'}</span>
                  )}
                </div>
              )}
            </div>
          </div>

          <button
            onClick={() => setExamSession(sampleExam)}
            className="text-xs text-zinc-500 hover:text-zinc-400 transition-colors"
          >
            Or try a sample exam
          </button>
        </div>
      </div>
    )
  }

  // Pre-start — compact confirmation
  if (!examSession.isActive) {
    return (
      <div className="max-w-3xl mx-auto px-5 py-5">
        <div className="bg-zinc-800/60 border border-zinc-700/40 rounded-xl p-5">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-base font-medium text-zinc-100">{examSession.examName}</h2>
              <p className="text-xs text-zinc-500">{examSession.questions.length} questions &middot; {examSession.timeLimit} min</p>
            </div>
            <button
              onClick={startExam}
              className="bg-cyan-600 text-white px-4 py-2 rounded-lg hover:bg-cyan-500 text-sm font-medium flex items-center gap-2 transition-colors"
            >
              <Play className="w-4 h-4" />
              Begin Exam
            </button>
          </div>
        </div>
      </div>
    )
  }

  // Active exam
  const currentQ = examSession.questions[examSession.currentQuestion]
  const progress = ((examSession.currentQuestion + 1) / examSession.questions.length) * 100

  return (
    <div className="max-w-3xl mx-auto p-5">
      {/* Header */}
      <div className="bg-zinc-800/60 border border-zinc-700/40 rounded-xl p-5 mb-5">
        <div className="flex items-center justify-between mb-3">
          <div>
            <h2 className="text-base font-medium text-zinc-100">{examSession.examName}</h2>
            <p className="text-xs text-zinc-500">
              Question {examSession.currentQuestion + 1} of {examSession.questions.length}
            </p>
          </div>
          <div className="flex items-center gap-4">
            <div className={`flex items-center gap-1.5 text-sm font-medium ${
              timeRemaining < 300 ? 'text-red-400' : 'text-zinc-400'
            }`}>
              <Timer className="w-4 h-4" />
              {formatTime(timeRemaining)}
            </div>
            <button
              onClick={pauseExam}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-zinc-800 hover:bg-zinc-700 rounded-lg transition-colors text-sm text-zinc-400"
            >
              {examSession.isPaused ? <Play className="w-3.5 h-3.5" /> : <Pause className="w-3.5 h-3.5" />}
              {examSession.isPaused ? 'Resume' : 'Pause'}
            </button>
          </div>
        </div>
        <div className="w-full bg-zinc-800 rounded-full h-1.5">
          <div
            className="bg-cyan-600 h-1.5 rounded-full transition-all duration-500"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      {/* Question */}
      <div className="bg-zinc-800/60 border border-zinc-700/40 rounded-xl p-5 mb-5">
        <div className="mb-5">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-xs font-medium text-cyan-400">
              {currentQ.points} pt{currentQ.points !== 1 ? 's' : ''}
            </span>
            <span className="text-zinc-600 text-xs">&middot;</span>
            <span className="text-xs text-zinc-500">{currentQ.topic}</span>
          </div>
          <h3 className="text-base font-medium text-zinc-100 leading-relaxed">
            {currentQ.question}
          </h3>
        </div>

        {/* Answer Input */}
        {currentQ.type === 'multiple_choice' && currentQ.options ? (
          <div className="space-y-2 mb-5">
            {currentQ.options.map((option, index) => {
              const letter = String.fromCharCode(65 + index)
              const isSelected = currentAnswer === letter
              return (
                <button
                  key={index}
                  onClick={() => setCurrentAnswer(letter)}
                  className={`w-full p-3 border rounded-lg text-left transition-all text-sm ${
                    isSelected
                      ? 'border-cyan-500/30 bg-cyan-500/10 text-cyan-400'
                      : 'border-zinc-700 hover:border-zinc-600 hover:bg-zinc-800'
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <div className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold ${
                      isSelected ? 'bg-cyan-600 text-white' : 'bg-zinc-800 text-zinc-400'
                    }`}>
                      {letter}
                    </div>
                    <span className="flex-1 text-zinc-100">{option}</span>
                  </div>
                </button>
              )
            })}
          </div>
        ) : (
          <div className="mb-5">
            <textarea
              value={currentAnswer}
              onChange={(e) => setCurrentAnswer(e.target.value)}
              placeholder="Enter your answer here..."
              className="w-full h-28 p-3 bg-zinc-800 border border-zinc-700 text-zinc-100 rounded-lg focus:ring-1 focus:ring-cyan-500/50 focus:border-transparent resize-none placeholder-zinc-500 text-sm"
            />
          </div>
        )}

        {/* Hint / Solution buttons */}
        <div className="flex items-center gap-2 mb-5">
          <button
            onClick={requestHint}
            disabled={hinting || !courseId}
            className="px-3 py-1.5 rounded-lg bg-amber-500/10 text-amber-400 hover:bg-amber-500/20 disabled:opacity-50 text-xs font-medium"
          >
            {hinting ? 'Getting hint...' : 'Get Hint'}
          </button>
          <button
            onClick={requestSolution}
            disabled={solving || !courseId}
            className="px-3 py-1.5 rounded-lg bg-emerald-500/10 text-emerald-400 hover:bg-emerald-500/20 disabled:opacity-50 text-xs font-medium"
          >
            {solving ? 'Solving...' : 'Show Solution'}
          </button>
        </div>

        {/* Hint / Solution panes */}
        {hints[currentQ.id] && (
          <div className="mb-4 rounded-lg border border-amber-500/20 bg-amber-500/10 p-3">
            <div className="text-xs font-medium text-amber-400 mb-1">Hint</div>
            <ul className="list-disc pl-4 text-xs text-amber-400/80 space-y-0.5">
              {hints[currentQ.id].steps?.map((s, i) => <li key={i}>{s}</li>)}
            </ul>
          </div>
        )}

        {solutions[currentQ.id] && (
          <div className="mb-4 rounded-lg border border-emerald-500/20 bg-emerald-500/10 p-3">
            <div className="text-xs font-medium text-emerald-400 mb-1">Solution</div>
            {solutions[currentQ.id].choice && (
              <div className="mb-1 text-xs text-emerald-400">Choice: <strong>{solutions[currentQ.id].choice}</strong></div>
            )}
            <div className="mb-1 text-xs text-emerald-400">
              Answer: <strong>{solutions[currentQ.id].final_answer}</strong> {solutions[currentQ.id].units ? `[${solutions[currentQ.id].units}]` : ''}
            </div>
            <ul className="list-disc pl-4 text-xs text-emerald-400/80 space-y-0.5">
              {solutions[currentQ.id].steps?.map((s, i) => <li key={i}>{s}</li>)}
            </ul>
          </div>
        )}

        {/* Navigation */}
        <div className="flex items-center justify-between">
          <button
            onClick={previousQuestion}
            disabled={examSession.currentQuestion === 0}
            className="px-4 py-2 bg-zinc-800 hover:bg-zinc-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg text-sm transition-colors text-zinc-300"
          >
            Previous
          </button>

          <div className="text-xs text-zinc-500">
            {Object.keys(examSession.userAnswers).length}/{examSession.questions.length} answered
          </div>

          {examSession.currentQuestion === examSession.questions.length - 1 ? (
            <button
              onClick={submitExam}
              className="bg-emerald-600 text-white px-4 py-2 rounded-lg hover:bg-emerald-500 text-sm font-medium flex items-center gap-2 transition-colors"
            >
              <CheckCircle className="w-4 h-4" />
              Submit
            </button>
          ) : (
            <button
              onClick={nextQuestion}
              className="bg-cyan-600 text-white px-4 py-2 rounded-lg hover:bg-cyan-500 text-sm font-medium transition-colors"
            >
              Next
            </button>
          )}
        </div>
      </div>

      {/* Question Overview */}
      <div className="bg-zinc-800/60 border border-zinc-700/40 rounded-xl p-5">
        <h3 className="text-sm font-medium text-zinc-100 mb-3">Question Overview</h3>
        <div className="grid grid-cols-10 gap-1.5">
          {examSession.questions.map((q, index) => {
            const isAnswered = !!examSession.userAnswers[q.id]
            const isCurrent = index === examSession.currentQuestion
            return (
              <button
                key={q.id}
                onClick={() => goToQuestion(index)}
                className={`w-8 h-8 rounded-lg text-xs font-bold transition-colors ${
                  isCurrent
                    ? 'bg-cyan-600 text-white'
                    : isAnswered
                    ? 'bg-emerald-500/10 text-emerald-400 hover:bg-emerald-500/20'
                    : 'bg-zinc-800 text-zinc-500 hover:bg-zinc-700'
                }`}
              >
                {index + 1}
              </button>
            )
          })}
        </div>
      </div>
    </div>
  )
}
