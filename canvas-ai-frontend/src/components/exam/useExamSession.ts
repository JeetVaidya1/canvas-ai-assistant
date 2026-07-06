// State machine for the Exam destination. Owns every piece of exam state —
// setup choices, the live session + countdown, resume-on-refresh persistence,
// hint/solve calls, and grading — so the screen components stay presentational.
//
// EXAM INTEGRITY — the following behaviors are load-bearing and must not drift:
//   * resume-on-refresh via localStorage (client view + savedAt → re-derived
//     remaining time), cleared on submit / new exam
//   * countdown semantics incl. the eslint-disable on the timer effect
//     (submitExam intentionally omitted from deps)
//   * auto-submit at 0:00, pause freezing the clock, answer persistence,
//     AI-judge grading flow, useInvalidateProgress after submit
import { useEffect, useRef, useState } from 'react'
import {
  generatePracticeExam,
  createExamSession,
  startExamSession,
  pauseExamSession,
  saveExamAnswer,
  navigateExamQuestion,
  submitExamSession as submitExamApi,
  solveExamQuestion,
  uploadPastPaper as uploadPastPaperApi,
} from '@/lib/api'
import { showError } from '@/lib/toast'
import { useInvalidateProgress } from '@/hooks/useInvalidateProgress'
import { errText, SAMPLE_EXAM } from './examMeta'
import type {
  ExamDifficulty,
  ExamQuestion,
  ExamResults,
  ExamSession,
  PastPaperAnalysis,
  RawExamQuestion,
  RawQuestionResult,
  SolveJSON,
} from './types'

export type ExamPhase = 'setup' | 'preStart' | 'live' | 'results'

export interface ExamController {
  phase: ExamPhase
  courseId: string

  // Setup
  examDifficulty: ExamDifficulty
  setExamDifficulty: (d: ExamDifficulty) => void
  examQuestionCount: number
  setExamQuestionCount: (n: number) => void
  loading: boolean
  genError: string | null
  generateExam: () => Promise<void>
  loadSample: () => void

  // Past paper
  uploading: boolean
  analysisSummary: PastPaperAnalysis | null
  uploadPaper: (file: File) => Promise<void>

  // Live session
  examSession: ExamSession | null
  timeRemaining: number
  currentAnswer: string
  setCurrentAnswer: (a: string) => void
  navDirection: number
  startExam: () => Promise<void>
  pauseExam: () => Promise<void>
  goToQuestion: (index: number) => Promise<void>
  nextQuestion: () => void
  previousQuestion: () => void
  submitExam: () => Promise<void>
  submitting: boolean
  abandonExam: () => void

  // Hints & solutions
  hinting: boolean
  solving: boolean
  hints: Record<string, SolveJSON>
  solutions: Record<string, SolveJSON>
  requestHint: () => Promise<void>
  requestSolution: () => Promise<void>

  // Results
  examResults: ExamResults | null
  resetExam: () => void
}

export function useExamSession(courseId: string, userId: string): ExamController {
  const [examSession, setExamSession] = useState<ExamSession | null>(null)
  const [examDifficulty, setExamDifficulty] = useState<ExamDifficulty>('mixed')
  const [examQuestionCount, setExamQuestionCount] = useState(12)
  const [loading, setLoading] = useState(false)
  const [genError, setGenError] = useState<string | null>(null)
  const [uploading, setUploading] = useState(false)
  const [analysisSummary, setAnalysisSummary] = useState<PastPaperAnalysis | null>(null)

  const [timeRemaining, setTimeRemaining] = useState(0)
  const [currentAnswer, setCurrentAnswer] = useState('')
  const [showResults, setShowResults] = useState(false)
  const [examResults, setExamResults] = useState<ExamResults | null>(null)

  const invalidateProgress = useInvalidateProgress(courseId)

  const [sessionId, setSessionId] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const submittingRef = useRef(false)

  const [hinting, setHinting] = useState(false)
  const [solving, setSolving] = useState(false)
  const [hints, setHints] = useState<Record<string, SolveJSON>>({})
  const [solutions, setSolutions] = useState<Record<string, SolveJSON>>({})

  // Direction of the last navigation, used to drive the question slide animation.
  const [navDirection, setNavDirection] = useState(1)

  // ---- Resume support -------------------------------------------------------
  // An in-progress exam survives a refresh / closed tab: we persist the client
  // view + a timestamp, and re-derive the remaining time from elapsed wall-clock
  // on restore. The server still holds the authoritative saved answers.
  const STORAGE_KEY = `vindexa_exam_active_${courseId}`
  const timeRemainingRef = useRef(timeRemaining)
  useEffect(() => { timeRemainingRef.current = timeRemaining }, [timeRemaining])

  const clearPersistedExam = () => {
    try { localStorage.removeItem(STORAGE_KEY) } catch { /* ignore */ }
  }

  // Restore an in-progress exam on mount.
  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY)
      if (!raw) return
      const saved = JSON.parse(raw)
      if (!saved?.session?.isActive) return
      const elapsed = Math.floor((Date.now() - (saved.savedAt ?? Date.now())) / 1000)
      const remaining = Math.max(0, (saved.timeRemaining ?? 0) - elapsed)
      const revived: ExamSession = {
        ...saved.session,
        startTime: saved.session.startTime ? new Date(saved.session.startTime) : undefined,
      }
      setExamSession(revived)
      setSessionId(saved.sessionId ?? null)
      setTimeRemaining(remaining)
      const cq = revived.questions?.[revived.currentQuestion]
      setCurrentAnswer(cq ? (revived.userAnswers?.[cq.id] ?? '') : '')
    } catch { /* ignore corrupt storage */ }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Persist on meaningful state changes (answers / navigation / start / pause).
  // Keyed on examSession (not the per-second tick) to avoid hammering storage;
  // the clock is reconstructed from savedAt on restore.
  useEffect(() => {
    if (!examSession?.isActive || showResults) return
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({
        sessionId,
        session: examSession,
        timeRemaining: timeRemainingRef.current,
        savedAt: Date.now(),
      }))
    } catch { /* storage full / disabled */ }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [examSession, sessionId, showResults])

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
    // eslint-disable-next-line react-hooks/exhaustive-deps -- submitExam is intentionally omitted: including it would recreate the countdown interval on every answer change and reset the timer mid-exam
  }, [examSession?.isActive, examSession?.isPaused, timeRemaining])

  const generateExam = async () => {
    if (!courseId) return
    clearPersistedExam()  // starting a new exam abandons any prior in-progress one
    setGenError(null)
    setTimeRemaining(0)
    setLoading(true)
    try {
      const result = await generatePracticeExam({
        courseId,
        examType: 'practice',
        difficulty: examDifficulty,
        questionCount: examQuestionCount,
        timeLimit: 120,
        userId: userId || 'anonymous',
        questionTypes: ['multiple_choice', 'calculation', 'short_answer'],
      })

      const examData = result.exam
      const rawQuestions = (examData.questions ?? []) as RawExamQuestion[]
      const mapped: ExamQuestion[] = rawQuestions.map((q) => ({
        id: q.id,
        type: q.type,
        question: q.question,
        options: q.options ?? undefined,
        points: q.points ?? 2,
        topic: q.topic ?? 'General',
        difficulty: q.difficulty ?? 'medium',
        answer: q.correct_answer ?? '',
        solution: q.explanation ?? '',
        timeEstimate: q.time_estimate ?? 3,
      }))

      const newSession: ExamSession = {
        id: examData.id,
        examName: examData.name,
        questions: mapped,
        timeLimit: examData.time_limit ?? 120,
        userAnswers: {},
        currentQuestion: 0,
        isActive: false,
        isPaused: false,
      }
      setExamSession(newSession)

      try {
        const created = await createExamSession(courseId, userId || 'anonymous', examData)
        if (created?.session?.id) setSessionId(created.session.id)
      } catch (e) {
        console.warn('create session failed (non-fatal):', e)
      }
    } catch (error) {
      console.error('Failed to generate exam:', error)
      setGenError(errText(error))
    } finally {
      setLoading(false)
    }
  }

  const loadSample = () => {
    setGenError(null)
    setExamSession(SAMPLE_EXAM)
  }

  const startExam = async () => {
    if (!examSession) return
    setExamSession({ ...examSession, isActive: true, startTime: new Date() })
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
      userAnswers: { ...examSession.userAnswers, [currentQ.id]: currentAnswer },
    })
    if (sessionId) {
      try { await saveExamAnswer(sessionId, currentQ.id, currentAnswer) } catch (e) { console.warn(e) }
    }
  }

  const goToQuestion = async (index: number) => {
    if (!examSession) return
    setNavDirection(index >= examSession.currentQuestion ? 1 : -1)
    await saveAnswer()
    setExamSession({ ...examSession, currentQuestion: index })
    setCurrentAnswer(examSession.userAnswers[examSession.questions[index].id] || '')
    if (sessionId) {
      try { await navigateExamQuestion(sessionId, index) } catch (e) { console.warn(e) }
    }
  }

  const nextQuestion = () => {
    if (!examSession) return
    setNavDirection(1)
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
    setNavDirection(-1)
    saveAnswer()
    if (examSession.currentQuestion > 0) {
      const prevIndex = examSession.currentQuestion - 1
      setExamSession({ ...examSession, currentQuestion: prevIndex })
      const prevQ = examSession.questions[prevIndex]
      setCurrentAnswer(examSession.userAnswers[prevQ.id] || '')
    }
  }

  const calculateResults = (session: ExamSession): ExamResults => {
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
        topic: q.topic,
      })),
    }
  }

  const submitExam = async () => {
    // The ref guards against a double submit when the countdown hits zero
    // while a manual submission is already in flight.
    if (!examSession || submittingRef.current) return
    submittingRef.current = true
    setSubmitting(true)
    try {
      await saveAnswer()
      const finalSession = { ...examSession, isActive: false, endTime: new Date() }
      setExamSession(finalSession)
      clearPersistedExam()

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
            letterGrade: results.letter_grade ?? null,
            topicPerformance: results.topic_performance ?? null,
            timeEfficiency: results.time_efficiency ?? null,
            timeSpent: results.time_metrics?.time_used_minutes ?? (finalSession.timeLimit - Math.floor(timeRemaining / 60)),
            breakdown: (results.question_results as RawQuestionResult[] | undefined)?.map((r) => ({
              question: r.question,
              userAnswer: r.user_answer,
              correctAnswer: r.correct_answer,
              points: r.points_possible,
              pointsEarned: r.points_earned,
              verdict: r.verdict,
              gradeReason: r.grade_reason,
              mistakeExplanation: r.mistake_explanation,
              timeSpent: r.time_spent,
              topic: r.topic,
            })) ?? [],
          })
          setShowResults(true)
          // Grading changed mastery server-side — refresh progress views.
          invalidateProgress()
          return
        } catch (e) {
          console.warn('Server submit failed; using local scoring.', e)
        }
      }

      const results = calculateResults(finalSession)
      setExamResults(results)
      setShowResults(true)
      // Saved answers may still have tracked activity server-side.
      invalidateProgress()
    } finally {
      submittingRef.current = false
      setSubmitting(false)
    }
  }

  const uploadPaper = async (file: File) => {
    if (!file || !courseId) return
    setUploading(true)
    try {
      const data = await uploadPastPaperApi(courseId, file, userId || 'anonymous')
      setAnalysisSummary(data as PastPaperAnalysis)
    } catch (e) {
      console.error(e)
      showError(`Upload failed: ${errText(e)}`)
    } finally {
      setUploading(false)
    }
  }

  const requestHint = async () => {
    if (!examSession || !courseId) return
    setHinting(true)
    try {
      const q = examSession.questions[examSession.currentQuestion]
      const res = await solveExamQuestion({ courseId, questionText: q.question, wantHint: true })
      const s: SolveJSON = res.solution
      setHints(prev => ({ ...prev, [q.id]: s }))
    } catch (e) {
      console.error(e)
      showError(`Hint failed: ${errText(e)}`)
    } finally {
      setHinting(false)
    }
  }

  const requestSolution = async () => {
    if (!examSession || !courseId) return
    setSolving(true)
    try {
      const q = examSession.questions[examSession.currentQuestion]
      const res = await solveExamQuestion({ courseId, questionText: q.question, wantHint: false })
      const s: SolveJSON = res.solution
      setSolutions(prev => ({ ...prev, [q.id]: s }))
    } catch (e) {
      console.error(e)
      showError(`Solve failed: ${errText(e)}`)
    } finally {
      setSolving(false)
    }
  }

  const resetExam = () => {
    setShowResults(false)
    setExamSession(null)
    setExamResults(null)
    setSessionId(null)
    setHints({})
    setSolutions({})
    setTimeRemaining(0)
    setGenError(null)
  }

  const abandonExam = () => {
    clearPersistedExam()
    setExamSession(null)
    setSessionId(null)
    setTimeRemaining(0)
  }

  const phase: ExamPhase =
    showResults && examResults ? 'results'
    : !examSession ? 'setup'
    : !examSession.isActive ? 'preStart'
    : 'live'

  return {
    phase,
    courseId,
    examDifficulty,
    setExamDifficulty,
    examQuestionCount,
    setExamQuestionCount,
    loading,
    genError,
    generateExam,
    loadSample,
    uploading,
    analysisSummary,
    uploadPaper,
    examSession,
    timeRemaining,
    currentAnswer,
    setCurrentAnswer,
    navDirection,
    startExam,
    pauseExam,
    goToQuestion,
    nextQuestion,
    previousQuestion,
    submitExam,
    submitting,
    abandonExam,
    hinting,
    solving,
    hints,
    solutions,
    requestHint,
    requestSolution,
    examResults,
    resetExam,
  }
}
