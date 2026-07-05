import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'motion/react'
import { Markdown } from '@/components/ui/Markdown'
import { Button } from '@/components/ui/Button'
import { Card } from '@/components/ui/Card'
import {
  Upload,
  Timer,
  Play,
  Pause,
  RotateCcw,
  CheckCircle,
  XCircle,
  CircleDot,
  Eye,
  Lightbulb,
  ChevronLeft,
  ChevronRight,
  FileText,
  Flag,
  Target,
  Trophy,
  LayoutGrid,
  X,
  GraduationCap,
} from 'lucide-react'
import { BrandMark } from '@/components/ui/BrandMark'

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
import { useInvalidateProgress } from '@/hooks/useInvalidateProgress'

/** Human-readable message from an unknown thrown value. */
function errText(e: unknown): string {
  return e instanceof Error ? e.message : String(e)
}

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

type Verdict = 'correct' | 'partial' | 'incorrect'

interface BreakdownItem {
  question: string
  userAnswer?: string
  correctAnswer?: string
  points?: number
  pointsEarned?: number
  verdict?: string
  gradeReason?: string
  mistakeExplanation?: string
  timeSpent?: number
  topic?: string
}

interface ExamResults {
  totalQuestions: number
  correctAnswers: number
  totalPoints: number
  earnedPoints: number
  percentage: number
  letterGrade?: string | null
  topicPerformance?: Record<string, { earned?: number; possible?: number; percentage?: number }> | null
  timeEfficiency?: string | null
  timeSpent: number
  breakdown: BreakdownItem[]
}

/** Shape of a generated exam question as returned by the backend. */
interface RawExamQuestion {
  id: string
  type: QType
  question: string
  options?: string[]
  points?: number
  topic?: string
  difficulty?: Diff
  correct_answer?: string
  explanation?: string
  time_estimate?: number
}

/** Shape of a graded question result as returned by the backend. */
interface RawQuestionResult {
  question: string
  user_answer?: string
  correct_answer?: string
  points_possible?: number
  points_earned?: number
  verdict?: string
  grade_reason?: string
  mistake_explanation?: string
  time_spent?: number
  topic?: string
}

/** Past-paper upload analysis summary from the backend. */
interface PastPaperAnalysis {
  status?: string
  questions_found?: number
  message?: string
}

// Centered setup choices — tactile, mirrors QuizMode's center-first flow.
const DIFFICULTIES: { value: 'easy' | 'medium' | 'hard' | 'mixed'; label: string; hint: string }[] = [
  { value: 'easy', label: 'Easy', hint: 'Warm-up' },
  { value: 'medium', label: 'Medium', hint: 'Balanced' },
  { value: 'hard', label: 'Hard', hint: 'Exam-grade' },
  { value: 'mixed', label: 'Mixed', hint: 'Like the real thing' },
]
const COUNTS = [5, 8, 12, 15] as const

export default function ExamMode({ courseId, userId }: ExamModeProps) {
  const [examSession, setExamSession] = useState<ExamSession | null>(null)
  const [examDifficulty, setExamDifficulty] = useState<'easy' | 'medium' | 'hard' | 'mixed'>('mixed')
  const [examQuestionCount, setExamQuestionCount] = useState(12)
  const [loading, setLoading] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [analysisSummary, setAnalysisSummary] = useState<PastPaperAnalysis | null>(null)

  const [timeRemaining, setTimeRemaining] = useState(0)
  const [currentAnswer, setCurrentAnswer] = useState('')
  const [showResults, setShowResults] = useState(false)
  const [examResults, setExamResults] = useState<ExamResults | null>(null)

  const invalidateProgress = useInvalidateProgress(courseId)

  const [sessionId, setSessionId] = useState<string | null>(null)

  const [hinting, setHinting] = useState(false)
  const [solving, setSolving] = useState(false)
  const [hints, setHints] = useState<Record<string, SolveJSON>>({})
  const [solutions, setSolutions] = useState<Record<string, SolveJSON>>({})

  // Past-paper upload is now a secondary, opt-in section on the setup screen.
  const [showUpload, setShowUpload] = useState(false)

  // Question navigator now lives in a slide-over panel rather than a competing card.
  const [navOpen, setNavOpen] = useState(false)

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

  const formatTime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    const secs = seconds % 60
    if (hours > 0) return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
    return `${minutes}:${secs.toString().padStart(2, '0')}`
  }

  const generateExamFromPastPaper = async () => {
    if (!courseId) return
    clearPersistedExam()  // starting a new exam abandons any prior in-progress one
    setLoading(true)
    try {
      const result = await generatePracticeExam({
        courseId,
        examType: 'practice',
        difficulty: examDifficulty,
        questionCount: examQuestionCount,
        timeLimit: 120,
        userId: userId || 'anonymous',
        questionTypes: ['multiple_choice','calculation','short_answer']
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
    } catch (error) {
      console.error('Failed to generate exam:', error)
      showError(`Could not generate exam: ${errText(error)}`)
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

  const submitExam = async () => {
    if (!examSession) return
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
            topic: r.topic
          })) ?? []
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
      const res = await solveExamQuestion({
        courseId,
        questionText: q.question,
        wantHint: true
      })
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
      const res = await solveExamQuestion({
        courseId,
        questionText: q.question,
        wantHint: false
      })
      const s: SolveJSON = res.solution
      setSolutions(prev => ({ ...prev, [q.id]: s }))
    } catch (e) {
      console.error(e)
      showError(`Solve failed: ${errText(e)}`)
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
        solution: 'Third harmonic: L = 3λ/2 → λ = 2L/3 = 0.24 m', timeEstimate: 5
      },
      {
        id: '3', type: 'short_answer',
        question: 'Three point charges are located on the x-axis. Calculate the magnitude of the electric force on the middle charge.',
        points: 4, topic: 'Electrostatics', difficulty: 'hard', timeEstimate: 8
      }
    ]
  }

  // ---- Presentation helpers -------------------------------------------------

  const verdictMeta = (v: Verdict) => {
    switch (v) {
      case 'correct':
        return {
          label: 'Correct',
          tone: 'border-emerald-500/25 bg-emerald-500/[0.06]',
          text: 'text-emerald-400',
          chip: 'bg-emerald-500/15 text-emerald-300 border border-emerald-500/25',
          Icon: CheckCircle,
        }
      case 'partial':
        return {
          label: 'Partial',
          tone: 'border-amber-500/25 bg-amber-500/[0.06]',
          text: 'text-amber-400',
          chip: 'bg-amber-500/15 text-amber-300 border border-amber-500/25',
          Icon: CircleDot,
        }
      default:
        return {
          label: 'Incorrect',
          tone: 'border-rose-500/25 bg-rose-500/[0.06]',
          text: 'text-rose-400',
          chip: 'bg-rose-500/15 text-rose-300 border border-rose-500/25',
          Icon: XCircle,
        }
    }
  }

  const readinessLabel = (pct: number): string => {
    if (pct >= 85) return 'Exam ready'
    if (pct >= 70) return 'Nearly ready'
    if (pct >= 50) return 'Getting there'
    return 'Needs work'
  }

  // ---- Results screen -------------------------------------------------------
  if (showResults && examResults) {
    const r = examResults
    const breakdown: BreakdownItem[] = Array.isArray(r.breakdown) ? r.breakdown : []
    const verdictOf = (b: BreakdownItem): Verdict =>
      (b.verdict as Verdict) ??
      ((b.pointsEarned ?? 0) >= (b.points ?? 0) ? 'correct' : (b.pointsEarned ?? 0) > 0 ? 'partial' : 'incorrect')

    const tallies = breakdown.reduce(
      (acc, b) => { acc[verdictOf(b)]++; return acc },
      { correct: 0, partial: 0, incorrect: 0 } as Record<Verdict, number>
    )

    // Per-topic performance: prefer server payload, else derive from breakdown.
    const topicRows: Array<{ topic: string; earned: number; possible: number; pct: number }> = (() => {
      if (r.topicPerformance && typeof r.topicPerformance === 'object') {
        return Object.entries(r.topicPerformance).map(([topic, v]) => {
          const earned = v?.earned ?? 0
          const possible = v?.possible ?? 0
          const pct = v?.percentage ?? (possible > 0 ? Math.round((earned / possible) * 100) : 0)
          return { topic, earned, possible, pct }
        })
      }
      const acc: Record<string, { earned: number; possible: number }> = {}
      breakdown.forEach((b) => {
        const t = b.topic ?? 'General'
        if (!acc[t]) acc[t] = { earned: 0, possible: 0 }
        acc[t].earned += b.pointsEarned ?? 0
        acc[t].possible += b.points ?? 0
      })
      return Object.entries(acc).map(([topic, v]) => ({
        topic,
        earned: v.earned,
        possible: v.possible,
        pct: v.possible > 0 ? Math.round((v.earned / v.possible) * 100) : 0,
      }))
    })().sort((a, b) => a.pct - b.pct)

    const ringPct = Math.max(0, Math.min(100, r.percentage))

    return (
      <div className="max-w-3xl mx-auto px-5 py-8 space-y-6">
        {/* Graded-report hero — big score ring, verdict, letter grade, tally */}
        <motion.div
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
        >
          <div className="mb-6 text-center">
            <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-gradient-brand mb-1.5">
              Your graded report
            </p>
            <h1 className="text-[28px] font-semibold tracking-tight text-zinc-50">{readinessLabel(r.percentage)}</h1>
          </div>

          <Card accent padding="lg">
            <div className="flex flex-col sm:flex-row items-center gap-7">
              {/* Big radial score */}
              <motion.div
                initial={{ scale: 0.85, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ delay: 0.1, type: 'spring', stiffness: 180, damping: 16 }}
                className="relative flex-shrink-0"
              >
                <svg width="156" height="156" viewBox="0 0 156 156" className="-rotate-90">
                  <circle cx="78" cy="78" r="68" fill="none" stroke="rgb(39 39 42)" strokeWidth="10" />
                  <defs>
                    <linearGradient id="examReadiness" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stopColor="#06b6d4" />
                      <stop offset="100%" stopColor="#3b82f6" />
                    </linearGradient>
                  </defs>
                  <motion.circle
                    cx="78" cy="78" r="68" fill="none"
                    stroke="url(#examReadiness)" strokeWidth="10" strokeLinecap="round"
                    strokeDasharray={2 * Math.PI * 68}
                    initial={{ strokeDashoffset: 2 * Math.PI * 68 }}
                    animate={{ strokeDashoffset: 2 * Math.PI * 68 * (1 - ringPct / 100) }}
                    transition={{ duration: 1, ease: 'easeOut', delay: 0.2 }}
                  />
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                  <span className="text-4xl font-bold tracking-tight text-gradient-brand">{r.percentage}%</span>
                  <span className="text-[10px] uppercase tracking-widest text-zinc-500 mt-0.5">Readiness</span>
                </div>
              </motion.div>

              <div className="flex-1 text-center sm:text-left">
                <div className="flex items-center justify-center sm:justify-start gap-2.5 mb-2">
                  <Trophy className="w-4 h-4 text-cyan-300" />
                  <span className="text-sm font-semibold text-zinc-100">{readinessLabel(r.percentage)}</span>
                  {r.letterGrade && (
                    <span className="text-lg font-bold text-zinc-100 px-2.5 py-0.5 rounded-lg bg-gradient-brand-soft border border-cyan-400/20">
                      {r.letterGrade}
                    </span>
                  )}
                </div>
                <p className="text-sm text-zinc-400">
                  {r.correctAnswers} of {r.totalQuestions} correct &middot; {r.earnedPoints}/{r.totalPoints} points
                </p>
                {typeof r.timeEfficiency === 'string' && (
                  <p className="text-xs text-zinc-500 mt-1">{r.timeEfficiency}</p>
                )}
                {/* Verdict tally bar */}
                <div className="mt-4 flex flex-wrap items-center gap-2 justify-center sm:justify-start">
                  {(['correct', 'partial', 'incorrect'] as Verdict[]).map((v) => {
                    const m = verdictMeta(v)
                    return (
                      <span key={v} className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium ${m.chip}`}>
                        <m.Icon className="w-3.5 h-3.5" /> {tallies[v]} {m.label.toLowerCase()}
                      </span>
                    )
                  })}
                </div>
              </div>
            </div>
          </Card>
        </motion.div>

        {/* Quick stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {[
            { value: `${r.percentage}%`, label: 'Score', cls: 'text-gradient-brand' },
            { value: `${r.correctAnswers}/${r.totalQuestions}`, label: 'Correct', cls: 'text-emerald-400' },
            { value: `${r.earnedPoints}/${r.totalPoints}`, label: 'Points', cls: 'text-zinc-100' },
            { value: `${r.timeSpent}m`, label: 'Time', cls: 'text-amber-400' },
          ].map((s, i) => (
            <motion.div
              key={s.label}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: 0.05 * i }}
            >
              <Card padding="sm" className="text-center">
                <div className={`text-2xl font-bold mb-0.5 ${s.cls}`}>{s.value}</div>
                <div className="text-xs text-zinc-500">{s.label}</div>
              </Card>
            </motion.div>
          ))}
        </div>

        {/* Per-topic performance */}
        {topicRows.length > 0 && (
          <Card padding="md">
            <div className="flex items-center gap-2 mb-3">
              <Target className="w-4 h-4 text-cyan-300" />
              <h3 className="text-sm font-semibold text-zinc-100">Performance by topic</h3>
            </div>
            <div className="space-y-3">
              {topicRows.map((t, i) => {
                const barTone = t.pct >= 70 ? 'bg-emerald-500' : t.pct >= 45 ? 'bg-amber-500' : 'bg-rose-500'
                return (
                  <div key={t.topic}>
                    <div className="flex items-center justify-between mb-1.5 text-xs">
                      <span className="text-zinc-300 font-medium">{t.topic}</span>
                      <span className="text-zinc-500 tabular-nums">
                        {t.earned}/{t.possible} pts &middot; <span className="text-zinc-300">{t.pct}%</span>
                      </span>
                    </div>
                    <div className="w-full bg-zinc-800 rounded-full h-1.5 overflow-hidden">
                      <motion.div
                        className={`h-1.5 rounded-full ${barTone}`}
                        initial={{ width: 0 }}
                        animate={{ width: `${t.pct}%` }}
                        transition={{ duration: 0.7, delay: 0.1 + 0.05 * i, ease: 'easeOut' }}
                      />
                    </div>
                  </div>
                )
              })}
            </div>
          </Card>
        )}

        {/* Per-question AI-judge verdicts */}
        {breakdown.length > 0 && (
          <div className="space-y-2.5">
            <h3 className="text-sm font-semibold text-zinc-100 px-1">Question-by-question review</h3>
            {breakdown.map((b, i) => {
              const v = verdictOf(b)
              const m = verdictMeta(v)
              return (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.25, delay: 0.02 * i }}
                  className={`border rounded-xl p-4 ${m.tone}`}
                >
                  <div className="flex items-start justify-between gap-3 mb-2">
                    <span className="text-xs text-zinc-400">Q{i + 1} &middot; {b.topic ?? 'General'}</span>
                    <span className={`inline-flex items-center gap-1.5 text-xs font-semibold ${m.text}`}>
                      <m.Icon className="w-3.5 h-3.5" />
                      {m.label} &middot; {b.pointsEarned ?? 0}/{b.points ?? 0} pts
                      {typeof b.timeSpent === 'number' ? ` · ${b.timeSpent}s` : ''}
                    </span>
                  </div>
                  <p className="text-sm text-zinc-100 mb-1.5 leading-relaxed">{b.question}</p>
                  {b.userAnswer && (
                    <p className="text-xs text-zinc-400 mb-1">
                      <span className="text-zinc-500">Your answer: </span>{b.userAnswer}
                    </p>
                  )}
                  {b.gradeReason && <p className="text-xs text-zinc-400 italic">{b.gradeReason}</p>}
                  {b.mistakeExplanation && (
                    <div className="mt-2 flex items-start gap-1.5 rounded-lg bg-amber-500/[0.08] border border-amber-500/15 p-2.5">
                      <Lightbulb className="w-3.5 h-3.5 text-amber-400 flex-shrink-0 mt-0.5" />
                      <p className="text-xs text-amber-300/90">
                        <span className="font-medium text-amber-300">Where it went wrong: </span>{b.mistakeExplanation}
                      </p>
                    </div>
                  )}
                </motion.div>
              )
            })}
          </div>
        )}

        <div className="flex gap-3 pt-1">
          <Button
            size="lg"
            onClick={() => {
              setShowResults(false)
              setExamSession(null)
              setExamResults(null)
              setSessionId(null)
              setHints({})
              setSolutions({})
            }}
            leftIcon={<RotateCcw className="w-4 h-4" />}
          >
            New Exam
          </Button>
        </div>
      </div>
    )
  }

  // ---- Setup — center-first, focused "Set up your mock exam" ----------------
  if (!examSession) {
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
              Set up your mock exam
            </h1>
            <p className="mx-auto mt-2 max-w-md text-sm text-zinc-400">
              A timed simulation built from your materials. Answers and the clock auto-save — close the tab and pick up
              exactly where you left off. AI judges with partial credit and grounded explanations.
            </p>
          </div>

          {/* Difficulty — 4 big selectable tiles */}
          <div className="mb-6">
            <div className="mb-2.5 flex items-center justify-center text-[11px] font-semibold uppercase tracking-[0.16em] text-zinc-500">
              Difficulty
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
              {DIFFICULTIES.map((d) => {
                const active = examDifficulty === d.value
                return (
                  <button
                    key={d.value}
                    onClick={() => setExamDifficulty(d.value)}
                    className={`group rounded-2xl border px-3 py-4 text-center transition-all ${
                      active
                        ? 'border-cyan-400/50 bg-gradient-brand-soft ring-2 ring-cyan-400/25 shadow-[0_8px_24px_-12px_rgba(34,211,238,0.5)]'
                        : 'border-white/10 bg-white/[0.03] hover:border-cyan-400/30 hover:bg-white/[0.05]'
                    }`}
                  >
                    <div className={`text-sm font-semibold ${active ? 'text-cyan-200' : 'text-zinc-200'}`}>
                      {d.label}
                    </div>
                    <div className="mt-0.5 text-[10px] text-zinc-500 leading-tight">{d.hint}</div>
                  </button>
                )
              })}
            </div>
          </div>

          {/* Question count — segmented control */}
          <div className="mb-8">
            <div className="mb-2.5 flex items-center justify-center text-[11px] font-semibold uppercase tracking-[0.16em] text-zinc-500">
              Questions
            </div>
            <div className="flex gap-1.5 rounded-2xl border border-white/10 bg-white/[0.03] p-1.5">
              {COUNTS.map((c) => {
                const active = examQuestionCount === c
                return (
                  <button
                    key={c}
                    onClick={() => setExamQuestionCount(c)}
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

          {/* Prominent CTA */}
          <Button
            onClick={generateExamFromPastPaper}
            disabled={loading || !courseId}
            loading={loading}
            size="lg"
            leftIcon={<Timer className="h-4 w-4" />}
            className="w-full !py-3.5 !text-base"
          >
            {loading ? 'Generating your exam…' : 'Start exam'}
          </Button>
          {loading && (
            <p className="mt-3 text-center text-xs text-zinc-500">
              Retrieving from your materials and writing exam questions — this can take a moment.
            </p>
          )}

          {/* Secondary options — past paper + sample, subtle */}
          <div className="mt-6 flex items-center justify-center gap-4 text-xs">
            <button
              onClick={() => setShowUpload((v) => !v)}
              className="text-zinc-400 transition-colors hover:text-cyan-300"
            >
              or solve a past paper
            </button>
            <span className="text-zinc-700">·</span>
            <button
              onClick={() => setExamSession(sampleExam)}
              className="text-zinc-400 transition-colors hover:text-cyan-300"
            >
              try a sample exam
            </button>
          </div>

          {/* Past-paper upload — revealed on demand, doesn't clutter the main flow */}
          <AnimatePresence>
            {showUpload && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.25 }}
                className="overflow-hidden"
              >
                <div className="mt-5 rounded-2xl border border-white/10 bg-white/[0.03] p-5 text-center">
                  <div className="mx-auto mb-3 flex h-9 w-9 items-center justify-center rounded-xl bg-white/[0.04] border border-white/10">
                    <FileText className="h-4.5 w-4.5 text-zinc-300" />
                  </div>
                  <h3 className="text-sm font-semibold text-zinc-100 mb-1">Upload a past paper</h3>
                  <p className="text-xs text-zinc-400 mb-4">We'll analyze it to generate similar practice questions.</p>
                  <label className={`inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium cursor-pointer transition-all border ${
                    courseId
                      ? 'bg-white/[0.04] border-white/10 text-zinc-200 hover:bg-white/[0.08] hover:border-white/20'
                      : 'bg-white/[0.04] border-white/10 text-zinc-500 cursor-not-allowed'
                  }`}>
                    <Upload className="w-4 h-4" />
                    {uploading ? 'Uploading…' : 'Choose file'}
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
                    <div className="text-xs text-zinc-500 mt-3">
                      {analysisSummary.status === 'success' ? (
                        <span className="inline-flex items-center gap-1.5 text-emerald-400">
                          <CheckCircle className="w-3.5 h-3.5" />
                          {analysisSummary.questions_found} questions found
                        </span>
                      ) : (
                        <span className="text-rose-400">{analysisSummary.message ?? 'Upload failed'}</span>
                      )}
                    </div>
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      </div>
    )
  }

  // ---- Pre-start — confirmation / resume ------------------------------------
  if (!examSession.isActive) {
    const totalPoints = examSession.questions.reduce((a, q) => a + (q.points ?? 0), 0)
    const answeredCount = Object.keys(examSession.userAnswers).length
    const isResume = answeredCount > 0 || timeRemaining > 0
    return (
      <div className="flex min-h-full flex-col items-center justify-center px-4 py-10">
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
          className="w-full max-w-md"
        >
          <div className="mb-7 text-center">
            <BrandMark className="mx-auto mb-5 h-14 w-14 glow-brand-sm" />
            <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-gradient-brand mb-1.5">
              {isResume ? 'Resume your exam' : 'Ready when you are'}
            </p>
            <h1 className="text-2xl font-semibold tracking-tight text-zinc-50 truncate">{examSession.examName}</h1>
            {isResume && (
              <p className="text-xs text-amber-400/90 mt-2">
                In progress — {answeredCount}/{examSession.questions.length} answered, clock resumes where you left off.
              </p>
            )}
          </div>

          <div className="grid grid-cols-3 gap-3 mb-7">
            {[
              { label: 'Questions', value: examSession.questions.length },
              { label: 'Points', value: totalPoints },
              { label: 'Time', value: `${examSession.timeLimit}m` },
            ].map((s) => (
              <div key={s.label} className="rounded-2xl border border-white/10 bg-white/[0.03] px-3 py-4 text-center">
                <div className="text-xl font-bold text-zinc-100">{s.value}</div>
                <div className="text-[11px] text-zinc-400 mt-0.5">{s.label}</div>
              </div>
            ))}
          </div>

          <Button
            size="lg"
            onClick={startExam}
            leftIcon={<Play className="w-4 h-4" />}
            className="w-full !py-3.5 !text-base"
          >
            {isResume ? 'Resume exam' : 'Begin exam'}
          </Button>
          <button
            onClick={() => { clearPersistedExam(); setExamSession(null); setSessionId(null) }}
            className="mt-4 w-full text-center text-xs text-zinc-500 transition-colors hover:text-zinc-300"
          >
            ← Back to setup
          </button>
        </motion.div>
      </div>
    )
  }

  // ---- Active exam ----------------------------------------------------------
  const currentQ = examSession.questions[examSession.currentQuestion]
  const progress = ((examSession.currentQuestion + 1) / examSession.questions.length) * 100
  const answeredCount = Object.keys(examSession.userAnswers).filter((k) => examSession.userAnswers[k]).length
  const timeLow = timeRemaining < 300
  const timeCritical = timeRemaining < 60
  const timerTone = timeCritical
    ? 'border-rose-500/40 bg-rose-500/10 text-rose-300'
    : timeLow
    ? 'border-amber-500/40 bg-amber-500/10 text-amber-300'
    : 'border-cyan-400/25 bg-gradient-brand-soft text-cyan-200'
  const isLast = examSession.currentQuestion === examSession.questions.length - 1

  const slideVariants = {
    enter: (dir: number) => ({ opacity: 0, x: dir > 0 ? 28 : -28 }),
    center: { opacity: 1, x: 0 },
    exit: (dir: number) => ({ opacity: 0, x: dir > 0 ? -28 : 28 }),
  }

  return (
    <div className="relative flex min-h-full flex-col">
      {/* Slim sticky top bar: title + counter · timer + Questions + Pause, thin progress under */}
      <div className="sticky top-0 z-20 glass-bar">
        <div className="mx-auto flex max-w-3xl items-center justify-between gap-4 px-5 py-2.5">
          <div className="min-w-0">
            <h2 className="text-sm font-semibold text-zinc-100 truncate">{examSession.examName}</h2>
            <p className="text-[11px] text-zinc-400 mt-0.5">
              Question {examSession.currentQuestion + 1} of {examSession.questions.length} &middot; {answeredCount} answered
            </p>
          </div>
          <div className="flex items-center gap-2 flex-shrink-0">
            <motion.div
              animate={timeLow && !examSession.isPaused ? { scale: [1, 1.04, 1] } : { scale: 1 }}
              transition={timeLow ? { repeat: Infinity, duration: 1.4 } : { duration: 0.2 }}
              className={`flex items-center gap-2 px-3.5 py-2 rounded-xl border tabular-nums ${timerTone}`}
            >
              <Timer className="w-4 h-4" />
              <span className="text-lg font-semibold tracking-tight">{formatTime(timeRemaining)}</span>
            </motion.div>
            <button
              onClick={() => setNavOpen(true)}
              className="inline-flex items-center gap-1.5 rounded-lg border border-white/10 bg-white/[0.04] px-2.5 py-2 text-[13px] text-zinc-300 transition-colors hover:bg-white/[0.08] hover:text-zinc-100"
              aria-label="Open question navigator"
            >
              <LayoutGrid className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">Questions</span>
            </button>
            <Button
              variant="secondary"
              size="sm"
              onClick={pauseExam}
              leftIcon={examSession.isPaused ? <Play className="w-3.5 h-3.5" /> : <Pause className="w-3.5 h-3.5" />}
            >
              {examSession.isPaused ? 'Resume' : 'Pause'}
            </Button>
          </div>
        </div>
        {/* Thin progress bar */}
        <div className="h-0.5 w-full bg-white/[0.06]">
          <motion.div
            className="h-0.5 bg-gradient-brand"
            animate={{ width: `${progress}%` }}
            transition={{ duration: 0.4, ease: 'easeOut' }}
          />
        </div>
      </div>

      {/* Focused reading column */}
      <div className="mx-auto w-full max-w-2xl flex-1 px-5 py-7">
        {/* Paused overlay banner */}
        <AnimatePresence>
          {examSession.isPaused && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="mb-5 overflow-hidden"
            >
              <div className="flex items-center gap-2.5 rounded-xl border border-amber-500/25 bg-amber-500/[0.08] px-4 py-3">
                <Pause className="w-4 h-4 text-amber-400 flex-shrink-0" />
                <span className="text-sm text-amber-300">Exam paused — the timer is frozen. Resume when you're ready.</span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Distraction-free question view with slide transition */}
        <div className="relative" style={{ minHeight: 320 }}>
          <AnimatePresence mode="wait" custom={navDirection}>
            <motion.div
              key={currentQ.id}
              custom={navDirection}
              variants={slideVariants}
              initial="enter"
              animate="center"
              exit="exit"
              transition={{ duration: 0.28, ease: 'easeOut' }}
            >
              {/* Meta chips — small + subtle */}
              <div className="flex items-center gap-2 mb-5">
                <span className="inline-flex items-center px-2 py-0.5 rounded-md text-xs font-semibold bg-gradient-brand-soft border border-cyan-400/20 text-cyan-200">
                  {currentQ.points} pt{currentQ.points !== 1 ? 's' : ''}
                </span>
                <span className="inline-flex items-center px-2 py-0.5 rounded-md text-xs font-medium border border-white/10 bg-white/[0.03] text-zinc-400">
                  {currentQ.topic}
                </span>
                <span className="inline-flex items-center px-2 py-0.5 rounded-md text-xs font-medium border border-white/10 bg-white/[0.03] text-zinc-400 capitalize">
                  {currentQ.difficulty}
                </span>
              </div>

              {/* Large readable question text */}
              <div className="text-lg sm:text-xl font-medium text-zinc-100 leading-relaxed mb-7">
                <Markdown content={currentQ.question} />
              </div>

              {/* Answer area — prominent */}
              {currentQ.type === 'multiple_choice' && currentQ.options ? (
                <div className="space-y-2.5 mb-6">
                  {currentQ.options.map((option, index) => {
                    const letter = String.fromCharCode(65 + index)
                    const isSelected = currentAnswer === letter
                    return (
                      <motion.button
                        key={index}
                        whileTap={{ scale: 0.99 }}
                        onClick={() => setCurrentAnswer(letter)}
                        className={`w-full p-4 border rounded-xl text-left transition-all text-[15px] ${
                          isSelected
                            ? 'border-cyan-400/50 bg-gradient-brand-soft ring-2 ring-cyan-400/30 glow-brand-sm'
                            : 'border-white/10 bg-white/[0.02] hover:border-cyan-400/40 hover:bg-cyan-400/[0.05]'
                        }`}
                      >
                        <div className="flex items-center gap-3.5">
                          <div className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0 transition-colors ${
                            isSelected ? 'bg-gradient-brand text-white' : 'bg-white/[0.06] text-zinc-300'
                          }`}>
                            {letter}
                          </div>
                          <span className="flex-1 text-zinc-100">{option}</span>
                        </div>
                      </motion.button>
                    )
                  })}
                </div>
              ) : (
                <div className="mb-6">
                  <textarea
                    value={currentAnswer}
                    onChange={(e) => setCurrentAnswer(e.target.value)}
                    placeholder="Write your answer here…"
                    className="w-full h-40 p-4 bg-white/[0.03] border border-white/10 text-zinc-100 rounded-xl focus:border-cyan-400/60 focus:ring-2 focus:ring-cyan-500/25 outline-none resize-none placeholder-zinc-500 text-[15px] leading-relaxed transition-colors"
                  />
                </div>
              )}

              {/* Hint / Solution — subtle secondary actions */}
              <div className="flex items-center gap-3 mb-5">
                <button
                  onClick={requestHint}
                  disabled={hinting || !courseId}
                  className="inline-flex items-center gap-1.5 text-xs font-medium text-zinc-400 hover:text-amber-300 disabled:opacity-50 transition-colors"
                >
                  <Lightbulb className="w-3.5 h-3.5" />
                  {hinting ? 'Getting hint…' : 'Get hint'}
                </button>
                <span className="text-zinc-700">·</span>
                <button
                  onClick={requestSolution}
                  disabled={solving || !courseId}
                  className="inline-flex items-center gap-1.5 text-xs font-medium text-zinc-400 hover:text-emerald-300 disabled:opacity-50 transition-colors"
                >
                  <Eye className="w-3.5 h-3.5" />
                  {solving ? 'Solving…' : 'Show solution'}
                </button>
              </div>

              {/* Hint / Solution panes */}
              <AnimatePresence>
                {hints[currentQ.id] && (
                  <motion.div
                    initial={{ opacity: 0, y: -6 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                    className="mb-4 rounded-xl border border-amber-500/20 bg-amber-500/10 p-3.5"
                  >
                    <div className="text-xs font-medium text-amber-400 mb-1">Hint</div>
                    <ul className="list-disc pl-4 text-xs text-amber-400/80 space-y-0.5">
                      {hints[currentQ.id].steps?.map((s, i) => <li key={i}>{s}</li>)}
                    </ul>
                  </motion.div>
                )}

                {solutions[currentQ.id] && (
                  <motion.div
                    initial={{ opacity: 0, y: -6 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                    className="mb-4 rounded-xl border border-emerald-500/20 bg-emerald-500/10 p-3.5"
                  >
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
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          </AnimatePresence>
        </div>

        {/* Bottom navigation — Previous / Next, Submit prominent only on the last question */}
        <div className="mt-6 flex items-center justify-between border-t border-white/10 pt-5">
          <Button
            variant="secondary"
            onClick={previousQuestion}
            disabled={examSession.currentQuestion === 0}
            leftIcon={<ChevronLeft className="w-4 h-4" />}
          >
            Previous
          </Button>

          <div className="flex items-center gap-2">
            {!isLast && (
              <button
                onClick={submitExam}
                className="text-xs text-zinc-500 transition-colors hover:text-zinc-300"
              >
                Finish early
              </button>
            )}
            {isLast ? (
              <Button
                onClick={submitExam}
                className="!bg-emerald-600 hover:!bg-emerald-500 !glow-brand-sm"
                leftIcon={<Flag className="w-4 h-4" />}
              >
                Submit exam
              </Button>
            ) : (
              <Button
                onClick={nextQuestion}
                rightIcon={<ChevronRight className="w-4 h-4" />}
              >
                Next
              </Button>
            )}
          </div>
        </div>
      </div>

      {/* Question navigator — slide-over panel */}
      <AnimatePresence>
        {navOpen && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setNavOpen(false)}
              className="fixed inset-0 z-30 bg-black/50 backdrop-blur-sm"
            />
            <motion.aside
              initial={{ x: 360 }}
              animate={{ x: 0 }}
              exit={{ x: 360 }}
              transition={{ type: 'spring', stiffness: 320, damping: 34 }}
              className="fixed inset-y-0 right-0 z-40 flex w-[340px] max-w-[88vw] flex-col border-l border-white/10 bg-[#0c0f18]"
            >
              <div className="flex h-14 items-center justify-between px-4 border-b border-white/10">
                <div className="flex items-center gap-2">
                  <LayoutGrid className="h-4 w-4 text-cyan-300" />
                  <span className="text-sm font-semibold text-zinc-100">Questions</span>
                </div>
                <button
                  onClick={() => setNavOpen(false)}
                  className="rounded-lg p-1.5 text-zinc-400 transition-colors hover:bg-white/[0.06] hover:text-zinc-100"
                  aria-label="Close navigator"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>

              {/* Legend */}
              <div className="flex items-center gap-3 px-4 py-3 text-[11px] text-zinc-400 border-b border-white/10">
                <span className="inline-flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-sm bg-gradient-brand" /> Current
                </span>
                <span className="inline-flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-sm bg-emerald-500/40 border border-emerald-500/40" /> Answered
                </span>
                <span className="inline-flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-sm bg-white/[0.04] border border-white/10" /> Empty
                </span>
              </div>

              {/* Numbered grid */}
              <div className="flex-1 overflow-y-auto p-4">
                <div className="grid grid-cols-6 gap-2">
                  {examSession.questions.map((q, index) => {
                    const isAnswered = !!examSession.userAnswers[q.id]
                    const isCurrent = index === examSession.currentQuestion
                    return (
                      <button
                        key={q.id}
                        onClick={() => { goToQuestion(index); setNavOpen(false) }}
                        aria-label={`Go to question ${index + 1}${isAnswered ? ', answered' : ', unanswered'}${isCurrent ? ', current' : ''}`}
                        className={`aspect-square rounded-lg text-sm font-bold transition-all ${
                          isCurrent
                            ? 'bg-gradient-brand text-white ring-1 ring-inset ring-cyan-400/30 glow-brand-sm'
                            : isAnswered
                            ? 'bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 hover:bg-emerald-500/20'
                            : 'bg-white/[0.04] border border-white/10 text-zinc-400 hover:border-cyan-400/30 hover:text-zinc-200'
                        }`}
                      >
                        {index + 1}
                      </button>
                    )
                  })}
                </div>
              </div>

              {/* Footer — progress + submit */}
              <div className="border-t border-white/10 p-4 space-y-3">
                <div className="flex items-center gap-2 text-xs text-zinc-400">
                  <GraduationCap className="h-3.5 w-3.5" />
                  {answeredCount}/{examSession.questions.length} answered
                </div>
                <Button
                  onClick={() => { setNavOpen(false); submitExam() }}
                  leftIcon={<Flag className="w-3.5 h-3.5" />}
                  className="w-full !bg-emerald-600 hover:!bg-emerald-500"
                >
                  Finish &amp; submit
                </Button>
              </div>
            </motion.aside>
          </>
        )}
      </AnimatePresence>
    </div>
  )
}
