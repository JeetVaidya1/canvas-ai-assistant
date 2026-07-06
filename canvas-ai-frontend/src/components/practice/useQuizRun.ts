import { useState, useEffect, useMemo, useCallback, useRef } from 'react'
import {
  generateQuiz,
  getQuizQuestions,
  getQuizResponses,
  submitQuizAnswer,
  submitQuiz,
  type QuizConfidence,
  type QuizQuestion,
  type QuizResult,
  type QuizStoredResponse,
} from '@/lib/api'
import { showError } from '@/lib/toast'
import { usePracticeTopics } from '@/hooks/useTopics'
import { useInvalidateProgress } from '@/hooks/useInvalidateProgress'
import { useSessionTimer } from './useSessionTimer'
import { WHOLE_COURSE } from './constants'
import type { AnsweredQuestion, QuizDifficulty, QuizRunState, TopicListState } from './types'

/** How often to check for freshly written questions while generating. */
const GENERATION_POLL_MS = 2500
/** Consecutive poll failures tolerated before ending the quiz at what exists. */
const MAX_POLL_FAILURES = 4

export interface QuizController {
  courseId: string
  run: QuizRunState | null
  result: QuizResult | null
  selectedTopic: string
  setSelectedTopic: (topic: string) => void
  difficulty: QuizDifficulty
  setDifficulty: (difficulty: QuizDifficulty) => void
  questionCount: number
  setQuestionCount: (count: number) => void
  loading: boolean
  submitting: boolean
  timeElapsed: number
  topics: TopicListState
  selectLetter: (letter: string) => void
  setConfidence: (confidence: QuizConfidence | null) => void
  /** Starts a run; `topicOverride` lets the debrief re-drill a weak topic directly. */
  startQuiz: (topicOverride?: string) => Promise<void>
  /** Rebuilds a run from a previously started quiz (server questions + graded answers). */
  restoreQuiz: (quizId: string, topicLabel?: string | null) => Promise<void>
  submitAnswer: () => Promise<void>
  nextQuestion: () => Promise<void>
  /** Score the run at whatever has been answered (recovery from a failed auto-finish). */
  finishNow: () => Promise<void>
  resetQuiz: () => void
}

/**
 * Rebuild the client-side answer log from server-stored responses. The server
 * only stores the pick + verdict, so restored entries carry an honest sparse
 * result: no explanation, and the correct letter only when the pick was right.
 */
function rebuildAnswers(
  questions: QuizQuestion[],
  byId: ReadonlyMap<string, QuizStoredResponse>,
): AnsweredQuestion[] {
  return questions.flatMap((question) => {
    const stored = byId.get(question.id)
    if (!stored) return []
    return [
      {
        question,
        selectedLetter: stored.selected,
        confidence: stored.confidence ?? null,
        result: {
          is_correct: stored.is_correct,
          correct_answer: stored.is_correct ? stored.selected : '',
          explanation: '',
          concept: question.concept,
          source: question.source,
        },
      },
    ]
  })
}

/** Append questions we haven't seen; never reorder ones already in play. */
function mergeQuestions(existing: QuizQuestion[], incoming: QuizQuestion[]): QuizQuestion[] {
  const known = new Set(existing.map((q) => q.id))
  const added = incoming.filter((q) => !known.has(q.id))
  return added.length ? [...existing, ...added] : existing
}

/**
 * Quick Quiz state machine, fast-start edition:
 * setup → generate (returns first ~3 questions immediately) → answer-by-answer
 * grading while a background poll merges the rest in → server-side scoring.
 * If the user outruns generation, the session shows an honest waiting state; if
 * the backend can only produce a partial set, the run ends gracefully at what
 * exists. Progress caches are invalidated after scoring (mastery moves server-side).
 */
export function useQuizRun(courseId: string, userId: string): QuizController {
  const [run, setRun] = useState<QuizRunState | null>(null)
  const [result, setResult] = useState<QuizResult | null>(null)
  const [selectedTopic, setSelectedTopic] = useState(WHOLE_COURSE)
  const [difficulty, setDifficulty] = useState<QuizDifficulty>('medium')
  const [questionCount, setQuestionCount] = useState(10)
  const [loading, setLoading] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  // One-shot latch so auto-finish (partial generation) can't double-submit.
  const finishingRef = useRef(false)
  // Auto-finish tries once; on failure the session offers a manual retry
  // instead of toast-looping.
  const autoFinishRef = useRef(false)

  // Whole-quiz timer (for the debrief) — ticks while a run is live.
  const { timeElapsed, reset: resetTimer } = useSessionTimer(!!run && !result)

  // "Whole course" is always available and the default — specific topics just
  // let the user narrow the focus. The shared hook attaches the auth token.
  const topicsQuery = usePracticeTopics(courseId)
  const invalidateProgress = useInvalidateProgress(courseId)

  const availableTopics = useMemo(
    () => [WHOLE_COURSE, ...(topicsQuery.data?.topics?.filter(Boolean) ?? [])],
    [topicsQuery.data],
  )

  // Keep the selected topic valid as the list loads/refreshes.
  useEffect(() => {
    if (!availableTopics.includes(selectedTopic)) setSelectedTopic(WHOLE_COURSE)
  }, [availableTopics, selectedTopic])

  // isFetching (not isPending) so the Refresh action also shows as loading.
  const topicsLoading = !!courseId && topicsQuery.isFetching
  const topics: TopicListState = {
    options: availableTopics.map((t) => ({
      value: t,
      label: t,
      hint: t === WHOLE_COURSE ? 'Broad — core concepts from everywhere' : undefined,
    })),
    loading: topicsLoading,
    pending: !!courseId && topicsQuery.isPending,
    // Whole course always works, so an empty topic list never blocks a quiz.
    empty: false,
    error: topicsQuery.isError
      ? 'Could not load specific topics — you can still quiz the whole course.'
      : topicsQuery.data?.error ?? null,
    refetch: () => {
      void topicsQuery.refetch()
    },
  }

  const startQuiz = useCallback(
    async (topicOverride?: string) => {
      if (!courseId) return
      setLoading(true)
      try {
        const topicLabel = topicOverride ?? selectedTopic
        // Null topic => backend retrieves broadly across the whole course.
        const topicArg = topicLabel === WHOLE_COURSE ? null : topicLabel
        const quiz = await generateQuiz(courseId, topicArg, difficulty, questionCount)
        if (!quiz.questions.length) {
          showError('No questions could be generated. Try another topic.')
          return
        }
        finishingRef.current = false
        autoFinishRef.current = false
        setRun({
          quizId: quiz.quiz_id,
          questions: quiz.questions,
          numRequested: quiz.num_requested || quiz.questions.length,
          generationStatus: quiz.generation_status ?? 'ready',
          topicLabel,
          currentIndex: 0,
          selectedLetter: '',
          confidence: null,
          feedback: null,
          questionStart: Date.now(),
          correctCount: 0,
          answers: [],
        })
        setResult(null)
        resetTimer()
      } catch (e) {
        showError(e instanceof Error ? e.message : 'Failed to generate quiz')
      } finally {
        setLoading(false)
      }
    },
    [courseId, selectedTopic, difficulty, questionCount, resetTimer],
  )

  // Resume a previously started quiz: refetch its questions and graded answers,
  // rebuild the run at the first unanswered question, and let the existing
  // poll/auto-finish effects take over (they key off generationStatus).
  const restoreQuiz = useCallback(
    async (restoreId: string, topicLabel?: string | null) => {
      if (!restoreId) return
      setLoading(true)
      try {
        const [questionData, responseData] = await Promise.all([
          getQuizQuestions(restoreId),
          getQuizResponses(restoreId),
        ])
        if (!questionData.questions.length) {
          showError('That quiz has no questions to resume — start a fresh drill instead.')
          return
        }
        const byId = new Map(
          (Array.isArray(responseData.responses) ? responseData.responses : []).map(
            (r) => [r.question_id, r] as const,
          ),
        )
        const answers = rebuildAnswers(questionData.questions, byId)
        const firstUnanswered = questionData.questions.findIndex((q) => !byId.has(q.id))
        // Everything available is answered -> park past the end: the existing
        // effects either wait for the next generated question or score the run.
        const currentIndex =
          firstUnanswered === -1 ? questionData.questions.length : firstUnanswered
        finishingRef.current = false
        autoFinishRef.current = false
        setRun({
          quizId: questionData.quiz_id || restoreId,
          questions: questionData.questions,
          numRequested: questionData.num_requested || questionData.questions.length,
          generationStatus: questionData.generation_status ?? 'ready',
          topicLabel: topicLabel ?? WHOLE_COURSE,
          currentIndex,
          selectedLetter: '',
          confidence: null,
          feedback: null,
          questionStart: Date.now(),
          correctCount: answers.filter((a) => a.result.is_correct).length,
          answers,
        })
        setResult(null)
        resetTimer()
      } catch (e) {
        // Run stays null, so the setup screen (with its resume cards) remains
        // the retry path.
        showError(e instanceof Error ? e.message : 'Could not resume that quiz — please try again.')
      } finally {
        setLoading(false)
      }
    },
    [resetTimer],
  )

  // Background poll: while the backend is still writing questions, pull what
  // exists every few seconds and merge it in. Stops on 'ready'/'partial', on
  // reset, or after repeated failures (then ends honestly at what we have).
  const quizId = run?.quizId ?? null
  const polling = !!quizId && run?.generationStatus === 'generating' && !result
  useEffect(() => {
    if (!polling || !quizId) return
    let cancelled = false
    let failures = 0
    const tick = async () => {
      try {
        const data = await getQuizQuestions(quizId)
        if (cancelled) return
        failures = 0
        setRun((prev) => {
          if (!prev || prev.quizId !== quizId) return prev
          return {
            ...prev,
            questions: mergeQuestions(prev.questions, data.questions),
            generationStatus: data.generation_status,
            numRequested: data.num_requested || prev.numRequested,
          }
        })
      } catch {
        if (cancelled) return
        failures += 1
        if (failures >= MAX_POLL_FAILURES) {
          showError('Lost contact while writing the remaining questions — ending at what we have.')
          setRun((prev) =>
            prev && prev.quizId === quizId ? { ...prev, generationStatus: 'partial' } : prev,
          )
        }
      }
    }
    const interval = window.setInterval(() => void tick(), GENERATION_POLL_MS)
    void tick()
    return () => {
      cancelled = true
      window.clearInterval(interval)
    }
  }, [polling, quizId])

  const selectLetter = useCallback((letter: string) => {
    setRun((prev) => (prev && !prev.feedback ? { ...prev, selectedLetter: letter } : prev))
  }, [])

  const setConfidence = useCallback((confidence: QuizConfidence | null) => {
    setRun((prev) => (prev && !prev.feedback ? { ...prev, confidence } : prev))
  }, [])

  const submitAnswer = useCallback(async () => {
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
        run.confidence ?? undefined,
      )
      // Functional update: the generation poll may have merged questions since.
      setRun((prev) =>
        prev
          ? {
              ...prev,
              feedback,
              correctCount: prev.correctCount + (feedback.is_correct ? 1 : 0),
              answers: [
                ...prev.answers,
                { question, selectedLetter: prev.selectedLetter, confidence: prev.confidence, result: feedback },
              ],
            }
          : prev,
      )
    } catch (e) {
      showError(e instanceof Error ? e.message : 'Failed to submit answer')
    } finally {
      setSubmitting(false)
    }
  }, [run, userId])

  const finishQuiz = useCallback(async () => {
    if (!run || finishingRef.current) return
    finishingRef.current = true
    setSubmitting(true)
    try {
      const finalResult = await submitQuiz(run.quizId, userId)
      setResult(finalResult)
      // Scoring the quiz changed mastery server-side — refresh progress views.
      invalidateProgress()
    } catch (e) {
      finishingRef.current = false
      showError(e instanceof Error ? e.message : 'Failed to score quiz')
    } finally {
      setSubmitting(false)
    }
  }, [run, userId, invalidateProgress])

  const nextQuestion = useCallback(async () => {
    if (!run || !run.feedback) return
    const answered = run.currentIndex + 1
    const doneRequested = answered >= run.numRequested
    const doneAvailable = run.generationStatus !== 'generating' && answered >= run.questions.length
    if (doneRequested || doneAvailable) {
      await finishQuiz()
      return
    }
    // May step past the last available question — the session shows an honest
    // waiting state and the poll effect fills it in as soon as it lands.
    setRun((prev) =>
      prev
        ? {
            ...prev,
            currentIndex: prev.currentIndex + 1,
            selectedLetter: '',
            confidence: null,
            feedback: null,
            questionStart: Date.now(),
          }
        : prev,
    )
  }, [run, finishQuiz])

  // If the user is parked on the waiting state and generation ends without
  // producing that question ('partial'), close the run out gracefully. One
  // attempt only — a failure surfaces a manual "score it" action instead.
  useEffect(() => {
    if (!run || result || submitting || autoFinishRef.current) return
    if (run.currentIndex >= run.questions.length && run.generationStatus !== 'generating') {
      autoFinishRef.current = true
      void finishQuiz()
    }
  }, [run, result, submitting, finishQuiz])

  const resetQuiz = useCallback(() => {
    finishingRef.current = false
    autoFinishRef.current = false
    setRun(null)
    setResult(null)
    resetTimer()
  }, [resetTimer])

  return {
    courseId,
    run,
    result,
    selectedTopic,
    setSelectedTopic,
    difficulty,
    setDifficulty,
    questionCount,
    setQuestionCount,
    loading,
    submitting,
    timeElapsed,
    topics,
    selectLetter,
    setConfidence,
    startQuiz,
    restoreQuiz,
    submitAnswer,
    nextQuestion,
    finishNow: finishQuiz,
    resetQuiz,
  }
}
