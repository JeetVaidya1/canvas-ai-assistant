import { useState, useEffect, useMemo, useCallback } from 'react'
import { generateQuiz, submitQuizAnswer, submitQuiz, type QuizResult } from '@/lib/api'
import { showError } from '@/lib/toast'
import { usePracticeTopics } from '@/hooks/useTopics'
import { useInvalidateProgress } from '@/hooks/useInvalidateProgress'
import { useSessionTimer } from './useSessionTimer'
import { WHOLE_COURSE } from './constants'
import type { QuizDifficulty, QuizRunState, TopicListState } from './types'

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
  startQuiz: () => Promise<void>
  submitAnswer: () => Promise<void>
  nextQuestion: () => Promise<void>
  resetQuiz: () => void
}

/**
 * Quick Quiz state machine: setup → generate → answer-by-answer grading →
 * server-side scoring. Progress caches are invalidated after scoring because
 * mastery changes server-side.
 */
export function useQuizRun(courseId: string, userId: string): QuizController {
  const [run, setRun] = useState<QuizRunState | null>(null)
  const [result, setResult] = useState<QuizResult | null>(null)
  const [selectedTopic, setSelectedTopic] = useState(WHOLE_COURSE)
  const [difficulty, setDifficulty] = useState<QuizDifficulty>('medium')
  const [questionCount, setQuestionCount] = useState(10)
  const [loading, setLoading] = useState(false)
  const [submitting, setSubmitting] = useState(false)

  // Whole-quiz timer (for the summary screen) — ticks while a run is live.
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

  const startQuiz = useCallback(async () => {
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
      resetTimer()
    } catch (e) {
      showError(e instanceof Error ? e.message : 'Failed to generate quiz')
    } finally {
      setLoading(false)
    }
  }, [courseId, selectedTopic, difficulty, questionCount, resetTimer])

  const selectLetter = useCallback((letter: string) => {
    setRun((prev) => (prev && !prev.feedback ? { ...prev, selectedLetter: letter } : prev))
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
      )
      setRun({
        ...run,
        feedback,
        correctCount: run.correctCount + (feedback.is_correct ? 1 : 0),
      })
    } catch (e) {
      showError(e instanceof Error ? e.message : 'Failed to submit answer')
    } finally {
      setSubmitting(false)
    }
  }, [run, userId])

  const finishQuiz = useCallback(async () => {
    if (!run) return
    setSubmitting(true)
    try {
      const finalResult = await submitQuiz(run.quizId, userId)
      setResult(finalResult)
      // Scoring the quiz changed mastery server-side — refresh progress views.
      invalidateProgress()
    } catch (e) {
      showError(e instanceof Error ? e.message : 'Failed to score quiz')
    } finally {
      setSubmitting(false)
    }
  }, [run, userId, invalidateProgress])

  const nextQuestion = useCallback(async () => {
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
  }, [run, finishQuiz])

  const resetQuiz = useCallback(() => {
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
    startQuiz,
    submitAnswer,
    nextQuestion,
    resetQuiz,
  }
}
