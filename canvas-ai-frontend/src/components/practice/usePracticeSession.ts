import { useState, useEffect, useMemo, useCallback } from 'react'
import {
  generatePracticeProblems,
  trackPracticeSession as apiTrackPracticeSession,
} from '@/lib/api'
import { showError } from '@/lib/toast'
import { usePracticeTopics } from '@/hooks/useTopics'
import { useInvalidateProgress } from '@/hooks/useInvalidateProgress'
import { useSessionTimer } from './useSessionTimer'
import type { PracticeDifficulty, PracticeSessionState, TopicListState } from './types'

export interface PracticeController {
  courseId: string
  session: PracticeSessionState | null
  selectedTopic: string
  setSelectedTopic: (topic: string) => void
  difficulty: PracticeDifficulty
  setDifficulty: (difficulty: PracticeDifficulty) => void
  problemCount: number
  setProblemCount: (count: number) => void
  loading: boolean
  selectedAnswer: string
  selectAnswer: (letter: string) => void
  showExplanation: boolean
  timeElapsed: number
  topics: TopicListState
  startSession: () => Promise<void>
  submitAnswer: () => void
  nextProblem: () => void
  resetSession: () => void
}

/**
 * Problem Set state machine: setup → generate (adaptive difficulty happens
 * server-side) → answer/reveal per problem → client-side grading + session
 * tracking, then progress-cache invalidation.
 */
export function usePracticeSession(courseId: string, userId: string): PracticeController {
  const [session, setSession] = useState<PracticeSessionState | null>(null)
  const [selectedTopic, setSelectedTopic] = useState('')
  const [difficulty, setDifficulty] = useState<PracticeDifficulty>('adaptive')
  const [problemCount, setProblemCount] = useState(5)
  const [loading, setLoading] = useState(false)
  const [selectedAnswer, setSelectedAnswer] = useState('')
  const [showExplanation, setShowExplanation] = useState(false)

  const { timeElapsed, reset: resetTimer } = useSessionTimer(!!session && !session.isComplete)

  const topicsQuery = usePracticeTopics(courseId)
  const invalidateProgress = useInvalidateProgress(courseId)

  const availableTopics = useMemo(() => {
    if (!courseId) return ['General Topics']
    const topics = topicsQuery.data?.topics
    return topics && topics.length > 0 ? topics : ['Course Content', 'General Review']
  }, [courseId, topicsQuery.data])

  // Keep the selected topic valid as the list loads/refreshes.
  useEffect(() => {
    setSelectedTopic((prev) =>
      prev && availableTopics.includes(prev) ? prev : availableTopics[0] ?? '',
    )
  }, [availableTopics])

  // isFetching (not isPending) so the Refresh action also shows as loading.
  const topicsLoading = !!courseId && topicsQuery.isFetching
  const topics: TopicListState = {
    options: availableTopics.map((t) => ({ value: t, label: t })),
    loading: topicsLoading,
    pending: !!courseId && topicsQuery.isPending,
    // Genuinely no indexed topics (successful fetch, nothing to practice from).
    empty:
      !!courseId &&
      topicsQuery.isSuccess &&
      !topicsQuery.data?.error &&
      !(topicsQuery.data?.topics && topicsQuery.data.topics.length > 0),
    error: !courseId
      ? 'Please select a course first'
      : topicsQuery.isError
        ? 'Failed to load topics. Please try again.'
        : topicsQuery.data?.error ?? null,
    refetch: () => {
      void topicsQuery.refetch()
    },
  }

  const startSession = useCallback(async () => {
    if (!courseId || !selectedTopic) return
    setLoading(true)
    try {
      const problems = await generatePracticeProblems(
        courseId,
        selectedTopic,
        difficulty,
        problemCount,
        userId,
      )
      setSession({
        problems,
        currentProblemIndex: 0,
        userAnswers: new Array(problems.length).fill(''),
        startTime: new Date(),
        isComplete: false,
        score: 0,
      })
      resetTimer()
      setSelectedAnswer('')
      setShowExplanation(false)
    } catch (e) {
      console.error('Failed to generate practice problems:', e)
      showError('Failed to generate practice problems. Please try again.')
    } finally {
      setLoading(false)
    }
  }, [courseId, selectedTopic, difficulty, problemCount, userId, resetTimer])

  const selectAnswer = useCallback(
    (letter: string) => {
      if (!showExplanation) setSelectedAnswer(letter)
    },
    [showExplanation],
  )

  const submitAnswer = useCallback(() => {
    if (!session || !selectedAnswer) return
    const userAnswers = [...session.userAnswers]
    userAnswers[session.currentProblemIndex] = selectedAnswer
    setSession({ ...session, userAnswers })
    setShowExplanation(true)
  }, [session, selectedAnswer])

  const trackPractice = useCallback(
    async (correct: number, total: number) => {
      try {
        await apiTrackPracticeSession(
          userId,
          courseId,
          selectedTopic,
          total,
          correct,
          Math.max(1, Math.round(timeElapsed / 60)),
          difficulty,
        )
        // The session changed mastery server-side — refresh progress views.
        invalidateProgress()
      } catch (e) {
        console.warn('Practice tracking failed (non-blocking):', e)
      }
    },
    [userId, courseId, selectedTopic, timeElapsed, difficulty, invalidateProgress],
  )

  const completeSession = useCallback(() => {
    if (!session) return
    const correct = session.userAnswers.filter(
      (a, i) => a === session.problems[i].correct_answer,
    ).length
    const score = Math.round((correct / session.problems.length) * 100)
    setSession({ ...session, isComplete: true, score })
    void trackPractice(correct, session.problems.length)
  }, [session, trackPractice])

  const nextProblem = useCallback(() => {
    if (!session) return
    if (session.currentProblemIndex < session.problems.length - 1) {
      setSession({ ...session, currentProblemIndex: session.currentProblemIndex + 1 })
      setSelectedAnswer('')
      setShowExplanation(false)
    } else {
      completeSession()
    }
  }, [session, completeSession])

  const resetSession = useCallback(() => {
    setSession(null)
    setSelectedAnswer('')
    setShowExplanation(false)
  }, [])

  return {
    courseId,
    session,
    selectedTopic,
    setSelectedTopic,
    difficulty,
    setDifficulty,
    problemCount,
    setProblemCount,
    loading,
    selectedAnswer,
    selectAnswer,
    showExplanation,
    timeElapsed,
    topics,
    startSession,
    submitAnswer,
    nextProblem,
    resetSession,
  }
}
