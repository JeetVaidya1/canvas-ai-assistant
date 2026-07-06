import { useState, useEffect, useMemo, useCallback } from 'react'
import {
  generatePracticeProblems,
  trackPracticeSession as apiTrackPracticeSession,
  type PracticeProblem,
} from '@/lib/api'
import { showError } from '@/lib/toast'
import { usePracticeTopics } from '@/hooks/useTopics'
import { useInvalidateProgress } from '@/hooks/useInvalidateProgress'
import { practiceSnapshotKey, readJson, removeKey, writeJson } from '@/lib/resumeKeys'
import { useSessionTimer } from './useSessionTimer'
import type { PracticeDifficulty, PracticeSessionState, TopicListState } from './types'

/** Persisted mid-session state so a reload/back-nav doesn't lose the set. */
export interface PracticeSnapshot {
  problems: PracticeProblem[]
  userAnswers: string[]
  currentProblemIndex: number
  /** Epoch ms of the original session start. */
  startTime: number
  /** Restored so end-of-session tracking stays accurate. */
  topic: string
  difficulty: PracticeDifficulty
}

const PRACTICE_DIFFICULTY_VALUES: readonly PracticeDifficulty[] = [
  'adaptive',
  'easy',
  'medium',
  'hard',
]

/** Shape-check stored snapshot data — never trust localStorage contents. */
function parsePracticeSnapshot(raw: unknown): PracticeSnapshot | null {
  if (!raw || typeof raw !== 'object') return null
  const c = raw as Record<string, unknown>
  if (!Array.isArray(c.problems) || c.problems.length === 0) return null
  const problemsValid = c.problems.every(
    (p) =>
      p &&
      typeof p === 'object' &&
      typeof (p as PracticeProblem).question === 'string' &&
      Array.isArray((p as PracticeProblem).options) &&
      typeof (p as PracticeProblem).correct_answer === 'string',
  )
  if (!problemsValid) return null
  if (!Array.isArray(c.userAnswers) || c.userAnswers.some((a) => typeof a !== 'string')) return null
  if (c.userAnswers.length !== c.problems.length) return null
  if (typeof c.currentProblemIndex !== 'number' || c.currentProblemIndex < 0) return null
  if (c.currentProblemIndex >= c.problems.length) return null
  if (typeof c.startTime !== 'number') return null
  if (typeof c.topic !== 'string') return null
  if (!PRACTICE_DIFFICULTY_VALUES.includes(c.difficulty as PracticeDifficulty)) return null
  return {
    problems: c.problems as PracticeProblem[],
    userAnswers: c.userAnswers as string[],
    currentProblemIndex: c.currentProblemIndex,
    startTime: c.startTime,
    topic: c.topic,
    difficulty: c.difficulty as PracticeDifficulty,
  }
}

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
  /** Interrupted session found in storage (only exposed while no session runs). */
  snapshot: PracticeSnapshot | null
  resumeSnapshot: () => void
  discardSnapshot: () => void
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
  const [snapshot, setSnapshot] = useState<PracticeSnapshot | null>(null)

  const { timeElapsed, reset: resetTimer } = useSessionTimer(!!session && !session.isComplete)

  // Look for an interrupted session when the course changes (mount included).
  // Only offered while nothing is running — an active session always wins.
  useEffect(() => {
    setSnapshot(courseId ? parsePracticeSnapshot(readJson(practiceSnapshotKey(courseId))) : null)
  }, [courseId])

  // Persist the live session; clear the key the moment it completes.
  useEffect(() => {
    if (!courseId || !session) return
    const key = practiceSnapshotKey(courseId)
    if (session.isComplete) {
      removeKey(key)
      return
    }
    const snap: PracticeSnapshot = {
      problems: session.problems,
      userAnswers: session.userAnswers,
      currentProblemIndex: session.currentProblemIndex,
      startTime: session.startTime.getTime(),
      topic: selectedTopic,
      difficulty,
    }
    writeJson(key, snap)
  }, [courseId, session, selectedTopic, difficulty])

  const resumeSnapshot = useCallback(() => {
    if (!snapshot || session) return
    // Land on the first unanswered problem (falls back to the saved index).
    const firstUnanswered = snapshot.userAnswers.findIndex((a) => a === '')
    setSelectedTopic(snapshot.topic)
    setDifficulty(snapshot.difficulty)
    setSession({
      problems: snapshot.problems,
      currentProblemIndex: firstUnanswered === -1 ? snapshot.currentProblemIndex : firstUnanswered,
      userAnswers: snapshot.userAnswers,
      startTime: new Date(snapshot.startTime),
      isComplete: false,
      score: 0,
    })
    setSelectedAnswer('')
    setShowExplanation(false)
    resetTimer()
    setSnapshot(null)
  }, [snapshot, session, resetTimer])

  const discardSnapshot = useCallback(() => {
    if (courseId) removeKey(practiceSnapshotKey(courseId))
    setSnapshot(null)
  }, [courseId])

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
      // A fresh session supersedes any stored snapshot (persist effect overwrites it).
      setSnapshot(null)
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
    snapshot,
    resumeSnapshot,
    discardSnapshot,
  }
}
