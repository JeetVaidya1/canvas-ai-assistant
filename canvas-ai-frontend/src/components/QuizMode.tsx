import { useEffect, useRef } from 'react'
import { useQuizRun } from '@/components/practice/useQuizRun'
import { QuizSetup } from '@/components/practice/QuizSetup'
import { QuizSession } from '@/components/practice/QuizSession'
import { QuizResults } from '@/components/practice/QuizResults'
import type { ModeChangeHandler } from '@/components/practice/types'

interface QuizModeProps {
  courseId: string
  userId: string
  /** Kept for API compatibility with PracticePage; the debrief routes directly. */
  onModeChange?: ModeChangeHandler
  /** Quiz to restore on mount (from ?resume= deep links / resume cards). */
  resumeQuizId?: string | null
}

/**
 * Quick Quiz — thin composition over the practice feature folder:
 * setup → fast-start run (questions stream in behind the session) → debrief.
 * All state lives in useQuizRun.
 */
export default function QuizMode({ courseId, userId, resumeQuizId }: QuizModeProps) {
  const quiz = useQuizRun(courseId, userId)

  // Deep-link restore: run exactly once per mounted quiz surface. On failure
  // useQuizRun surfaces the error and the setup screen remains the retry path.
  const restoredRef = useRef(false)
  const { restoreQuiz } = quiz
  useEffect(() => {
    if (!resumeQuizId || restoredRef.current) return
    restoredRef.current = true
    void restoreQuiz(resumeQuizId)
  }, [resumeQuizId, restoreQuiz])

  if (quiz.result) return <QuizResults quiz={quiz} />
  if (!quiz.run) return <QuizSetup quiz={quiz} />
  return <QuizSession quiz={quiz} />
}
