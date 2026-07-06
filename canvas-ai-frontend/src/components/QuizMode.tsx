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
}

/**
 * Quick Quiz — thin composition over the practice feature folder:
 * setup → fast-start run (questions stream in behind the session) → debrief.
 * All state lives in useQuizRun.
 */
export default function QuizMode({ courseId, userId }: QuizModeProps) {
  const quiz = useQuizRun(courseId, userId)

  if (quiz.result) return <QuizResults quiz={quiz} />
  if (!quiz.run) return <QuizSetup quiz={quiz} />
  return <QuizSession quiz={quiz} />
}
