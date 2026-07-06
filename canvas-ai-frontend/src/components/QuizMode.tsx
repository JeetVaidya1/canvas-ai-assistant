import { useQuizRun } from '@/components/practice/useQuizRun'
import { QuizSetup } from '@/components/practice/QuizSetup'
import { QuizSession } from '@/components/practice/QuizSession'
import { QuizResults } from '@/components/practice/QuizResults'
import type { ModeChangeHandler } from '@/components/practice/types'

interface QuizModeProps {
  courseId: string
  userId: string
  onModeChange?: ModeChangeHandler
}

/**
 * Quick Quiz — thin composition over the practice feature folder:
 * setup → active run → results. All state lives in useQuizRun.
 */
export default function QuizMode({ courseId, userId, onModeChange }: QuizModeProps) {
  const quiz = useQuizRun(courseId, userId)

  if (!quiz.run) return <QuizSetup quiz={quiz} />
  if (quiz.result) return <QuizResults quiz={quiz} onModeChange={onModeChange} />
  return <QuizSession quiz={quiz} />
}
