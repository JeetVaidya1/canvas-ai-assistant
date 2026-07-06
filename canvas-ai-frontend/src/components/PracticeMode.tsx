import { usePracticeSession } from '@/components/practice/usePracticeSession'
import { ProblemSetSetup } from '@/components/practice/ProblemSetSetup'
import { ProblemCard } from '@/components/practice/ProblemCard'
import { SessionSummary } from '@/components/practice/SessionSummary'
import type { ModeChangeHandler } from '@/components/practice/types'

interface PracticeModeProps {
  courseId: string
  userId: string
  onModeChange?: ModeChangeHandler
}

/**
 * Problem Set — thin composition over the practice feature folder:
 * setup → active session → summary. All state lives in usePracticeSession.
 */
export default function PracticeMode({ courseId, userId, onModeChange }: PracticeModeProps) {
  const practice = usePracticeSession(courseId, userId)

  if (!practice.session) return <ProblemSetSetup practice={practice} />
  if (practice.session.isComplete) {
    return <SessionSummary practice={practice} onModeChange={onModeChange} />
  }
  return <ProblemCard practice={practice} />
}
