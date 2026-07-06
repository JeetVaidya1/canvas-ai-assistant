import { useCallback, useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getInProgressQuizzes, type QuizInProgressSession } from '@/lib/api'
import { addDismissedQuiz, readDismissedQuizzes } from '@/lib/resumeKeys'

export type { QuizInProgressSession }

/** Sessions change with every answered question — keep this reasonably fresh. */
export const QUIZ_IN_PROGRESS_STALE_MS = 30_000

export interface QuizInProgressState {
  /** Resumable sessions, newest first, with client-dismissed ids filtered out. */
  sessions: readonly QuizInProgressSession[]
  /** Hide a session everywhere (persists via localStorage). */
  dismiss: (quizId: string) => void
  isPending: boolean
  isError: boolean
  refetch: () => void
}

/**
 * Resumable (non-completed) quizzes for a course. Dismissals are client-side
 * only: ids live in localStorage and are filtered out of `sessions` for every
 * consumer of this hook.
 */
export function useQuizInProgress(courseId: string | undefined): QuizInProgressState {
  const query = useQuery({
    queryKey: ['quizInProgress', courseId],
    queryFn: async () => {
      const data = await getInProgressQuizzes(courseId ?? '')
      // Never trust external data: keep only rows with a usable quiz id.
      return (Array.isArray(data.sessions) ? data.sessions : []).filter(
        (s) => typeof s?.quiz_id === 'string' && s.quiz_id.length > 0,
      )
    },
    staleTime: QUIZ_IN_PROGRESS_STALE_MS,
    enabled: !!courseId,
  })

  const [dismissed, setDismissed] = useState<readonly string[]>(readDismissedQuizzes)

  const dismiss = useCallback((quizId: string) => {
    setDismissed(addDismissedQuiz(quizId))
  }, [])

  const sessions = useMemo(
    () => (query.data ?? []).filter((s) => !dismissed.includes(s.quiz_id)),
    [query.data, dismissed],
  )

  return {
    sessions,
    dismiss,
    isPending: query.isPending,
    isError: query.isError,
    refetch: () => {
      void query.refetch()
    },
  }
}
