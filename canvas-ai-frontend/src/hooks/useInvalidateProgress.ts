import { useCallback } from 'react'
import { useQueryClient } from '@tanstack/react-query'

/**
 * Returns a callback that invalidates every query derived from study activity
 * (readiness, analytics, concept graph, review queue). Call it after actions
 * that change mastery server-side — submitting a quiz or exam, finishing a
 * practice session, or grading spaced-review items — so progress views show
 * fresh numbers instead of a 5-minute-old cache.
 */
export function useInvalidateProgress(courseId: string | undefined) {
  const qc = useQueryClient()
  return useCallback(() => {
    if (!courseId) return
    void qc.invalidateQueries({ queryKey: ['readiness', courseId] })
    void qc.invalidateQueries({ queryKey: ['analytics', courseId] })
    void qc.invalidateQueries({ queryKey: ['conceptGraph', courseId] })
    void qc.invalidateQueries({ queryKey: ['reviews', courseId] })
  }, [qc, courseId])
}
