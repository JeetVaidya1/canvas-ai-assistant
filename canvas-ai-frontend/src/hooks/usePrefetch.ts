import { useCallback } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { listFiles } from '@/lib/api'
import { useUser } from './useUser'
import { sessionsOptions } from './useChatSessions'
import { readinessOptions } from './useReadiness'
import { analyticsOptions, conceptGraphOptions } from './useAnalytics'
import { topicsOptions } from './useTopics'
import { notesLibraryOptions } from './useNotesLibrary'
import { reviewQueueOptions } from './useReviews'

/**
 * Hover-intent prefetching. Wire these to onMouseEnter on nav links / action
 * cards so a destination's primary data is already cached when it mounts.
 * prefetchQuery is a no-op when fresh data exists, and prefetch errors are
 * intentionally silent — the destination surfaces failures with retry UI.
 */
export function usePrefetch() {
  const qc = useQueryClient()
  const userId = useUser()

  /** Course home: files + readiness. */
  const prefetchCourse = useCallback(
    (courseId: string) => {
      void qc.prefetchQuery({ queryKey: ['files', courseId], queryFn: () => listFiles(courseId) })
      if (userId) void qc.prefetchQuery(readinessOptions(courseId, userId))
    },
    [qc, userId],
  )

  /** Learn (chat): session list. */
  const prefetchLearn = useCallback(() => {
    if (userId) void qc.prefetchQuery(sessionsOptions(userId))
  }, [qc, userId])

  /** Practice / quiz setup: topic list. */
  const prefetchPractice = useCallback(
    (courseId: string) => {
      void qc.prefetchQuery(topicsOptions(courseId))
    },
    [qc],
  )

  /** Study Kit (notes studio): notes library + files. */
  const prefetchStudyKit = useCallback(
    (courseId: string) => {
      void qc.prefetchQuery(notesLibraryOptions(courseId))
      void qc.prefetchQuery({ queryKey: ['files', courseId], queryFn: () => listFiles(courseId) })
    },
    [qc],
  )

  /** Progress: analytics + readiness + concept graph + review queue. */
  const prefetchProgress = useCallback(
    (courseId: string) => {
      if (!userId) return
      void qc.prefetchQuery(analyticsOptions(courseId, userId))
      void qc.prefetchQuery(readinessOptions(courseId, userId))
      void qc.prefetchQuery(conceptGraphOptions(courseId, userId))
      void qc.prefetchQuery(reviewQueueOptions(courseId, userId))
    },
    [qc, userId],
  )

  return { prefetchCourse, prefetchLearn, prefetchPractice, prefetchStudyKit, prefetchProgress }
}
