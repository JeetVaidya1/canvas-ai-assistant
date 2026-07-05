import { queryOptions, useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { getReviewQueue, gradeReview } from '@/lib/api'

/** The review queue changes as items come due / get graded. */
export const REVIEWS_STALE_MS = 2 * 60 * 1000

export function reviewQueueOptions(courseId: string, userId: string) {
  return queryOptions({
    queryKey: ['reviews', courseId, userId],
    queryFn: () => getReviewQueue(courseId, userId),
    staleTime: REVIEWS_STALE_MS,
  })
}

export function useReviewQueue(courseId: string | undefined, userId: string) {
  return useQuery({
    ...reviewQueueOptions(courseId ?? '', userId),
    enabled: !!courseId && !!userId,
  })
}

export function useGradeReview(courseId: string | undefined, userId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ itemId, grade }: { itemId: string; grade: number }) =>
      gradeReview(itemId, grade, userId),
    onSuccess: () => {
      if (!courseId) return
      // Grading changes the queue and feeds mastery → readiness. Active review
      // sessions work off a local snapshot, so refetches never disrupt them.
      void qc.invalidateQueries({ queryKey: ['reviews', courseId] })
      void qc.invalidateQueries({ queryKey: ['readiness', courseId] })
    },
  })
}
