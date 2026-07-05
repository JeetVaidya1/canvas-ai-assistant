import { queryOptions, useQuery } from '@tanstack/react-query'
import { getReadiness } from '@/lib/api'

/** Readiness moves as the user studies — keep it fresher than the 5min default. */
export const READINESS_STALE_MS = 2 * 60 * 1000

export function readinessOptions(courseId: string, userId: string) {
  return queryOptions({
    queryKey: ['readiness', courseId, userId],
    queryFn: () => getReadiness(courseId, userId),
    staleTime: READINESS_STALE_MS,
  })
}

export function useReadiness(courseId: string | undefined, userId: string) {
  return useQuery({
    ...readinessOptions(courseId ?? '', userId),
    enabled: !!courseId && !!userId,
  })
}
