import { queryOptions, useQuery } from '@tanstack/react-query'
import { getLearningAnalytics, getConceptGraph } from '@/lib/api'

/** Analytics move with study activity — fresher than the 5min default. */
export const ANALYTICS_STALE_MS = 2 * 60 * 1000
/** The concept graph is expensive to build server-side; cache it longer. */
export const CONCEPT_GRAPH_STALE_MS = 5 * 60 * 1000

export function analyticsOptions(courseId: string, userId: string) {
  return queryOptions({
    queryKey: ['analytics', courseId, userId],
    // Backend may answer without an analytics payload — normalize to null so
    // callers can render an explicit "no data yet" state.
    queryFn: async () => (await getLearningAnalytics(courseId, userId)) ?? null,
    staleTime: ANALYTICS_STALE_MS,
  })
}

export function conceptGraphOptions(courseId: string, userId: string) {
  return queryOptions({
    queryKey: ['conceptGraph', courseId, userId],
    queryFn: () => getConceptGraph(courseId, userId),
    staleTime: CONCEPT_GRAPH_STALE_MS,
  })
}

export function useLearningAnalytics(courseId: string | undefined, userId: string) {
  return useQuery({
    ...analyticsOptions(courseId ?? '', userId),
    enabled: !!courseId && !!userId,
  })
}

export function useConceptGraph(courseId: string | undefined, userId: string) {
  return useQuery({
    ...conceptGraphOptions(courseId ?? '', userId),
    enabled: !!courseId && !!userId,
  })
}
