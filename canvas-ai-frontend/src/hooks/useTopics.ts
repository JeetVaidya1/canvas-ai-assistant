import { queryOptions, useQuery } from '@tanstack/react-query'
import { fetchPracticeTopics } from '@/lib/api'

/** Topics come from indexed course content — they change rarely. */
export const TOPICS_STALE_MS = 5 * 60 * 1000

export function topicsOptions(courseId: string) {
  return queryOptions({
    queryKey: ['topics', courseId],
    queryFn: () => fetchPracticeTopics(courseId),
    staleTime: TOPICS_STALE_MS,
  })
}

/**
 * Practice/quiz topic list for a course. Returns the raw backend response
 * ({ topics, error }) so each consumer can apply its own fallback list.
 */
export function usePracticeTopics(courseId: string | undefined) {
  return useQuery({
    ...topicsOptions(courseId ?? ''),
    enabled: !!courseId,
  })
}
