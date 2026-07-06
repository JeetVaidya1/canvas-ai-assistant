import { queryOptions, useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { getCourseTopics, rebuildCourseTopics } from '@/lib/api/topics'

/** Topics only change when materials change — cache generously. */
export const COURSE_TOPICS_STALE_MS = 10 * 60 * 1000

export function courseTopicsOptions(courseId: string) {
  return queryOptions({
    queryKey: ['courseTopics', courseId],
    queryFn: () => getCourseTopics(courseId),
    staleTime: COURSE_TOPICS_STALE_MS,
  })
}

/** The Course Brain taxonomy: clean names, descriptions, prereqs, coverage. */
export function useCourseTopics(courseId: string | undefined) {
  return useQuery({
    ...courseTopicsOptions(courseId ?? ''),
    enabled: !!courseId,
  })
}

export function useRebuildCourseTopics(courseId: string | undefined) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: () => rebuildCourseTopics(courseId ?? ''),
    onSuccess: (data) => {
      qc.setQueryData(['courseTopics', courseId], data)
      // Topic names feed the practice/quiz pickers too.
      void qc.invalidateQueries({ queryKey: ['topics', courseId] })
    },
  })
}
