import { queryOptions, useQuery } from '@tanstack/react-query'
import { getShareInfo, type SharedCourse } from '@/lib/api'

export const SHARE_INFO_STALE_MS = 5 * 60 * 1000

export function shareInfoOptions(courseId: string) {
  return queryOptions({
    // The backend answers with an error status when a course has never been
    // published — that is the expected "not published yet" state, not a
    // failure, so it is normalized to null instead of surfacing an error.
    queryKey: ['shareInfo', courseId],
    queryFn: async (): Promise<SharedCourse | null> => {
      try {
        return await getShareInfo(courseId)
      } catch {
        return null
      }
    },
    staleTime: SHARE_INFO_STALE_MS,
  })
}

export function useShareInfo(courseId: string | undefined) {
  return useQuery({
    ...shareInfoOptions(courseId ?? ''),
    enabled: !!courseId,
  })
}
