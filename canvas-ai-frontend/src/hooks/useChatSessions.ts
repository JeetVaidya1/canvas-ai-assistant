import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  getChatSessions,
  getSessionMessages,
  deleteSession,
} from '@/lib/api'

export function useSessions(userId: string) {
  return useQuery({
    queryKey: ['sessions', userId],
    queryFn: () => getChatSessions(userId),
  })
}

export function useSessionMessages(sessionId: string | undefined) {
  return useQuery({
    queryKey: ['messages', sessionId],
    queryFn: () => getSessionMessages(sessionId!),
    enabled: !!sessionId,
  })
}

export function useDeleteSession(userId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (sessionId: string) => deleteSession(sessionId),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['sessions', userId] }),
  })
}
