import { queryOptions, useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  getChatSessions,
  getSessionMessages,
  deleteSession,
} from '@/lib/api'

export function sessionsOptions(userId: string) {
  return queryOptions({
    queryKey: ['sessions', userId],
    queryFn: () => getChatSessions(userId),
  })
}

export function messagesOptions(sessionId: string) {
  return queryOptions({
    queryKey: ['messages', sessionId],
    queryFn: () => getSessionMessages(sessionId),
  })
}

export function useSessions(userId: string) {
  return useQuery({
    ...sessionsOptions(userId),
    enabled: !!userId,
  })
}

export function useSessionMessages(sessionId: string | undefined) {
  return useQuery({
    ...messagesOptions(sessionId ?? ''),
    enabled: !!sessionId,
  })
}

export function useDeleteSession(userId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (sessionId: string) => deleteSession(sessionId),
    onSuccess: (_data, sessionId) => {
      qc.removeQueries({ queryKey: ['messages', sessionId] })
      void qc.invalidateQueries({ queryKey: ['sessions', userId] })
    },
  })
}
