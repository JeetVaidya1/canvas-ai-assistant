import { queryOptions, useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  getFlashcardDeck,
  saveFlashcards,
  reviewFlashcard,
} from '@/lib/api'

export const FLASHCARDS_STALE_MS = 5 * 60 * 1000

export function flashcardDeckOptions(courseId: string, userId: string) {
  return queryOptions({
    queryKey: ['flashcards', courseId, userId],
    queryFn: () => getFlashcardDeck(courseId, userId),
    staleTime: FLASHCARDS_STALE_MS,
  })
}

export function useFlashcardDeck(courseId: string | undefined, userId: string, enabled = true) {
  return useQuery({
    ...flashcardDeckOptions(courseId ?? '', userId),
    enabled: enabled && !!courseId,
  })
}

export function useSaveFlashcards(courseId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (cards: { q: string; a: string }[]) => saveFlashcards(courseId, cards),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['flashcards', courseId] }),
  })
}

export function useReviewFlashcard(courseId: string | undefined, userId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ cardId, grade }: { cardId: string; grade: number }) =>
      reviewFlashcard(cardId, grade, userId),
    onSuccess: () => {
      // Marks the cached deck stale; active review sessions work off a local
      // snapshot, so this only affects the next deck load.
      if (courseId) void qc.invalidateQueries({ queryKey: ['flashcards', courseId] })
    },
  })
}
