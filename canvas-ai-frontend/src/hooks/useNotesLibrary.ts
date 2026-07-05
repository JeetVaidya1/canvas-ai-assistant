import { queryOptions, useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  getNotes,
  saveNotes,
  updateNote,
  deleteNotes,
  type SavedNote,
} from '@/lib/api'

export const NOTES_STALE_MS = 5 * 60 * 1000

export function notesLibraryOptions(courseId: string) {
  return queryOptions({
    queryKey: ['notes', courseId],
    queryFn: () => getNotes(courseId),
    staleTime: NOTES_STALE_MS,
  })
}

export function useNotesLibrary(courseId: string | undefined) {
  return useQuery({
    ...notesLibraryOptions(courseId ?? ''),
    enabled: !!courseId,
  })
}

export interface SaveNotePayload {
  title: string
  content: string
  sourceFiles: string[]
  topic: string
  /** When present, updates the existing note instead of creating a new one. */
  noteId?: string
}

export function useSaveNote(courseId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ title, content, sourceFiles, topic, noteId }: SaveNotePayload): Promise<SavedNote> =>
      noteId
        ? updateNote(noteId, courseId, title, content, sourceFiles, topic)
        : saveNotes(courseId, title, content, sourceFiles, topic),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['notes', courseId] }),
  })
}

export function useDeleteNote(courseId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (noteId: string) => deleteNotes(noteId),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['notes', courseId] }),
  })
}
