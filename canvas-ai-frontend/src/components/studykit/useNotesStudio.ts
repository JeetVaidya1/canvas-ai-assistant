// src/components/studykit/useNotesStudio.ts — all Study Kit state + server interactions
import { useEffect, useMemo, useState } from 'react'
import { generateNotes, type NotesResponse, type SavedNote } from '@/lib/api'
import { showError } from '@/lib/toast'
import { useUser } from '@/hooks/useUser'
import { useCourseFiles } from '@/hooks/useCourseFiles'
import { useNotesLibrary, useSaveNote, useDeleteNote } from '@/hooks/useNotesLibrary'
import {
  LOADING_STAGES,
  parseFlashcardsFromText,
  type Flashcard,
  type NoteStyle,
} from './noteUtils'

/**
 * Single source of truth for the Notes studio. All server data flows through
 * the React Query hooks (useCourseFiles / useNotesLibrary / mutations); this
 * hook only adds the in-progress composition state on top.
 */
export function useNotesStudio(courseId: string) {
  const userId = useUser()
  const filesQuery = useCourseFiles(courseId)
  const notesQuery = useNotesLibrary(courseId)
  const saveNoteMutation = useSaveNote(courseId)
  const deleteNoteMutation = useDeleteNote(courseId)

  const availableFiles = useMemo(() => filesQuery.data ?? [], [filesQuery.data])
  const savedNotes = useMemo(() => notesQuery.data ?? [], [notesQuery.data])
  const saving = saveNoteMutation.isPending

  const [selectedFiles, setSelectedFiles] = useState<string[]>([])
  const [topic, setTopic] = useState('')
  const [noteStyle, setNoteStyle] = useState<NoteStyle>('detailed')
  const [generatedNotes, setGeneratedNotes] = useState('')
  const [currentNoteId, setCurrentNoteId] = useState<string | undefined>(undefined)
  const [noteTitle, setNoteTitle] = useState('')
  const [loading, setLoading] = useState(false)
  const [loadStage, setLoadStage] = useState(0)
  const [libraryOpen, setLibraryOpen] = useState(false)
  const [errMsg, setErrMsg] = useState<string | null>(null)
  const [flashcards, setFlashcards] = useState<Flashcard[]>([])

  // Reset composition state when the course changes.
  useEffect(() => {
    setSelectedFiles([])
    setGeneratedNotes('')
    setNoteTitle('')
    setCurrentNoteId(undefined)
    setTopic('')
    setErrMsg(null)
    setFlashcards([])
    setLibraryOpen(false)
  }, [courseId])

  // Default to ALL files selected so the common path needs zero file-clicks.
  // Only auto-fill when nothing is selected yet (don't clobber a viewed note's set).
  useEffect(() => {
    const list = filesQuery.data
    if (!list || list.length === 0) return
    setSelectedFiles((prev) => (prev.length === 0 ? [...list] : prev))
  }, [filesQuery.data])

  // Advance the honest loading narrative while generating.
  useEffect(() => {
    if (!loading) {
      setLoadStage(0)
      return
    }
    const id = setInterval(() => {
      setLoadStage((s) => Math.min(s + 1, LOADING_STAGES.length - 1))
    }, 2200)
    return () => clearInterval(id)
  }, [loading])

  const generate = async () => {
    if (!courseId || availableFiles.length === 0) return
    // Blank file selection == whole course: fall back to all available files.
    const filesForGen = selectedFiles.length > 0 ? selectedFiles : availableFiles
    setLoading(true)
    setGeneratedNotes('')
    setErrMsg(null)
    setFlashcards([])

    try {
      const response: NotesResponse & { flashcards?: Flashcard[] } = await generateNotes(
        courseId,
        filesForGen,
        topic,
        noteStyle,
      )
      const notes = response.notes || ''
      setGeneratedNotes(notes)
      setNoteTitle(response.suggested_title || `Notes: ${topic || 'Lecture Summary'}`)
      // Use the STRUCTURED flashcards from the backend response.
      const fc = Array.isArray(response.flashcards)
        ? response.flashcards
        : parseFlashcardsFromText(notes)
      setFlashcards(fc || [])
      setCurrentNoteId(undefined)
    } catch (error: unknown) {
      console.error('Failed to generate notes:', error)
      const msg = error instanceof Error ? error.message : 'Failed to generate notes. Please try again.'
      setErrMsg(msg)
      setGeneratedNotes('')
      setFlashcards([])
    } finally {
      setLoading(false)
    }
  }

  const save = async () => {
    if (!courseId || !generatedNotes.trim() || !noteTitle.trim()) return
    setErrMsg(null)
    try {
      // Editing an existing note -> update in place (PUT); otherwise create
      // (POST). The mutation invalidates the notes library on success.
      const savedNote = await saveNoteMutation.mutateAsync({
        title: noteTitle.trim(),
        content: generatedNotes,
        sourceFiles: selectedFiles,
        topic,
        noteId: currentNoteId,
      })
      setCurrentNoteId(savedNote.id)
    } catch (error: unknown) {
      console.error('Failed to save notes:', error)
      setErrMsg(error instanceof Error ? error.message : 'Failed to save notes.')
    }
  }

  const removeNote = async (noteId: string) => {
    try {
      await deleteNoteMutation.mutateAsync(noteId)
      if (currentNoteId === noteId) {
        setCurrentNoteId(undefined)
        setGeneratedNotes('')
        setNoteTitle('')
        setFlashcards([])
      }
    } catch (error) {
      showError(error instanceof Error ? error.message : 'Failed to delete note.')
    }
  }

  const openNote = (note: SavedNote) => {
    setGeneratedNotes(note.content || '')
    setNoteTitle(note.title || '')
    setCurrentNoteId(note.id)
    setSelectedFiles(note.source_files || [])
    setFlashcards(parseFlashcardsFromText(note.content || ''))
    setLibraryOpen(false)
  }

  const reset = () => {
    setGeneratedNotes('')
    setNoteTitle('')
    setCurrentNoteId(undefined)
    setTopic('')
    setSelectedFiles([])
    setErrMsg(null)
    setFlashcards([])
  }

  const allSelected = availableFiles.length > 0 && selectedFiles.length === availableFiles.length
  const toggleSelectAll = () => {
    setSelectedFiles(allSelected ? [] : [...availableFiles])
  }
  const toggleFile = (file: string) => {
    setSelectedFiles((prev) =>
      prev.includes(file) ? prev.filter((f) => f !== file) : [...prev, file],
    )
  }
  const usingAllFiles = selectedFiles.length === 0 || selectedFiles.length === availableFiles.length

  return {
    courseId,
    userId,
    filesQuery,
    notesQuery,
    availableFiles,
    savedNotes,
    saving,
    selectedFiles,
    allSelected,
    usingAllFiles,
    toggleFile,
    toggleSelectAll,
    topic,
    setTopic,
    noteStyle,
    setNoteStyle,
    generatedNotes,
    setGeneratedNotes,
    noteTitle,
    setNoteTitle,
    currentNoteId,
    flashcards,
    loading,
    loadStage,
    errMsg,
    libraryOpen,
    setLibraryOpen,
    generate,
    save,
    removeNote,
    openNote,
    reset,
  }
}

export type NotesStudio = ReturnType<typeof useNotesStudio>
