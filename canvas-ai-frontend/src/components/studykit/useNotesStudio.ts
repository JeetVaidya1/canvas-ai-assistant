// src/components/studykit/useNotesStudio.ts — all Study Kit state + server interactions
import { useEffect, useMemo, useRef, useState } from 'react'
import { generateNotes, type NotesResponse, type SavedNote } from '@/lib/api'
import { showError, showInfo } from '@/lib/toast'
import { noteDraftKey, readJson, removeKey, writeJson } from '@/lib/resumeKeys'
import { useUser } from '@/hooks/useUser'
import { useCourseFiles } from '@/hooks/useCourseFiles'
import { useNotesLibrary, useSaveNote, useDeleteNote } from '@/hooks/useNotesLibrary'
import {
  LOADING_STAGES,
  parseFlashcardsFromText,
  type Flashcard,
  type NoteStyle,
} from './noteUtils'

/** Debounce for writing the composer draft to localStorage. */
const DRAFT_SAVE_MS = 500

const NOTE_STYLE_VALUES: readonly NoteStyle[] = ['detailed', 'summary', 'outline']

interface NoteDraft {
  topic: string
  style: NoteStyle
  files: string[]
}

/** Shape-check stored draft data — never trust localStorage contents. */
function parseNoteDraft(raw: unknown): NoteDraft | null {
  if (!raw || typeof raw !== 'object') return null
  const candidate = raw as Record<string, unknown>
  if (typeof candidate.topic !== 'string') return null
  if (!NOTE_STYLE_VALUES.includes(candidate.style as NoteStyle)) return null
  if (!Array.isArray(candidate.files)) return null
  return {
    topic: candidate.topic,
    style: candidate.style as NoteStyle,
    files: candidate.files.filter((f): f is string => typeof f === 'string'),
  }
}

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

  // Restore a persisted composer draft once per course. Runs after the reset
  // effect above (declaration order), so restore wins on mount/course switch.
  const draftReadyRef = useRef(false)
  useEffect(() => {
    draftReadyRef.current = false
    if (!courseId) return
    const draft = parseNoteDraft(readJson(noteDraftKey(courseId)))
    if (draft && (draft.topic.trim() || draft.style !== 'detailed' || draft.files.length > 0)) {
      setTopic(draft.topic)
      setNoteStyle(draft.style)
      if (draft.files.length > 0) setSelectedFiles(draft.files)
      showInfo('Draft restored')
    }
    // Only start persisting once restore had its chance (avoids clobbering).
    draftReadyRef.current = true
  }, [courseId])

  // Persist the composer draft (debounced). An empty composer clears the key
  // so stale drafts never linger. File selection alone is not persisted — it
  // auto-fills to "all files" and would create junk drafts for everyone.
  useEffect(() => {
    if (!courseId || !draftReadyRef.current) return
    const key = noteDraftKey(courseId)
    const timer = setTimeout(() => {
      if (topic.trim() || noteStyle !== 'detailed') {
        const draft: NoteDraft = { topic, style: noteStyle, files: [...selectedFiles] }
        writeJson(key, draft)
      } else {
        removeKey(key)
      }
    }, DRAFT_SAVE_MS)
    return () => clearTimeout(timer)
  }, [courseId, topic, noteStyle, selectedFiles])

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
      // Generation succeeded — the draft served its purpose.
      removeKey(noteDraftKey(courseId))
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
