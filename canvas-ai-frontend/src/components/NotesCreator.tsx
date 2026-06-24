// src/components/NotesCreator.tsx — GPT-5 aligned, robust (Saved Notes practice button always visible)
import { useState, useEffect, useMemo } from 'react'
import { Markdown } from '@/components/ui/Markdown'
import { Button } from '@/components/ui/Button'
import { Card, PageHeader } from '@/components/ui/Card'
import {
  FileText,
  Save,
  Download,
  Trash2,
  Edit3,
  BookOpen,
  Clock,
  Eye,
  Copy,
  Brain,
  Search,
  Bookmark,
  RotateCcw,
  X
} from 'lucide-react'
import {
  generateNotes,
  saveNotes as apiSaveNotes,
  updateNote as apiUpdateNote,
  getNotes as apiGetNotes,
  deleteNotes as apiDeleteNotes,
  listFiles,
  type NotesResponse
} from '../lib/api'

import Flashcards from './FlashCards'
import ConfirmDialog from './shared/ConfirmDialog'
import { useUser } from '@/hooks/useUser'

interface NotesCreatorProps {
  courseId: string
  courseName: string
}

interface SavedNote {
  id: string
  title: string
  content: string
  source_files: string[]
  created_at: string
  updated_at: string
  word_count: string | number
  reading_time: string
  topics: string[]
}

type Flashcard = { q: string; a: string }

type NoteStyle = 'detailed' | 'summary' | 'outline'

// Segmented control options for note generation style.
const NOTE_STYLES: { value: NoteStyle; label: string; hint: string }[] = [
  { value: 'detailed', label: 'Detailed', hint: 'Comprehensive, in-depth notes' },
  { value: 'summary', label: 'Summary', hint: 'Key points only' },
  { value: 'outline', label: 'Outline', hint: 'Structured outline format' },
]

/** More forgiving parser:
 * - Matches "FLASHCARDS" with or without "(JSON)" and ":".
 * - Supports fenced code blocks ```json ... ``` or raw arrays.
 * - Fallback: scans for the first JSON array that looks like [{q,a},...].
 */
const parseFlashcardsFromText = (text: string): { q: string; a: string }[] => {
  try {
    if (!text) return []

    // 1) Find the "FLASHCARDS" header anywhere (handles "## FLASHCARDS (JSON)" etc.)
    const headerMatch = /flashcards(?:\s*\(json\))?/i.exec(text)
    if (headerMatch) {
      // Start scanning after the header
      let i = headerMatch.index + headerMatch[0].length
      // Skip whitespace, optional colon, and more whitespace/newlines
      while (i < text.length && /\s|:/.test(text[i])) i++

      // 2) If fenced block starts, capture inside it
      if (text.slice(i, i + 3) === '```') {
        // Advance past ```
        i += 3
        // Optional "json"
        if (/^json/i.test(text.slice(i, i + 4))) i += 4
        // Skip whitespace/newlines
        while (i < text.length && /\s/.test(text[i])) i++
        // Find closing fence
        const fence = text.indexOf('```', i)
        const raw = fence !== -1 ? text.slice(i, fence) : text.slice(i)
        try {
          const cleaned = raw.replace(/,\s*([\]}])/g, '$1')
          const arr = JSON.parse(cleaned)
          if (Array.isArray(arr)) {
            return arr
              .map((it: any) => ({ q: (it?.q || '').trim(), a: (it?.a || '').trim() }))
              .filter((it: any) => it.q && it.a)
          }
        } catch {/* fall through */}
      }

      // 3) Otherwise, grab the first JSON array after the header using bracket balancing
      //    Find first '[' after the header zone
      let start = text.indexOf('[', i)
      if (start !== -1) {
        let depth = 0
        let end = -1
        for (let j = start; j < text.length; j++) {
          const ch = text[j]
          if (ch === '[') depth++
          else if (ch === ']') {
            depth--
            if (depth === 0) { end = j; break }
          }
        }
        if (end !== -1) {
          const raw = text.slice(start, end + 1)
          try {
            const cleaned = raw.replace(/,\s*([\]}])/g, '$1')
            const arr = JSON.parse(cleaned)
            if (Array.isArray(arr)) {
              return arr
                .map((it: any) => ({ q: (it?.q || '').trim(), a: (it?.a || '').trim() }))
                .filter((it: any) => it.q && it.a)
            }
          } catch {/* fall through */}
        }
      }
    }

    // 4) Last-chance fallback: first plausible array anywhere in doc
    const anyArr = text.match(/\[[\s\S]+?\]/)
    if (anyArr) {
      try {
        const cleaned = anyArr[0].replace(/,\s*([\]}])/g, '$1')
        const arr = JSON.parse(cleaned)
        if (Array.isArray(arr)) {
          return arr
            .map((it: any) => ({ q: (it?.q || '').trim(), a: (it?.a || '').trim() }))
            .filter((it: any) => it.q && it.a)
        }
      } catch {/* ignore */}
    }

    return []
  } catch {
    return []
  }
}

export default function NotesCreator({ courseId, courseName }: NotesCreatorProps) {
  const userId = useUser()
  const [selectedFiles, setSelectedFiles] = useState<string[]>([])
  const [availableFiles, setAvailableFiles] = useState<string[]>([])
  const [topic, setTopic] = useState('')
  const [noteStyle, setNoteStyle] = useState<NoteStyle>('detailed')
  const [generatedNotes, setGeneratedNotes] = useState<string>('')
  const [savedNotes, setSavedNotes] = useState<SavedNote[]>([])
  const [currentNoteId, setCurrentNoteId] = useState<string | undefined>(undefined)
  const [noteTitle, setNoteTitle] = useState('')
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [activeTab, setActiveTab] = useState<'create' | 'saved'>('create')
  const [searchTerm, setSearchTerm] = useState('')
  const [previewMode, setPreviewMode] = useState(true)
  const [errMsg, setErrMsg] = useState<string | null>(null)
  const [flashcards, setFlashcards] = useState<Flashcard[]>([])

  // Modal state for Saved tab
  const [showCardsFor, setShowCardsFor] = useState<string | null>(null)
  const [savedFlashcards, setSavedFlashcards] = useState<Flashcard[]>([])

  // Reset when course changes
  useEffect(() => {
    setSelectedFiles([])
    setGeneratedNotes('')
    setNoteTitle('')
    setCurrentNoteId(undefined)
    setTopic('')
    setErrMsg(null)
    setFlashcards([])
    setShowCardsFor(null)
    setSavedFlashcards([])
  }, [courseId])

  useEffect(() => {
    if (!courseId) return
    loadAvailableFiles()
    loadSavedNotes()
  }, [courseId])

  const loadAvailableFiles = async () => {
    try {
      const files = await listFiles(courseId)
      setAvailableFiles(files || [])
    } catch (error) {
      console.error('Failed to load files:', error)
      setAvailableFiles([])
    }
  }

  const loadSavedNotes = async () => {
    try {
      const notes = await apiGetNotes(courseId)
      setSavedNotes(notes || [])
    } catch (error) {
      console.error('Failed to load saved notes:', error)
      setSavedNotes([])
    }
  }

  const handleGenerateNotes = async () => {
    if (!courseId || selectedFiles.length === 0) return
    setLoading(true)
    setGeneratedNotes('')
    setErrMsg(null)
    setFlashcards([])

    try {
      const response: NotesResponse & { flashcards?: Flashcard[] } = await generateNotes(
        courseId,
        selectedFiles,
        topic,
        noteStyle
      )
      const notes = response.notes || ''
      setGeneratedNotes(notes)
      setNoteTitle(response.suggested_title || `Notes: ${topic || 'Lecture Summary'}`)
      const fc = Array.isArray(response.flashcards)
        ? response.flashcards
        : parseFlashcardsFromText(notes)
      setFlashcards(fc || [])
      setCurrentNoteId(undefined)
    } catch (error: any) {
      console.error('Failed to generate notes:', error)
      setErrMsg(error?.message || 'Failed to generate notes. Please try again.')
      setGeneratedNotes('Failed to generate notes. Please try again.')
      setFlashcards([])
    } finally {
      setLoading(false)
    }
  }

  const handleSaveNotes = async () => {
    if (!courseId || !generatedNotes.trim() || !noteTitle.trim()) return

    setSaving(true)
    setErrMsg(null)
    try {
      // Editing an existing note -> update in place (PUT); otherwise create (POST).
      const savedNote = currentNoteId
        ? await apiUpdateNote(currentNoteId, courseId, noteTitle.trim(), generatedNotes, selectedFiles, topic)
        : await apiSaveNotes(courseId, noteTitle.trim(), generatedNotes, selectedFiles, topic)
      await loadSavedNotes()
      setCurrentNoteId(savedNote.id)
    } catch (error: any) {
      console.error('Failed to save notes:', error)
      setErrMsg(error?.message || 'Failed to save notes.')
    } finally {
      setSaving(false)
    }
  }

  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null)

  const confirmDeleteNote = (noteId: string) => {
    setDeleteConfirmId(noteId)
  }

  const executeDeleteNote = async () => {
    if (!deleteConfirmId) return
    const noteId = deleteConfirmId
    setDeleteConfirmId(null)
    try {
      await apiDeleteNotes(noteId)
      await loadSavedNotes()
      if (currentNoteId === noteId) {
        setCurrentNoteId(undefined)
        setGeneratedNotes('')
        setNoteTitle('')
        setFlashcards([])
      }
    } catch (error) {
      console.error('Failed to delete note:', error)
    }
  }

  const loadNote = (note: SavedNote) => {
    setGeneratedNotes(note.content || '')
    setNoteTitle(note.title || '')
    setCurrentNoteId(note.id)
    setSelectedFiles(note.source_files || [])
    setFlashcards(parseFlashcardsFromText(note.content || ''))
    setActiveTab('create')
  }

  const exportNote = (note: SavedNote) => {
    const element = document.createElement('a')
    const file = new Blob([note.content || ''], { type: 'text/plain' })
    element.href = URL.createObjectURL(file)
    element.download = `${(note.title || 'notes').replace(/[^a-z0-9]/gi, '_').toLowerCase()}.txt`
    document.body.appendChild(element)
    element.click()
    document.body.removeChild(element)
  }

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text || '')
    } catch {
      // non-blocking
    }
  }

  const clearConversations = () => {
    setGeneratedNotes('')
    setNoteTitle('')
    setCurrentNoteId(undefined)
    setTopic('')
    setSelectedFiles([])
    setErrMsg(null)
    setFlashcards([])
    setShowCardsFor(null)
    setSavedFlashcards([])
  }

  const filteredNotes = useMemo(() => {
    const q = (searchTerm || '').toLowerCase()
    return (savedNotes || []).filter(note =>
      (note.title || '').toLowerCase().includes(q) ||
      (note.topics || []).some(t => (t || '').toLowerCase().includes(q))
    )
  }, [savedNotes, searchTerm])

  const getWordCount = (text: string) => (text.trim() ? text.trim().split(/\s+/).length : 0)
  const getReadingTime = (wc: number) => `${Math.max(1, Math.ceil(wc / 200))} min read`

  const renderCreateTab = () => (
    <div className="space-y-8">
      {/* File Selection */}
      <Card accent padding="lg">
        <h3 className="text-base font-semibold text-zinc-100 mb-4 flex items-center gap-3">
          <span className="w-9 h-9 rounded-xl bg-gradient-brand-soft border border-cyan-500/15 flex items-center justify-center flex-shrink-0">
            <FileText className="w-4 h-4 text-cyan-300" />
          </span>
          Select Lecture Files
        </h3>

        {!courseId ? (
          <div className="text-center py-8 text-amber-500">
            Select a course to choose files.
          </div>
        ) : availableFiles.length === 0 ? (
          <div className="text-center py-8 text-zinc-400">
            <BookOpen className="w-12 h-12 text-zinc-600 mx-auto mb-3" />
            <p>No files uploaded to this course yet.</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {availableFiles.map((file) => {
              const checked = selectedFiles.includes(file)
              return (
                <label
                  key={file}
                  className={`flex items-center p-3 border rounded-lg cursor-pointer transition-all ${
                    checked
                      ? 'bg-gradient-brand-soft border-cyan-500/30 text-cyan-300'
                      : 'bg-zinc-900 border-zinc-700 text-zinc-300 hover:bg-zinc-800 hover:border-zinc-600'
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={(e) => {
                      if (e.target.checked) setSelectedFiles(prev => [...prev, file])
                      else setSelectedFiles(prev => prev.filter(f => f !== file))
                    }}
                    className="mr-3 accent-cyan-500"
                  />
                  <FileText className="w-4 h-4 mr-2 opacity-70" />
                  <span className="text-sm truncate">{file}</span>
                </label>
              )
            })}
          </div>
        )}

        <div className="mt-4 text-sm text-zinc-500">
          {selectedFiles.length} file(s) selected
        </div>
      </Card>

      {/* Notes Configuration */}
      <Card accent padding="lg">
        <h3 className="text-base font-semibold text-zinc-100 mb-4 flex items-center gap-3">
          <span className="w-9 h-9 rounded-xl bg-gradient-brand-soft border border-cyan-500/15 flex items-center justify-center flex-shrink-0">
            <Brain className="w-4 h-4 text-cyan-300" />
          </span>
          Notes Configuration
        </h3>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="space-y-2">
            <label className="text-xs font-medium text-zinc-400">Specific Topic (Optional)</label>
            <input
              type="text"
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              placeholder="e.g., Binary Search Trees, Machine Learning Algorithms"
              className="w-full px-3 py-2.5 bg-zinc-800/70 border border-zinc-700 text-zinc-100 placeholder-zinc-600 rounded-lg text-sm outline-none focus:border-cyan-500/60 focus:ring-2 focus:ring-cyan-500/20 transition-colors"
            />
            <p className="text-xs text-zinc-500">Leave blank to generate notes for all content</p>
          </div>

          <div className="space-y-2">
            <label className="text-xs font-medium text-zinc-400">Note Style</label>
            <div className="flex items-center gap-1 bg-zinc-800/70 border border-zinc-700 p-1 rounded-lg">
              {NOTE_STYLES.map((style) => {
                const active = noteStyle === style.value
                return (
                  <button
                    key={style.value}
                    type="button"
                    onClick={() => setNoteStyle(style.value)}
                    className={`flex-1 px-3 py-2 rounded-md text-sm font-medium transition-all ${
                      active
                        ? 'bg-gradient-brand-soft text-cyan-300 border border-cyan-500/30'
                        : 'text-zinc-400 border border-transparent hover:text-zinc-200'
                    }`}
                  >
                    {style.label}
                  </button>
                )
              })}
            </div>
            <p className="text-xs text-zinc-500">{NOTE_STYLES.find(s => s.value === noteStyle)?.hint}</p>
          </div>
        </div>

        <Button
          size="lg"
          onClick={handleGenerateNotes}
          loading={loading}
          disabled={loading || selectedFiles.length === 0 || !courseId}
          leftIcon={<Brain className="w-5 h-5" />}
          className="mt-6"
        >
          {loading ? 'Generating Notes...' : 'Generate Notes'}
        </Button>

        {errMsg && (
          <div className="mt-4 text-sm text-red-400">{errMsg}</div>
        )}
      </Card>

      {/* Generated Notes */}
      {(generatedNotes || loading) && (
        <Card accent padding="none" className="overflow-hidden">
          <div className="bg-zinc-800/40 px-6 py-4 border-b border-zinc-800">
            <div className="flex items-center justify-between gap-3 flex-wrap">
              <div className="flex items-center gap-3 min-w-0">
                <Edit3 className="w-5 h-5 text-cyan-300 flex-shrink-0" />
                <input
                  type="text"
                  value={noteTitle}
                  onChange={(e) => setNoteTitle(e.target.value)}
                  placeholder="Enter note title..."
                  className="text-lg font-semibold bg-transparent border-none focus:outline-none focus:ring-0 text-zinc-50 placeholder-zinc-600 min-w-0"
                />
              </div>

              <div className="flex items-center gap-2">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setPreviewMode(!previewMode)}
                  title={previewMode ? 'Edit mode' : 'Preview mode'}
                  aria-label={previewMode ? 'Switch to edit mode' : 'Switch to preview mode'}
                  className="!px-2"
                >
                  {previewMode ? <Edit3 className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </Button>

                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => copyToClipboard(generatedNotes)}
                  title="Copy notes"
                  aria-label="Copy notes"
                  className="!px-2"
                >
                  <Copy className="w-4 h-4" />
                </Button>

                <Button
                  onClick={handleSaveNotes}
                  loading={saving}
                  disabled={saving || !generatedNotes.trim() || !noteTitle.trim()}
                  leftIcon={<Save className="w-4 h-4" />}
                >
                  {saving
                    ? (currentNoteId ? 'Updating...' : 'Saving...')
                    : (currentNoteId ? 'Update Note' : 'Save')}
                </Button>
              </div>
            </div>
          </div>

          <div className="p-6">
            {loading ? (
              <div className="text-center py-12">
                <div className="w-16 h-16 border-4 border-cyan-500/20 border-t-cyan-400 rounded-full animate-spin mx-auto mb-4"></div>
                <h3 className="text-lg font-semibold text-zinc-200 mb-2">Generating Your Notes...</h3>
                <p className="text-zinc-500">Analyzing {selectedFiles.length} file(s) and creating detailed notes</p>
              </div>
            ) : previewMode ? (
              <div className="prose prose-lg prose-invert max-w-none">
                <Markdown content={generatedNotes} className="text-zinc-300" />
              </div>
            ) : (
              <textarea
                value={generatedNotes}
                onChange={(e) => setGeneratedNotes(e.target.value)}
                className="w-full h-96 p-4 bg-zinc-800/70 border border-zinc-700 text-zinc-100 rounded-lg outline-none focus:border-cyan-500/60 focus:ring-2 focus:ring-cyan-500/20 resize-none font-mono text-sm transition-colors"
                placeholder="Your generated notes will appear here..."
              />
            )}

            {/* Create tab flashcards */}
            {flashcards.length > 0 && (
              <div className="mt-8">
                <h3 className="text-base font-semibold text-zinc-100 mb-3 flex items-center gap-2">
                  <BookOpen className="w-5 h-5 text-cyan-300" />
                  Practice Flashcards
                  <span className="ml-2 text-xs font-medium text-cyan-300 bg-gradient-brand-soft border border-cyan-500/20 px-2 py-0.5 rounded-full">
                    {flashcards.length}
                  </span>
                </h3>
                <Flashcards cards={flashcards} courseId={courseId} userId={userId} />
              </div>
            )}

            {generatedNotes && (
              <div className="mt-6 pt-4 border-t border-zinc-800 flex items-center gap-6 text-sm text-zinc-500">
                <div className="flex items-center gap-2">
                  <Clock className="w-4 h-4" />
                  <span>{getReadingTime(getWordCount(generatedNotes))}</span>
                </div>
                <div className="flex items-center gap-2">
                  <FileText className="w-4 h-4" />
                  <span>{getWordCount(generatedNotes)} words</span>
                </div>
                <div className="flex items-center gap-2">
                  <BookOpen className="w-4 h-4" />
                  <span>{selectedFiles.length} source file(s)</span>
                </div>
              </div>
            )}
          </div>
        </Card>
      )}
    </div>
  )

  const renderSavedTab = () => (
    <div className="space-y-6">
      {/* Search and Count */}
      <Card padding="md">
        <div className="flex items-center gap-4">
          <div className="flex-1 relative">
            <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-zinc-500" />
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Search your notes..."
              className="w-full pl-10 pr-4 py-2.5 bg-zinc-800/70 border border-zinc-700 text-zinc-100 placeholder-zinc-600 rounded-lg text-sm outline-none focus:border-cyan-500/60 focus:ring-2 focus:ring-cyan-500/20 transition-colors"
            />
          </div>
          <div className="text-sm text-zinc-500">
            {filteredNotes.length} note(s)
          </div>
        </div>
      </Card>

      {/* Notes List */}
      {filteredNotes.length === 0 ? (
        <Card padding="none" className="py-16 px-8 text-center">
          <div className="w-14 h-14 rounded-2xl bg-gradient-brand-soft border border-cyan-500/20 flex items-center justify-center mx-auto mb-5">
            <BookOpen className="w-7 h-7 text-cyan-300" />
          </div>
          <h3 className="text-lg font-semibold text-zinc-100 mb-2">
            {searchTerm ? 'No notes found' : 'No saved notes yet'}
          </h3>
          <p className="text-sm text-zinc-500 mb-6">
            {searchTerm ? 'Try adjusting your search terms' : 'Generate your first set of notes to get started'}
          </p>
          {!searchTerm && (
            <Button onClick={() => setActiveTab('create')}>
              Create Your First Notes
            </Button>
          )}
        </Card>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {filteredNotes.map((note) => {
            const cardsCount = parseFlashcardsFromText(note.content || '').length

            return (
              <Card key={note.id} accent padding="lg" className="group">
                <div className="flex items-start justify-between mb-4 gap-3">
                  <div className="flex-1 min-w-0">
                    <h3 className="text-base font-semibold text-zinc-100 mb-2 truncate">{note.title}</h3>
                    <div className="flex items-center gap-4 text-sm text-zinc-500 mb-3 flex-wrap">
                      <div className="flex items-center gap-1.5">
                        <Clock className="w-4 h-4" />
                        <span>{note.reading_time}</span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <FileText className="w-4 h-4" />
                        <span>{note.word_count} words</span>
                      </div>
                      <span className={`px-2 py-0.5 text-xs rounded-full ${cardsCount > 0 ? 'bg-gradient-brand-soft border border-cyan-500/20 text-cyan-300' : 'bg-zinc-800 text-zinc-400'}`}>
                        {cardsCount} cards
                      </span>
                    </div>
                  </div>

                  <div className="flex items-center gap-1 flex-shrink-0">
                    <button
                      onClick={() => loadNote(note)}
                      className="p-2 text-cyan-300 hover:bg-cyan-500/10 rounded-lg transition-colors"
                      title="Open note"
                      aria-label="Open note"
                    >
                      <Edit3 className="w-4 h-4" />
                    </button>

                    {/* Always show the Practice button */}
                    <button
                      onClick={() => {
                        setSavedFlashcards(parseFlashcardsFromText(note.content || ''))
                        setShowCardsFor(note.id)
                      }}
                      className="p-2 text-cyan-300 hover:bg-cyan-500/10 rounded-lg transition-colors"
                      title="Practice flashcards"
                      aria-label="Practice flashcards"
                    >
                      <BookOpen className="w-4 h-4" />
                    </button>

                    <button
                      onClick={() => exportNote(note)}
                      className="p-2 text-emerald-400 hover:bg-emerald-500/10 rounded-lg transition-colors"
                      title="Download note"
                      aria-label="Download note"
                    >
                      <Download className="w-4 h-4" />
                    </button>

                    <button
                      onClick={() => confirmDeleteNote(note.id)}
                      className="p-2 text-zinc-500 hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-colors"
                      title="Delete note"
                      aria-label="Delete note"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>

                <div className="text-zinc-400 text-sm mb-4 line-clamp-3">
                  {(note.content || '').slice(0, 200)}...
                </div>

                <div className="flex items-center justify-between gap-3">
                  <div className="flex flex-wrap gap-2">
                    {(note.topics || []).slice(0, 3).map((t, i) => (
                      <span
                        key={i}
                        className="px-2 py-1 bg-gradient-brand-soft border border-cyan-500/20 text-cyan-300 text-xs rounded-full"
                      >
                        {t}
                      </span>
                    ))}
                    {note.topics && note.topics.length > 3 && (
                      <span className="px-2 py-1 bg-zinc-800 text-zinc-400 text-xs rounded-full">
                        +{note.topics.length - 3} more
                      </span>
                    )}
                  </div>

                  <div className="text-xs text-zinc-600 flex-shrink-0">
                    {new Date(note.created_at).toLocaleDateString()}
                  </div>
                </div>
              </Card>
            )
          })}
        </div>
      )}

      {/* Flashcards Modal for Saved Notes */}
      {showCardsFor && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" onClick={() => { setShowCardsFor(null); setSavedFlashcards([]) }} />
          <Card padding="lg" className="relative z-10 w-full max-w-3xl space-y-4 shadow-2xl !bg-zinc-900">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-semibold text-zinc-100 flex items-center gap-2">
                <BookOpen className="w-5 h-5 text-cyan-300" />
                Practice Flashcards
                <span className="ml-2 text-xs font-medium text-cyan-300 bg-gradient-brand-soft border border-cyan-500/20 px-2 py-0.5 rounded-full">
                  {savedFlashcards.length}
                </span>
              </h3>
              <button
                onClick={() => { setShowCardsFor(null); setSavedFlashcards([]) }}
                className="p-1.5 text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800 rounded-lg transition-colors"
                aria-label="Close"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {savedFlashcards.length > 0 ? (
              <Flashcards cards={savedFlashcards} courseId={courseId} userId={userId} />
            ) : (
              <div className="text-zinc-400">No flashcards found in this note.</div>
            )}
          </Card>
        </div>
      )}
    </div>
  )

  return (
    <div className="max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <PageHeader
          eyebrow="Notes"
          title="Notes Creator"
          subtitle={`Generate comprehensive notes from your ${courseName} lectures`}
          className="mb-5"
          actions={
            (generatedNotes || selectedFiles.length > 0) ? (
              <Button
                variant="secondary"
                size="sm"
                onClick={clearConversations}
                leftIcon={<RotateCcw className="w-4 h-4" />}
              >
                Clear All
              </Button>
            ) : undefined
          }
        />

        {/* Tab Navigation */}
        <div className="flex items-center gap-1 bg-zinc-800/70 border border-zinc-700 p-1 rounded-lg w-fit">
          <button
            onClick={() => setActiveTab('create')}
            className={`px-5 py-2 rounded-md text-sm font-medium transition-all ${
              activeTab === 'create'
                ? 'bg-gradient-brand-soft text-cyan-300 border border-cyan-500/30'
                : 'text-zinc-400 border border-transparent hover:text-zinc-200'
            }`}
          >
            <Brain className="w-4 h-4 inline mr-2" />
            Create Notes
          </button>
          <button
            onClick={() => setActiveTab('saved')}
            className={`px-5 py-2 rounded-md text-sm font-medium transition-all ${
              activeTab === 'saved'
                ? 'bg-gradient-brand-soft text-cyan-300 border border-cyan-500/30'
                : 'text-zinc-400 border border-transparent hover:text-zinc-200'
            }`}
          >
            <Bookmark className="w-4 h-4 inline mr-2" />
            Saved Notes ({savedNotes.length})
          </button>
        </div>
      </div>

      {/* Tab Content */}
      {activeTab === 'create' ? renderCreateTab() : renderSavedTab()}

      <ConfirmDialog
        open={!!deleteConfirmId}
        title="Delete Note"
        description="Delete this note? This cannot be undone."
        confirmLabel="Delete"
        variant="danger"
        onConfirm={executeDeleteNote}
        onCancel={() => setDeleteConfirmId(null)}
      />
    </div>
  )
}
