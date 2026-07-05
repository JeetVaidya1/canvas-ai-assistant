// src/components/NotesCreator.tsx — "Notes Studio": grounded notes + auto-generated flashcard deck
import { useState, useEffect, useMemo } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { Markdown } from '@/components/ui/Markdown'
import { Button } from '@/components/ui/Button'
import { Card } from '@/components/ui/Card'
import { cn } from '@/lib/utils'
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
  X,
  Sparkles,
  Layers,
  ChevronLeft,
  ChevronRight,
  ChevronDown,
  Check,
} from 'lucide-react'
import { BrandMark } from '@/components/ui/BrandMark'
import {
  generateNotes,
  type NotesResponse,
  type SavedNote,
} from '../lib/api'
import { showError } from '../lib/toast'

import Flashcards from './FlashCards'
import ConfirmDialog from './shared/ConfirmDialog'
import ErrorInline from './shared/ErrorInline'
import { useUser } from '@/hooks/useUser'
import { useCourseFiles } from '@/hooks/useCourseFiles'
import { useNotesLibrary, useSaveNote, useDeleteNote } from '@/hooks/useNotesLibrary'

interface NotesCreatorProps {
  courseId: string
  courseName: string
}

type Flashcard = { q: string; a: string }

type NoteStyle = 'detailed' | 'summary' | 'outline'

// Segmented control options for note generation style.
const NOTE_STYLES: { value: NoteStyle; label: string; hint: string }[] = [
  { value: 'detailed', label: 'Detailed', hint: 'Comprehensive, in-depth notes' },
  { value: 'summary', label: 'Summary', hint: 'Key points only' },
  { value: 'outline', label: 'Outline', hint: 'Structured outline format' },
]

const LOADING_STAGES = [
  'Reading your source files…',
  'Grounding key concepts in your material…',
  'Structuring comprehensive notes…',
  'Generating study flashcards…',
]

/** Forgiving parser kept ONLY as a fallback for legacy saved notes whose
 *  flashcards were embedded inline. New generations use response.flashcards. */
const parseFlashcardsFromText = (text: string): Flashcard[] => {
  try {
    if (!text) return []
    const headerMatch = /flashcards(?:\s*\(json\))?/i.exec(text)
    if (headerMatch) {
      let i = headerMatch.index + headerMatch[0].length
      while (i < text.length && /\s|:/.test(text[i])) i++
      if (text.slice(i, i + 3) === '```') {
        i += 3
        if (/^json/i.test(text.slice(i, i + 4))) i += 4
        while (i < text.length && /\s/.test(text[i])) i++
        const fence = text.indexOf('```', i)
        const raw = fence !== -1 ? text.slice(i, fence) : text.slice(i)
        try {
          const arr = JSON.parse(raw.replace(/,\s*([\]}])/g, '$1'))
          if (Array.isArray(arr)) {
            return arr
              .map((it) => ({ q: String(it?.q || '').trim(), a: String(it?.a || '').trim() }))
              .filter((it) => it.q && it.a)
          }
        } catch {/* fall through */}
      }
      const start = text.indexOf('[', i)
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
          try {
            const arr = JSON.parse(text.slice(start, end + 1).replace(/,\s*([\]}])/g, '$1'))
            if (Array.isArray(arr)) {
              return arr
                .map((it) => ({ q: String(it?.q || '').trim(), a: String(it?.a || '').trim() }))
                .filter((it) => it.q && it.a)
            }
          } catch {/* fall through */}
        }
      }
    }
    const anyArr = text.match(/\[[\s\S]+?\]/)
    if (anyArr) {
      try {
        const arr = JSON.parse(anyArr[0].replace(/,\s*([\]}])/g, '$1'))
        if (Array.isArray(arr)) {
          return arr
            .map((it) => ({ q: String(it?.q || '').trim(), a: String(it?.a || '').trim() }))
            .filter((it) => it.q && it.a)
        }
      } catch {/* ignore */}
    }
    return []
  } catch {
    return []
  }
}

/** A single flippable study card — the centerpiece of the deck. */
function FlipCard({ card, flipped, onFlip }: { card: Flashcard; flipped: boolean; onFlip: () => void }) {
  return (
    <button
      type="button"
      onClick={onFlip}
      aria-label={flipped ? 'Show question' : 'Reveal answer'}
      className="group relative w-full h-72 [perspective:1600px] outline-none"
    >
      <motion.div
        className="relative w-full h-full [transform-style:preserve-3d]"
        animate={{ rotateY: flipped ? 180 : 0 }}
        transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
      >
        {/* Front — Question */}
        <div className="absolute inset-0 [backface-visibility:hidden] rounded-2xl card-surface accent-top glow-brand p-7 flex flex-col">
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-widest text-gradient-brand">
            <Brain className="w-3.5 h-3.5" /> Question
          </div>
          <div className="flex-1 flex items-center justify-center text-center px-2">
            <p className="text-lg font-medium text-zinc-100 leading-relaxed">{card.q}</p>
          </div>
          <p className="text-xs text-zinc-500 text-center">Tap to reveal answer</p>
        </div>
        {/* Back — Answer */}
        <div className="absolute inset-0 [backface-visibility:hidden] [transform:rotateY(180deg)] rounded-2xl bg-gradient-brand-soft border border-cyan-400/25 p-7 flex flex-col">
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-widest text-cyan-300">
            <Sparkles className="w-3.5 h-3.5" /> Answer
          </div>
          <div className="flex-1 flex items-center justify-center text-center overflow-auto px-2">
            <div className="prose prose-invert prose-sm max-w-none text-zinc-100">
              <Markdown content={card.a} />
            </div>
          </div>
          <p className="text-xs text-cyan-300/60 text-center">Tap to flip back</p>
        </div>
      </motion.div>
    </button>
  )
}

/** Prominent deck presentation with navigation + progress — the auto-generated study tool. */
function FlashcardDeck({ cards }: { cards: Flashcard[] }) {
  const [index, setIndex] = useState(0)
  const [flipped, setFlipped] = useState(false)
  const [seen, setSeen] = useState<Set<number>>(() => new Set([0]))

  useEffect(() => {
    setIndex(0)
    setFlipped(false)
    setSeen(new Set([0]))
  }, [cards])

  if (cards.length === 0) return null
  const safeIndex = Math.min(index, cards.length - 1)

  const go = (next: number) => {
    const clamped = Math.max(0, Math.min(cards.length - 1, next))
    setFlipped(false)
    setIndex(clamped)
    setSeen((prev) => new Set(prev).add(clamped))
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <div className="flex items-center gap-2 text-sm text-zinc-400">
          <Layers className="w-4 h-4 text-cyan-300" />
          Card <span className="text-zinc-100 font-semibold">{safeIndex + 1}</span> of {cards.length}
        </div>
        <div className="flex items-center gap-1.5">
          {cards.map((_, i) => (
            <span
              key={i}
              className={`h-1.5 rounded-full transition-all ${
                i === safeIndex
                  ? 'w-6 bg-gradient-brand'
                  : seen.has(i)
                  ? 'w-1.5 bg-cyan-400/50'
                  : 'w-1.5 bg-white/15'
              }`}
            />
          ))}
        </div>
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={safeIndex}
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -12 }}
          transition={{ duration: 0.25 }}
        >
          <FlipCard card={cards[safeIndex]} flipped={flipped} onFlip={() => setFlipped((f) => !f)} />
        </motion.div>
      </AnimatePresence>

      <div className="flex items-center justify-between gap-3">
        <Button
          variant="secondary"
          size="sm"
          onClick={() => go(safeIndex - 1)}
          disabled={safeIndex === 0}
          leftIcon={<ChevronLeft className="w-4 h-4" />}
        >
          Previous
        </Button>
        <span className="text-xs text-zinc-500">{seen.size}/{cards.length} reviewed</span>
        <Button
          variant="secondary"
          size="sm"
          onClick={() => go(safeIndex + 1)}
          disabled={safeIndex === cards.length - 1}
          rightIcon={<ChevronRight className="w-4 h-4" />}
        >
          Next
        </Button>
      </div>
    </div>
  )
}

export default function NotesCreator({ courseId }: NotesCreatorProps) {
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
  const [generatedNotes, setGeneratedNotes] = useState<string>('')
  const [currentNoteId, setCurrentNoteId] = useState<string | undefined>(undefined)
  const [noteTitle, setNoteTitle] = useState('')
  const [loading, setLoading] = useState(false)
  const [loadStage, setLoadStage] = useState(0)
  const [libraryOpen, setLibraryOpen] = useState(false)
  const [sourcesOpen, setSourcesOpen] = useState(false)
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
    setSourcesOpen(false)
    setLibraryOpen(false)
  }, [courseId])

  // Default to ALL files selected so the common path needs zero file-clicks.
  // Only auto-fill when nothing is selected yet (don't clobber a viewed note's set).
  useEffect(() => {
    const list = filesQuery.data
    if (!list || list.length === 0) return
    setSelectedFiles((prev) => (prev.length === 0 ? [...list] : prev))
  }, [filesQuery.data])

  // Advance the loading narrative while generating.
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

  const handleGenerateNotes = async () => {
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
        noteStyle
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

  const handleSaveNotes = async () => {
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

  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null)

  const confirmDeleteNote = (noteId: string) => {
    setDeleteConfirmId(noteId)
  }

  const executeDeleteNote = async () => {
    if (!deleteConfirmId) return
    const noteId = deleteConfirmId
    setDeleteConfirmId(null)
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

  const loadNote = (note: SavedNote) => {
    setGeneratedNotes(note.content || '')
    setNoteTitle(note.title || '')
    setCurrentNoteId(note.id)
    setSelectedFiles(note.source_files || [])
    setFlashcards(parseFlashcardsFromText(note.content || ''))
    setLibraryOpen(false)
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

  // ===== Create tab =====
  const allSelected = availableFiles.length > 0 && selectedFiles.length === availableFiles.length
  const toggleSelectAll = () => {
    setSelectedFiles(allSelected ? [] : [...availableFiles])
  }
  const toggleFile = (file: string) => {
    setSelectedFiles((prev) =>
      prev.includes(file) ? prev.filter((f) => f !== file) : [...prev, file]
    )
  }

  // ── Center-first studio (mirrors ChatPage's empty state) ──────────────
  const usingAllFiles = selectedFiles.length === 0 || selectedFiles.length === availableFiles.length
  const noFiles = availableFiles.length === 0

  const renderStudio = () => (
    <div className="flex min-h-full flex-col items-center justify-center px-4 py-10">
      <motion.div
        initial={{ opacity: 0, y: 14 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
        className="w-full max-w-2xl"
      >
        {/* Heading */}
        <div className="mb-6 text-center">
          <BrandMark className="mx-auto mb-5 h-14 w-14 glow-brand" />
          <h1 className="text-[28px] font-semibold tracking-tight text-zinc-50">
            Create study notes
          </h1>
          <p className="mx-auto mt-2 max-w-md text-sm text-zinc-400">
            Grounded in your materials, with auto-generated flashcards.
          </p>
        </div>

        {/* Centerpiece: focus-topic input (styled like the chat composer) */}
        <div
          className={cn(
            'relative flex w-full items-center rounded-[20px] border border-white/12 bg-white/[0.03] p-2 shadow-lg transition-all',
            'focus-within:border-cyan-400/60 focus-within:bg-white/[0.05] focus-within:glow-brand-sm',
          )}
        >
          <span className="pl-3 pr-1 text-cyan-300/70">
            <Sparkles className="h-5 w-5" />
          </span>
          <input
            type="text"
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !noFiles && !loading) handleGenerateNotes()
            }}
            placeholder="What should these notes cover?  (e.g. Binary Search Trees — or leave blank for the whole course)"
            className="flex-1 bg-transparent px-2 py-2.5 text-[15px] text-zinc-100 placeholder-zinc-500 outline-none"
          />
        </div>

        {/* Style — inline segmented pills */}
        <div className="mt-4 flex items-center justify-center gap-2">
          {NOTE_STYLES.map((style) => {
            const active = noteStyle === style.value
            return (
              <button
                key={style.value}
                type="button"
                onClick={() => setNoteStyle(style.value)}
                title={style.hint}
                className={cn(
                  'rounded-full border px-4 py-1.5 text-[13px] font-medium transition-all',
                  active
                    ? 'bg-gradient-brand-soft border-cyan-400/40 text-cyan-100 ring-1 ring-inset ring-cyan-400/30'
                    : 'border-white/10 bg-white/[0.02] text-zinc-300 hover:border-white/20 hover:bg-white/[0.05] hover:text-zinc-100',
                )}
              >
                {style.label}
              </button>
            )
          })}
        </div>

        {/* Sources — single collapsible control, defaults to ALL selected */}
        <div className="mt-4 flex justify-center">
          <button
            type="button"
            onClick={() => setSourcesOpen((v) => !v)}
            disabled={noFiles}
            className="inline-flex items-center gap-1.5 rounded-full border border-white/10 bg-white/[0.03] px-3.5 py-1.5 text-[13px] text-zinc-300 transition-colors hover:border-cyan-400/40 hover:bg-white/[0.06] hover:text-zinc-100 disabled:opacity-50"
          >
            <FileText className="h-3.5 w-3.5 text-cyan-300/80" />
            {noFiles
              ? 'No files in this course'
              : usingAllFiles
                ? `Using all ${availableFiles.length} file${availableFiles.length === 1 ? '' : 's'}`
                : `Using ${selectedFiles.length} of ${availableFiles.length} files`}
            {!noFiles && <ChevronDown className={cn('h-3.5 w-3.5 transition-transform', sourcesOpen && 'rotate-180')} />}
          </button>
        </div>

        <AnimatePresence initial={false}>
          {sourcesOpen && !noFiles && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.2 }}
              className="overflow-hidden"
            >
              <Card padding="md" className="mt-3">
                <div className="mb-3 flex items-center justify-between">
                  <span className="text-xs font-medium text-zinc-400">Pick the files to ground these notes in</span>
                  <button
                    type="button"
                    onClick={toggleSelectAll}
                    className="text-xs font-medium text-cyan-300 transition-colors hover:text-cyan-200"
                  >
                    {allSelected ? 'Clear' : 'Select all'}
                  </button>
                </div>
                <div className="grid max-h-72 grid-cols-1 gap-1.5 overflow-auto pr-1 sm:grid-cols-2">
                  {availableFiles.map((file) => {
                    const checked = selectedFiles.includes(file)
                    return (
                      <button
                        type="button"
                        key={file}
                        onClick={() => toggleFile(file)}
                        title={file}
                        aria-pressed={checked}
                        className={cn(
                          'group flex items-center gap-2.5 rounded-lg border px-3 py-2 text-left transition-all',
                          checked
                            ? 'border-cyan-400/40 bg-gradient-brand-soft'
                            : 'border-white/10 bg-white/[0.02] hover:border-white/20 hover:bg-white/[0.05]',
                        )}
                      >
                        <span
                          className={cn(
                            'flex h-[18px] w-[18px] flex-shrink-0 items-center justify-center rounded-[5px] border transition-colors',
                            checked ? 'border-transparent bg-gradient-brand' : 'border-white/20 group-hover:border-white/40',
                          )}
                        >
                          {checked && <Check className="h-3 w-3 text-white" />}
                        </span>
                        <span className={cn('truncate text-sm', checked ? 'font-medium text-zinc-100' : 'text-zinc-300')}>
                          {file}
                        </span>
                      </button>
                    )
                  })}
                </div>
              </Card>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Primary action */}
        <Button
          size="lg"
          onClick={handleGenerateNotes}
          loading={loading}
          disabled={loading || noFiles || !courseId}
          leftIcon={<Sparkles className="w-5 h-5" />}
          className="mt-5 w-full"
        >
          {loading ? 'Generating…' : 'Generate notes'}
        </Button>

        {filesQuery.isError && (
          <ErrorInline
            message="Couldn't load your course files."
            onRetry={() => void filesQuery.refetch()}
            className="mt-3"
          />
        )}
        {noFiles && courseId && !filesQuery.isError && (
          <p className="mt-2.5 text-center text-xs text-zinc-500">
            Upload course files from Materials to generate notes.
          </p>
        )}
        {!courseId && (
          <p className="mt-2.5 text-center text-xs text-amber-400">Select a course to get started.</p>
        )}
        {errMsg && <div className="mt-3 text-center text-sm text-rose-400">{errMsg}</div>}
      </motion.div>
    </div>
  )

  const renderLoading = () => (
    <Card accent padding="lg" className="glow-brand">
      <div className="text-center py-10">
        <div className="relative w-20 h-20 mx-auto mb-6">
          <div className="absolute inset-0 rounded-full bg-gradient-brand opacity-20 blur-xl animate-pulse" />
          <div className="absolute inset-0 border-4 border-cyan-400/15 border-t-cyan-400 rounded-full animate-spin" />
          <div className="absolute inset-0 flex items-center justify-center">
            <Sparkles className="w-7 h-7 text-cyan-300" />
          </div>
        </div>
        <h3 className="text-lg font-semibold text-zinc-100 mb-2">Crafting your study kit</h3>
        <AnimatePresence mode="wait">
          <motion.p
            key={loadStage}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            transition={{ duration: 0.3 }}
            className="text-sm text-cyan-300/80"
          >
            {LOADING_STAGES[loadStage]}
          </motion.p>
        </AnimatePresence>
        <p className="text-xs text-zinc-500 mt-4">Analyzing {selectedFiles.length} file(s)</p>

        <div className="mt-6 max-w-md mx-auto space-y-2">
          {[0, 1, 2].map((i) => (
            <div key={i} className="h-3 rounded bg-white/[0.04] overflow-hidden">
              <motion.div
                className="h-full bg-gradient-brand-soft"
                initial={{ width: '0%' }}
                animate={{ width: ['0%', '90%', '60%'] }}
                transition={{ duration: 3, repeat: Infinity, delay: i * 0.3, ease: 'easeInOut' }}
              />
            </div>
          ))}
        </div>
      </div>
    </Card>
  )

  const renderReader = () => (
    <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
      {/* Notes reader */}
      <Card accent padding="none" className="overflow-hidden xl:col-span-2">
        <div className="bg-white/[0.03] px-6 py-4 border-b border-white/10">
          <div className="flex items-center justify-between gap-3 flex-wrap">
            <div className="flex items-center gap-3 min-w-0 flex-1">
              <Edit3 className="w-5 h-5 text-cyan-300 flex-shrink-0" />
              <input
                type="text"
                value={noteTitle}
                onChange={(e) => setNoteTitle(e.target.value)}
                placeholder="Enter note title…"
                className="text-lg font-semibold bg-transparent border-none focus:outline-none focus:ring-0 text-zinc-50 placeholder-zinc-500 min-w-0 w-full"
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
                  ? (currentNoteId ? 'Updating…' : 'Saving…')
                  : (currentNoteId ? 'Update' : 'Save')}
              </Button>
            </div>
          </div>
        </div>

        <div className="p-6">
          {previewMode ? (
            <div className="prose prose-lg prose-invert max-w-none">
              <Markdown content={generatedNotes} className="text-zinc-300" />
            </div>
          ) : (
            <textarea
              value={generatedNotes}
              onChange={(e) => setGeneratedNotes(e.target.value)}
              className="w-full h-[28rem] p-4 bg-white/[0.03] border border-white/10 text-zinc-100 rounded-lg outline-none focus:border-cyan-400/60 focus:ring-2 focus:ring-cyan-500/25 resize-none font-mono text-sm transition-colors"
              placeholder="Your generated notes will appear here…"
            />
          )}

          {generatedNotes && (
            <div className="mt-6 pt-4 border-t border-white/10 flex items-center gap-6 text-sm text-zinc-400 flex-wrap">
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

      {/* Flashcard deck — prominent study tool */}
      <div className="xl:col-span-1 space-y-6">
        {flashcards.length > 0 ? (
          <Card accent padding="lg" className="sticky top-6">
            <h3 className="text-base font-semibold text-zinc-100 mb-1 flex items-center gap-2">
              <Layers className="w-5 h-5 text-cyan-300" />
              Flashcard Deck
              <span className="ml-1 text-xs font-medium text-cyan-300 bg-gradient-brand-soft border border-cyan-400/20 px-2 py-0.5 rounded-full">
                {flashcards.length}
              </span>
            </h3>
            <p className="text-xs text-zinc-400 mb-5">Auto-generated from your notes. Flip to study.</p>
            <FlashcardDeck cards={flashcards} />
          </Card>
        ) : (
          <Card padding="lg" className="text-center text-sm text-zinc-400">
            <Layers className="w-8 h-8 text-zinc-600 mx-auto mb-3" />
            No flashcards were generated for this note.
          </Card>
        )}
      </div>

      {/* Spaced repetition: save deck + review (unchanged data layer) */}
      {flashcards.length > 0 && (
        <div className="xl:col-span-3">
          <Card accent padding="lg">
            <h3 className="text-base font-semibold text-zinc-100 mb-1 flex items-center gap-2">
              <BookOpen className="w-5 h-5 text-cyan-300" />
              Spaced Repetition
            </h3>
            <p className="text-xs text-zinc-400 mb-5">
              Save this deck and review it on an SM-2 schedule to lock it into memory.
            </p>
            <Flashcards cards={flashcards} courseId={courseId} userId={userId} />
          </Card>
        </div>
      )}
    </div>
  )

  // When no note is being viewed and we're not generating, show the centered studio.
  // Otherwise show the loading state or the reader.
  const showReader = !loading && !!generatedNotes

  // ===== Library slide-over (compact saved-notes list) =====
  const renderLibrary = () => (
    <AnimatePresence>
      {libraryOpen && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setLibraryOpen(false)}
            className="absolute inset-0 z-30 bg-black/50 backdrop-blur-sm"
          />
          <motion.aside
            initial={{ x: 380 }}
            animate={{ x: 0 }}
            exit={{ x: 380 }}
            transition={{ type: 'spring', stiffness: 320, damping: 34 }}
            className="absolute inset-y-0 right-0 z-40 flex w-[380px] max-w-[88vw] flex-col border-l border-white/10 bg-[#0c0f18]"
          >
            <div className="flex h-14 flex-shrink-0 items-center justify-between px-4">
              <span className="flex items-center gap-2 text-sm font-semibold text-zinc-100">
                <Bookmark className="h-4 w-4 text-cyan-300" />
                Library ({savedNotes.length})
              </span>
              <button
                onClick={() => setLibraryOpen(false)}
                className="rounded-lg p-1.5 text-zinc-400 transition-colors hover:bg-white/[0.06] hover:text-zinc-100"
                aria-label="Close library"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            <div className="flex-shrink-0 px-3 pb-3">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-zinc-500" />
                <input
                  type="text"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  placeholder="Search your notes…"
                  className="w-full rounded-lg border border-white/10 bg-white/[0.03] py-2 pl-10 pr-4 text-sm text-zinc-100 placeholder-zinc-500 outline-none transition-colors focus:border-cyan-400/60 focus:ring-2 focus:ring-cyan-500/25"
                />
              </div>
            </div>

            <div className="flex-1 space-y-2 overflow-y-auto px-3 pb-4">
              {notesQuery.isError ? (
                <ErrorInline
                  message="Couldn't load your saved notes."
                  onRetry={() => void notesQuery.refetch()}
                  className="mt-2"
                />
              ) : filteredNotes.length === 0 ? (
                <div className="px-2 py-12 text-center">
                  <BrandMark className="mx-auto mb-4 h-12 w-12" />
                  <p className="text-sm font-medium text-zinc-200">
                    {searchTerm ? 'No notes found' : 'No saved notes yet'}
                  </p>
                  <p className="mt-1 text-xs text-zinc-500">
                    {searchTerm ? 'Try a different search.' : 'Generate notes to build your library.'}
                  </p>
                </div>
              ) : (
                filteredNotes.map((note) => {
                  const cardsCount = parseFlashcardsFromText(note.content || '').length
                  const active = currentNoteId === note.id
                  return (
                    <motion.div
                      key={note.id}
                      initial={{ opacity: 0, y: 6 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.2 }}
                      onClick={() => loadNote(note)}
                      className={cn(
                        'group cursor-pointer rounded-xl border p-3 transition-colors',
                        active
                          ? 'border-cyan-400/30 bg-cyan-500/[0.08]'
                          : 'border-white/10 bg-white/[0.02] hover:border-white/20 hover:bg-white/[0.05]',
                      )}
                    >
                      <div className="flex items-start justify-between gap-2">
                        <h3 className="min-w-0 flex-1 truncate text-sm font-semibold text-zinc-100">{note.title}</h3>
                        <span className="flex-shrink-0 text-[11px] text-zinc-500">
                          {new Date(note.created_at).toLocaleDateString()}
                        </span>
                      </div>
                      <div className="mt-1.5 flex items-center gap-3 text-[11px] text-zinc-400">
                        <span className="inline-flex items-center gap-1">
                          <Clock className="h-3 w-3" />
                          {note.reading_time}
                        </span>
                        <span className="inline-flex items-center gap-1">
                          <FileText className="h-3 w-3" />
                          {note.word_count} words
                        </span>
                        {cardsCount > 0 && (
                          <span className="inline-flex items-center gap-1 rounded-full bg-gradient-brand-soft border border-cyan-400/20 px-1.5 py-0.5 text-cyan-300">
                            <Layers className="h-3 w-3" />
                            {cardsCount}
                          </span>
                        )}
                      </div>
                      <div className="mt-2.5 flex items-center gap-1 opacity-0 transition-opacity group-hover:opacity-100">
                        <button
                          onClick={(e) => { e.stopPropagation(); loadNote(note) }}
                          className="rounded-md p-1.5 text-cyan-300 transition-colors hover:bg-cyan-500/10"
                          title="Open note"
                          aria-label="Open note"
                        >
                          <Edit3 className="h-3.5 w-3.5" />
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            setSavedFlashcards(parseFlashcardsFromText(note.content || ''))
                            setShowCardsFor(note.id)
                          }}
                          className="rounded-md p-1.5 text-cyan-300 transition-colors hover:bg-cyan-500/10"
                          title="Study flashcards"
                          aria-label="Study flashcards"
                        >
                          <Layers className="h-3.5 w-3.5" />
                        </button>
                        <button
                          onClick={(e) => { e.stopPropagation(); exportNote(note) }}
                          className="rounded-md p-1.5 text-emerald-400 transition-colors hover:bg-emerald-500/10"
                          title="Download note"
                          aria-label="Download note"
                        >
                          <Download className="h-3.5 w-3.5" />
                        </button>
                        <button
                          onClick={(e) => { e.stopPropagation(); confirmDeleteNote(note.id) }}
                          className="rounded-md p-1.5 text-zinc-500 transition-colors hover:bg-rose-500/10 hover:text-rose-400"
                          title="Delete note"
                          aria-label="Delete note"
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </button>
                      </div>
                    </motion.div>
                  )
                })
              )}
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  )

  // ===== Flashcards Modal for Saved Notes =====
  const renderCardsModal = () => (
    <>
      {/* Flashcards Modal for Saved Notes */}
      <AnimatePresence>
        {showCardsFor && (
          <motion.div
            className="fixed inset-0 z-50 flex items-center justify-center p-4"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" onClick={() => { setShowCardsFor(null); setSavedFlashcards([]) }} />
            <motion.div
              className="relative z-10 w-full max-w-3xl"
              initial={{ scale: 0.96, y: 8 }}
              animate={{ scale: 1, y: 0 }}
              exit={{ scale: 0.96, y: 8 }}
              transition={{ duration: 0.2 }}
            >
              <Card padding="lg" className="space-y-5 elev-3 !bg-[#16161b]">
                <div className="flex items-center justify-between">
                  <h3 className="text-base font-semibold text-zinc-100 flex items-center gap-2">
                    <Layers className="w-5 h-5 text-cyan-300" />
                    Study Flashcards
                    <span className="ml-2 text-xs font-medium text-cyan-300 bg-gradient-brand-soft border border-cyan-400/20 px-2 py-0.5 rounded-full">
                      {savedFlashcards.length}
                    </span>
                  </h3>
                  <button
                    onClick={() => { setShowCardsFor(null); setSavedFlashcards([]) }}
                    className="p-1.5 text-zinc-500 hover:text-zinc-200 hover:bg-white/[0.06] rounded-lg transition-colors"
                    aria-label="Close"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>

                {savedFlashcards.length > 0 ? (
                  <>
                    <FlashcardDeck cards={savedFlashcards} />
                    <div className="pt-4 border-t border-white/10">
                      <Flashcards cards={savedFlashcards} courseId={courseId} userId={userId} />
                    </div>
                  </>
                ) : (
                  <div className="text-zinc-400">No flashcards found in this note.</div>
                )}
              </Card>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )

  return (
    <div className="relative flex min-h-full flex-col">
      {/* Floating controls (no competing tab-pill row) — mirrors ChatPage's History button */}
      <div className="absolute right-0 top-0 z-20 flex items-center gap-2">
        {(generatedNotes || !usingAllFiles || topic) && (
          <button
            onClick={clearConversations}
            className="inline-flex items-center gap-1.5 rounded-lg border border-white/10 bg-white/[0.04] px-2.5 py-1.5 text-[13px] text-zinc-300 backdrop-blur transition-colors hover:bg-white/[0.08] hover:text-zinc-100"
          >
            <RotateCcw className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">New</span>
          </button>
        )}
        <button
          onClick={() => setLibraryOpen(true)}
          className="inline-flex items-center gap-1.5 rounded-lg border border-white/10 bg-white/[0.04] px-2.5 py-1.5 text-[13px] text-zinc-300 backdrop-blur transition-colors hover:bg-white/[0.08] hover:text-zinc-100"
        >
          <Bookmark className="h-3.5 w-3.5" />
          <span className="hidden sm:inline">Library</span>
          {savedNotes.length > 0 && (
            <span className="rounded-full bg-white/[0.1] px-1.5 text-[11px] text-zinc-400">{savedNotes.length}</span>
          )}
        </button>
      </div>

      {/* Main content: centered studio, loading, or reader */}
      {loading ? (
        <div className="flex-1 pt-14">{renderLoading()}</div>
      ) : showReader ? (
        <div className="flex-1 pt-14">{renderReader()}</div>
      ) : (
        <div className="flex-1">{renderStudio()}</div>
      )}

      {renderLibrary()}
      {renderCardsModal()}

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
