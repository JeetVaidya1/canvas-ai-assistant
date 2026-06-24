// src/components/NotesCreator.tsx — "Notes Studio": grounded notes + auto-generated flashcard deck
import { useState, useEffect, useMemo } from 'react'
import { AnimatePresence, motion } from 'motion/react'
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
  X,
  Sparkles,
  Layers,
  ChevronLeft,
  ChevronRight,
  Check,
} from 'lucide-react'
import {
  generateNotes,
  saveNotes as apiSaveNotes,
  updateNote as apiUpdateNote,
  getNotes as apiGetNotes,
  deleteNotes as apiDeleteNotes,
  listFiles,
  type NotesResponse,
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
        <div className="absolute inset-0 [backface-visibility:hidden] [transform:rotateY(180deg)] rounded-2xl bg-gradient-brand-soft border border-cyan-500/25 p-7 flex flex-col">
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-widest text-cyan-300">
            <Sparkles className="w-3.5 h-3.5" /> Answer
          </div>
          <div className="flex-1 flex items-center justify-center text-center overflow-auto px-2">
            <div className="prose prose-invert prose-sm max-w-none text-zinc-100">
              <Markdown content={card.a} />
            </div>
          </div>
          <p className="text-xs text-cyan-400/60 text-center">Tap to flip back</p>
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
                  ? 'w-1.5 bg-cyan-500/50'
                  : 'w-1.5 bg-zinc-700'
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
        <span className="text-xs text-zinc-600">{seen.size}/{cards.length} reviewed</span>
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
  const [loadStage, setLoadStage] = useState(0)
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

    setSaving(true)
    setErrMsg(null)
    try {
      // Editing an existing note -> update in place (PUT); otherwise create (POST).
      const savedNote = currentNoteId
        ? await apiUpdateNote(currentNoteId, courseId, noteTitle.trim(), generatedNotes, selectedFiles, topic)
        : await apiSaveNotes(courseId, noteTitle.trim(), generatedNotes, selectedFiles, topic)
      await loadSavedNotes()
      setCurrentNoteId(savedNote.id)
    } catch (error: unknown) {
      console.error('Failed to save notes:', error)
      setErrMsg(error instanceof Error ? error.message : 'Failed to save notes.')
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

  // ===== Create tab =====
  const renderConfig = () => (
    <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
      {/* File Selection */}
      <Card accent padding="lg" className="lg:col-span-3">
        <h3 className="text-base font-semibold text-zinc-100 mb-1 flex items-center gap-3">
          <span className="w-9 h-9 rounded-xl bg-gradient-brand-soft border border-cyan-500/15 flex items-center justify-center flex-shrink-0">
            <FileText className="w-4 h-4 text-cyan-300" />
          </span>
          Source Files
        </h3>
        <p className="text-xs text-zinc-500 mb-4 ml-12 -mt-1">
          Notes are grounded in — and cited from — the files you pick.
        </p>

        {!courseId ? (
          <div className="text-center py-8 text-amber-500">Select a course to choose files.</div>
        ) : availableFiles.length === 0 ? (
          <div className="text-center py-10 text-zinc-400">
            <BookOpen className="w-12 h-12 text-zinc-600 mx-auto mb-3" />
            <p>No files uploaded to this course yet.</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 max-h-72 overflow-auto pr-1">
            {availableFiles.map((file) => {
              const checked = selectedFiles.includes(file)
              return (
                <label
                  key={file}
                  className={`flex items-center p-3 border rounded-lg cursor-pointer transition-all ${
                    checked
                      ? 'bg-gradient-brand-soft border-cyan-500/30 text-cyan-200'
                      : 'bg-zinc-900 border-zinc-700 text-zinc-300 hover:bg-zinc-800 hover:border-zinc-600'
                  }`}
                >
                  <span
                    className={`w-5 h-5 mr-3 rounded-md border flex items-center justify-center flex-shrink-0 transition-colors ${
                      checked ? 'bg-cyan-500 border-cyan-400' : 'border-zinc-600'
                    }`}
                  >
                    {checked && <Check className="w-3.5 h-3.5 text-zinc-950" />}
                  </span>
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={(e) => {
                      if (e.target.checked) setSelectedFiles(prev => [...prev, file])
                      else setSelectedFiles(prev => prev.filter(f => f !== file))
                    }}
                    className="sr-only"
                  />
                  <FileText className="w-4 h-4 mr-2 opacity-70 flex-shrink-0" />
                  <span className="text-sm truncate">{file}</span>
                </label>
              )
            })}
          </div>
        )}

        <div className="mt-4 text-sm text-zinc-500">
          {selectedFiles.length} file{selectedFiles.length === 1 ? '' : 's'} selected
        </div>
      </Card>

      {/* Configuration */}
      <Card accent padding="lg" className="lg:col-span-2 flex flex-col">
        <h3 className="text-base font-semibold text-zinc-100 mb-4 flex items-center gap-3">
          <span className="w-9 h-9 rounded-xl bg-gradient-brand-soft border border-cyan-500/15 flex items-center justify-center flex-shrink-0">
            <Brain className="w-4 h-4 text-cyan-300" />
          </span>
          Configure
        </h3>

        <div className="space-y-5 flex-1">
          <div className="space-y-2">
            <label className="text-xs font-medium text-zinc-400">Focus topic (optional)</label>
            <input
              type="text"
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              placeholder="e.g., Binary Search Trees"
              className="w-full px-3 py-2.5 bg-zinc-800/70 border border-zinc-700 text-zinc-100 placeholder-zinc-600 rounded-lg text-sm outline-none focus:border-cyan-500/60 focus:ring-2 focus:ring-cyan-500/20 transition-colors"
            />
            <p className="text-xs text-zinc-500">Leave blank for full coverage.</p>
          </div>

          <div className="space-y-2">
            <label className="text-xs font-medium text-zinc-400">Style</label>
            <div className="flex flex-col gap-1.5">
              {NOTE_STYLES.map((style) => {
                const active = noteStyle === style.value
                return (
                  <button
                    key={style.value}
                    type="button"
                    onClick={() => setNoteStyle(style.value)}
                    className={`text-left px-3 py-2.5 rounded-lg border transition-all ${
                      active
                        ? 'bg-gradient-brand-soft text-cyan-200 border-cyan-500/30'
                        : 'bg-zinc-800/40 text-zinc-300 border-zinc-700 hover:border-zinc-600'
                    }`}
                  >
                    <span className="text-sm font-medium">{style.label}</span>
                    <span className="block text-xs text-zinc-500">{style.hint}</span>
                  </button>
                )
              })}
            </div>
          </div>
        </div>

        <Button
          size="lg"
          onClick={handleGenerateNotes}
          loading={loading}
          disabled={loading || selectedFiles.length === 0 || !courseId}
          leftIcon={<Sparkles className="w-5 h-5" />}
          className="mt-6 w-full"
        >
          {loading ? 'Generating…' : 'Generate Notes'}
        </Button>

        {errMsg && <div className="mt-3 text-sm text-red-400">{errMsg}</div>}
      </Card>
    </div>
  )

  const renderLoading = () => (
    <Card accent padding="lg" className="glow-brand">
      <div className="text-center py-10">
        <div className="relative w-20 h-20 mx-auto mb-6">
          <div className="absolute inset-0 rounded-full bg-gradient-brand opacity-20 blur-xl animate-pulse" />
          <div className="absolute inset-0 border-4 border-cyan-500/15 border-t-cyan-400 rounded-full animate-spin" />
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
        <p className="text-xs text-zinc-600 mt-4">Analyzing {selectedFiles.length} file(s)</p>

        <div className="mt-6 max-w-md mx-auto space-y-2">
          {[0, 1, 2].map((i) => (
            <div key={i} className="h-3 rounded bg-zinc-800/60 overflow-hidden">
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
        <div className="bg-zinc-800/40 px-6 py-4 border-b border-zinc-800">
          <div className="flex items-center justify-between gap-3 flex-wrap">
            <div className="flex items-center gap-3 min-w-0 flex-1">
              <Edit3 className="w-5 h-5 text-cyan-300 flex-shrink-0" />
              <input
                type="text"
                value={noteTitle}
                onChange={(e) => setNoteTitle(e.target.value)}
                placeholder="Enter note title…"
                className="text-lg font-semibold bg-transparent border-none focus:outline-none focus:ring-0 text-zinc-50 placeholder-zinc-600 min-w-0 w-full"
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
              className="w-full h-[28rem] p-4 bg-zinc-800/70 border border-zinc-700 text-zinc-100 rounded-lg outline-none focus:border-cyan-500/60 focus:ring-2 focus:ring-cyan-500/20 resize-none font-mono text-sm transition-colors"
              placeholder="Your generated notes will appear here…"
            />
          )}

          {generatedNotes && (
            <div className="mt-6 pt-4 border-t border-zinc-800 flex items-center gap-6 text-sm text-zinc-500 flex-wrap">
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
              <span className="ml-1 text-xs font-medium text-cyan-300 bg-gradient-brand-soft border border-cyan-500/20 px-2 py-0.5 rounded-full">
                {flashcards.length}
              </span>
            </h3>
            <p className="text-xs text-zinc-500 mb-5">Auto-generated from your notes. Flip to study.</p>
            <FlashcardDeck cards={flashcards} />
          </Card>
        ) : (
          <Card padding="lg" className="text-center text-sm text-zinc-500">
            <Layers className="w-8 h-8 text-zinc-700 mx-auto mb-3" />
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
            <p className="text-xs text-zinc-500 mb-5">
              Save this deck and review it on an SM-2 schedule to lock it into memory.
            </p>
            <Flashcards cards={flashcards} courseId={courseId} userId={userId} />
          </Card>
        </div>
      )}
    </div>
  )

  const renderCreateTab = () => (
    <div className="space-y-8">
      {renderConfig()}
      {loading && renderLoading()}
      {!loading && generatedNotes && renderReader()}
    </div>
  )

  // ===== Saved tab =====
  const renderSavedTab = () => (
    <div className="space-y-6">
      <Card padding="md">
        <div className="flex items-center gap-4">
          <div className="flex-1 relative">
            <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-zinc-500" />
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Search your notes…"
              className="w-full pl-10 pr-4 py-2.5 bg-zinc-800/70 border border-zinc-700 text-zinc-100 placeholder-zinc-600 rounded-lg text-sm outline-none focus:border-cyan-500/60 focus:ring-2 focus:ring-cyan-500/20 transition-colors"
            />
          </div>
          <div className="text-sm text-zinc-500">{filteredNotes.length} note(s)</div>
        </div>
      </Card>

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
            <Button onClick={() => setActiveTab('create')}>Create Your First Notes</Button>
          )}
        </Card>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {filteredNotes.map((note) => {
            const cardsCount = parseFlashcardsFromText(note.content || '').length
            return (
              <motion.div key={note.id} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.25 }}>
                <Card accent padding="lg" interactive className="group h-full">
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
                      <button
                        onClick={() => {
                          setSavedFlashcards(parseFlashcardsFromText(note.content || ''))
                          setShowCardsFor(note.id)
                        }}
                        className="p-2 text-cyan-300 hover:bg-cyan-500/10 rounded-lg transition-colors"
                        title="Study flashcards"
                        aria-label="Study flashcards"
                      >
                        <Layers className="w-4 h-4" />
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
                    {(note.content || '').slice(0, 200)}…
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
              </motion.div>
            )
          })}
        </div>
      )}

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
              <Card padding="lg" className="space-y-5 shadow-2xl !bg-zinc-900">
                <div className="flex items-center justify-between">
                  <h3 className="text-base font-semibold text-zinc-100 flex items-center gap-2">
                    <Layers className="w-5 h-5 text-cyan-300" />
                    Study Flashcards
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
                  <>
                    <FlashcardDeck cards={savedFlashcards} />
                    <div className="pt-4 border-t border-zinc-800">
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
    </div>
  )

  return (
    <div className="max-w-7xl mx-auto">
      <div className="mb-8">
        <PageHeader
          eyebrow="Notes Studio"
          title="Notes Creator"
          subtitle={`Grounded notes + auto-generated flashcards from your ${courseName} materials`}
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

        <div className="flex items-center gap-1 bg-zinc-800/70 border border-zinc-700 p-1 rounded-lg w-fit">
          <button
            onClick={() => setActiveTab('create')}
            className={`px-5 py-2 rounded-md text-sm font-medium transition-all ${
              activeTab === 'create'
                ? 'bg-gradient-brand-soft text-cyan-300 border border-cyan-500/30'
                : 'text-zinc-400 border border-transparent hover:text-zinc-200'
            }`}
          >
            <Sparkles className="w-4 h-4 inline mr-2" />
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
            Library ({savedNotes.length})
          </button>
        </div>
      </div>

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
