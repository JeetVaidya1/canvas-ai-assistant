// src/components/studykit/NotesLibrary.tsx — saved-notes slide-over + flashcard modal + delete confirm
import { useMemo, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { Bookmark, Clock, Download, Edit3, FileText, Layers, Search, Trash2, X } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/Button'
import { Modal } from '@/components/ui/Modal'
import { EmptyState } from '@/components/ui/States'
import ErrorInline from '@/components/shared/ErrorInline'
import Flashcards from '@/components/FlashCards'
import type { SavedNote } from '@/lib/api'
import { downloadNoteAsText, parseFlashcardsFromText, type Flashcard } from './noteUtils'
import { FlashcardDeck } from './GeneratedFlashcards'
import type { NotesStudio } from './useNotesStudio'

function LibrarySkeleton() {
  return (
    <div className="space-y-2" aria-hidden>
      {[0, 1, 2].map((i) => (
        <div key={i} className="animate-pulse rounded-xl border border-line bg-surface p-3">
          <div className="h-3.5 w-2/3 rounded bg-paper-deep" />
          <div className="mt-2.5 h-3 w-1/3 rounded bg-paper-deep" />
        </div>
      ))}
    </div>
  )
}

/** Slide-over library of saved notes: search, reopen, study cards, export, delete. */
export default function NotesLibrary({ studio }: { studio: NotesStudio }) {
  const {
    courseId, userId, notesQuery, savedNotes, currentNoteId,
    libraryOpen, setLibraryOpen, openNote, removeNote,
  } = studio

  const [searchTerm, setSearchTerm] = useState('')
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null)
  const [cardsNoteId, setCardsNoteId] = useState<string | null>(null)
  const [savedCards, setSavedCards] = useState<Flashcard[]>([])

  const filteredNotes = useMemo(() => {
    const q = (searchTerm || '').toLowerCase()
    return (savedNotes || []).filter((note) =>
      (note.title || '').toLowerCase().includes(q) ||
      (note.topics || []).some((t) => (t || '').toLowerCase().includes(q)),
    )
  }, [savedNotes, searchTerm])

  const openCards = (note: SavedNote) => {
    setSavedCards(parseFlashcardsFromText(note.content || ''))
    setCardsNoteId(note.id)
  }
  const closeCards = () => {
    setCardsNoteId(null)
    setSavedCards([])
  }

  const confirmDelete = () => {
    if (!deleteConfirmId) return
    const noteId = deleteConfirmId
    setDeleteConfirmId(null)
    void removeNote(noteId)
  }

  return (
    <>
      <AnimatePresence>
        {libraryOpen && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setLibraryOpen(false)}
              className="absolute inset-0 z-30 bg-ink/30"
            />
            <motion.aside
              initial={{ x: 380 }}
              animate={{ x: 0 }}
              exit={{ x: 380 }}
              transition={{ type: 'spring', stiffness: 320, damping: 34 }}
              className="absolute inset-y-0 right-0 z-40 flex w-[380px] max-w-[88vw] flex-col border-l border-line bg-paper elev-3"
            >
              <div className="flex h-14 flex-shrink-0 items-center justify-between px-4 border-b border-line">
                <span className="flex items-center gap-2 text-sm font-semibold text-ink">
                  <Bookmark className="h-4 w-4 text-accent" />
                  Library ({savedNotes.length})
                </span>
                <button
                  onClick={() => setLibraryOpen(false)}
                  className="rounded-lg p-1.5 text-ink-faint transition-colors hover:bg-paper-deep hover:text-ink"
                  aria-label="Close library"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>

              <div className="flex-shrink-0 px-3 py-3">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-ink-faint" />
                  <input
                    type="text"
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    placeholder="Search your notes…"
                    className="w-full rounded-lg border border-line bg-surface py-2 pl-10 pr-4 text-sm text-ink placeholder-ink-faint outline-none transition-colors hover:border-line-strong focus:border-accent focus:ring-2 focus:ring-accent/20"
                  />
                </div>
              </div>

              <div className="flex-1 space-y-2 overflow-y-auto px-3 pb-4">
                {notesQuery.isLoading ? (
                  <LibrarySkeleton />
                ) : notesQuery.isError ? (
                  <ErrorInline
                    message="Couldn't load your saved notes."
                    onRetry={() => void notesQuery.refetch()}
                    className="mt-2"
                  />
                ) : filteredNotes.length === 0 ? (
                  <EmptyState
                    icon={<Bookmark />}
                    title={searchTerm ? 'No notes found' : 'No notes yet'}
                    description={
                      searchTerm
                        ? 'Try a different search.'
                        : 'Generate your first study note — it will be saved here.'
                    }
                  />
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
                        onClick={() => openNote(note)}
                        className={cn(
                          'group cursor-pointer rounded-xl border p-3 transition-colors',
                          active
                            ? 'border-accent-line bg-accent-wash'
                            : 'border-line bg-surface hover:border-line-strong hover:bg-surface-hover',
                        )}
                      >
                        <div className="flex items-start justify-between gap-2">
                          <h3 className="min-w-0 flex-1 truncate text-sm font-semibold text-ink">{note.title}</h3>
                          <span className="flex-shrink-0 text-[11px] text-ink-faint">
                            {new Date(note.created_at).toLocaleDateString()}
                          </span>
                        </div>
                        <div className="mt-1.5 flex items-center gap-3 text-[11px] text-ink-soft">
                          <span className="inline-flex items-center gap-1">
                            <Clock className="h-3 w-3" />
                            {note.reading_time}
                          </span>
                          <span className="inline-flex items-center gap-1">
                            <FileText className="h-3 w-3" />
                            {note.word_count} words
                          </span>
                          {cardsCount > 0 && (
                            <span className="inline-flex items-center gap-1 rounded-full bg-marker-soft border border-marker px-1.5 py-0.5 text-ink tnum">
                              <Layers className="h-3 w-3" />
                              {cardsCount}
                            </span>
                          )}
                        </div>
                        <div className="mt-2.5 flex items-center gap-1 opacity-0 transition-opacity group-hover:opacity-100">
                          <button
                            onClick={(e) => { e.stopPropagation(); openNote(note) }}
                            className="rounded-md p-1.5 text-accent transition-colors hover:bg-accent-wash"
                            title="Open note"
                            aria-label="Open note"
                          >
                            <Edit3 className="h-3.5 w-3.5" />
                          </button>
                          <button
                            onClick={(e) => { e.stopPropagation(); openCards(note) }}
                            className="rounded-md p-1.5 text-accent transition-colors hover:bg-accent-wash"
                            title="Study flashcards"
                            aria-label="Study flashcards"
                          >
                            <Layers className="h-3.5 w-3.5" />
                          </button>
                          <button
                            onClick={(e) => { e.stopPropagation(); downloadNoteAsText(note) }}
                            className="rounded-md p-1.5 text-success transition-colors hover:bg-success-wash"
                            title="Download note"
                            aria-label="Download note"
                          >
                            <Download className="h-3.5 w-3.5" />
                          </button>
                          <button
                            onClick={(e) => { e.stopPropagation(); setDeleteConfirmId(note.id) }}
                            className="rounded-md p-1.5 text-ink-faint transition-colors hover:bg-danger-wash hover:text-danger"
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

      {/* Flashcards from a saved note */}
      <Modal open={!!cardsNoteId} onClose={closeCards} title="Study flashcards" size="lg">
        {savedCards.length > 0 ? (
          <div className="space-y-5">
            <FlashcardDeck cards={savedCards} />
            <div className="border-t border-line pt-4">
              <Flashcards cards={savedCards} courseId={courseId} userId={userId} />
            </div>
          </div>
        ) : (
          <EmptyState
            icon={<Layers />}
            title="No flashcards in this note"
            description="This note was saved without an embedded deck."
          />
        )}
      </Modal>

      {/* Delete confirmation — Modal primitive with danger action */}
      <Modal
        open={!!deleteConfirmId}
        onClose={() => setDeleteConfirmId(null)}
        title="Delete note"
        description="Delete this note? This cannot be undone."
        size="sm"
        footer={
          <>
            <Button variant="secondary" size="sm" onClick={() => setDeleteConfirmId(null)}>
              Cancel
            </Button>
            <Button variant="danger" size="sm" onClick={confirmDelete}>
              Delete
            </Button>
          </>
        }
      >
        <p className="text-sm text-ink-soft">
          The note and its embedded flashcards will be removed from your library.
        </p>
      </Modal>
    </>
  )
}
