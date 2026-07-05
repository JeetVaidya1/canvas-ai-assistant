import { useMemo, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { Shuffle, RotateCcw, Download, Save, Brain, CheckCircle } from 'lucide-react'
import { Markdown } from '@/components/ui/Markdown'
import { Button } from '@/components/ui/Button'
import { Card } from '@/components/ui/Card'
import {
  exportFlashcardsAnki,
  type DeckCard,
} from '@/lib/api'
import { flashcardDeckOptions, useSaveFlashcards, useReviewFlashcard } from '@/hooks/useFlashcards'
import { showError, showSuccess } from '@/lib/toast'

export type Flashcard = { q: string; a: string }

type StudyMode = 'all' | 'hide-answers' | 'typing' | 'sr'

type GradeVariant = 'primary' | 'secondary' | 'ghost' | 'danger'

// SM-2 recall grades surfaced as four buttons.
const GRADES: { label: string; grade: number; variant: GradeVariant; className: string }[] = [
  { label: 'Again', grade: 1, variant: 'secondary', className: '!bg-rose-500/15 !border-rose-400/30 !text-rose-200 hover:!bg-rose-500/25' },
  { label: 'Hard', grade: 3, variant: 'secondary', className: '!bg-amber-500/15 !border-amber-400/30 !text-amber-200 hover:!bg-amber-500/25' },
  { label: 'Good', grade: 4, variant: 'secondary', className: '!bg-cyan-500/15 !border-cyan-400/30 !text-cyan-200 hover:!bg-cyan-500/25' },
  { label: 'Easy', grade: 5, variant: 'secondary', className: '!bg-emerald-500/15 !border-emerald-400/30 !text-emerald-200 hover:!bg-emerald-500/25' },
]

function downloadCSV(cards: Flashcard[]) {
  const header = 'Front,Back'
  const rows = cards.map(c =>
    `"${(c.q || '').replace(/"/g,'""')}","${(c.a || '').replace(/"/g,'""')}"`
  )
  const csv = [header, ...rows].join('\n')
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = 'flashcards.csv'
  a.click()
  URL.revokeObjectURL(url)
}

export default function Flashcards({
  cards,
  title = 'Flashcards',
  courseId,
  userId = 'anonymous',
}: { cards: Flashcard[]; title?: string; courseId?: string; userId?: string }) {
  const [order, setOrder] = useState<number[]>(() => cards.map((_, i) => i))
  const [showBack, setShowBack] = useState<Record<number, boolean>>({})
  const [studyMode, setStudyMode] = useState<StudyMode>('all')
  const [typingAnswers, setTypingAnswers] = useState<Record<number,string>>({})

  // Spaced-repetition review state (only used when courseId is provided).
  // The deck is loaded through the query cache but reviewed off a local
  // snapshot so grading (which invalidates the cache) can't reshuffle cards
  // mid-session.
  const qc = useQueryClient()
  const saveCardsMutation = useSaveFlashcards(courseId ?? '')
  const reviewCardMutation = useReviewFlashcard(courseId, userId)
  const saving = saveCardsMutation.isPending
  const [deck, setDeck] = useState<DeckCard[] | null>(null)
  const [deckLoading, setDeckLoading] = useState(false)
  const [srIndex, setSrIndex] = useState(0)
  const [srRevealed, setSrRevealed] = useState(false)

  const handleSaveToDeck = async () => {
    if (!courseId) return
    try {
      const res = await saveCardsMutation.mutateAsync(cards)
      showSuccess(`Saved ${res.saved} card${res.saved === 1 ? '' : 's'} to your deck${res.skipped ? ` (${res.skipped} already there)` : ''}`)
    } catch (e) {
      showError(e instanceof Error ? e.message : 'Failed to save cards')
    }
  }

  const loadDeck = async () => {
    if (!courseId) return
    setDeckLoading(true)
    try {
      // staleTime 0: entering review must always see the freshest due list.
      const d = await qc.fetchQuery({ ...flashcardDeckOptions(courseId, userId), staleTime: 0 })
      // Due cards first (backend already sorts); review only the due ones.
      setDeck(d.cards.filter((c) => c.due))
      setSrIndex(0)
      setSrRevealed(false)
    } catch (e) {
      showError(e instanceof Error ? e.message : 'Failed to load deck')
      setDeck([])
    } finally {
      setDeckLoading(false)
    }
  }

  const gradeCurrent = async (grade: number) => {
    if (!deck || !deck[srIndex]) return
    const card = deck[srIndex]
    try {
      await reviewCardMutation.mutateAsync({ cardId: card.id, grade })
    } catch (e) {
      showError(e instanceof Error ? e.message : 'Failed to record review')
    }
    setSrRevealed(false)
    setSrIndex((i) => i + 1)
  }

  const enterSrMode = () => {
    setStudyMode('sr')
    void loadDeck()
  }

  const handleAnkiExport = async () => {
    if (!courseId) return
    try {
      const blob = await exportFlashcardsAnki(courseId, userId)
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${courseId}_flashcards.apkg`
      a.click()
      URL.revokeObjectURL(url)
    } catch (e) {
      showError(e instanceof Error ? e.message : 'Anki export failed')
    }
  }

  const shuffled = () => {
    const arr = [...order]
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1))
      ;[arr[i], arr[j]] = [arr[j], arr[i]]
    }
    setOrder(arr)
    setShowBack({})
    setTypingAnswers({})
  }

  const reset = () => {
    setOrder(cards.map((_, i) => i))
    setShowBack({})
    setTypingAnswers({})
  }

  const visibleCards = useMemo(() => order.map(i => ({ idx: i, ...cards[i] })), [order, cards])

  if (!cards?.length) return null

  return (
    <Card accent padding="lg">
      <div className="flex items-center justify-between mb-4 flex-wrap gap-3">
        <h3 className="text-lg font-semibold text-zinc-50">
          {title} <span className="text-zinc-400 font-normal">({cards.length})</span>
        </h3>
        <div className="flex items-center gap-2 flex-wrap">
          <select
            value={studyMode}
            onChange={e => { setStudyMode(e.target.value as StudyMode); setShowBack({}); setTypingAnswers({}) }}
            className="bg-white/[0.04] border border-white/10 text-zinc-100 rounded-lg px-3 py-1.5 text-sm outline-none focus:border-cyan-400/60 focus:ring-2 focus:ring-cyan-500/25 transition-colors"
            title="Study mode"
          >
            <option value="all">Flip to reveal</option>
            <option value="hide-answers">Prompt only</option>
            <option value="typing">Typing practice</option>
          </select>

          <Button variant="ghost" size="sm" onClick={shuffled} leftIcon={<Shuffle className="w-4 h-4" />}>
            Shuffle
          </Button>
          <Button variant="ghost" size="sm" onClick={reset} leftIcon={<RotateCcw className="w-4 h-4" />}>
            Reset
          </Button>
          <Button variant="ghost" size="sm" onClick={() => downloadCSV(cards)} leftIcon={<Download className="w-4 h-4" />}>
            CSV
          </Button>
          {courseId && (
            <>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => void handleAnkiExport()}
                leftIcon={<Download className="w-4 h-4" />}
                title="Export your saved deck to Anki, keeping spaced-repetition state"
              >
                Anki
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => void handleSaveToDeck()}
                loading={saving}
                leftIcon={<Save className="w-4 h-4" />}
              >
                {saving ? 'Saving…' : 'Save to deck'}
              </Button>
              <Button
                variant={studyMode === 'sr' ? 'primary' : 'secondary'}
                size="sm"
                onClick={() => (studyMode === 'sr' ? setStudyMode('all') : enterSrMode())}
                leftIcon={<Brain className="w-4 h-4" />}
              >
                Review
              </Button>
            </>
          )}
        </div>
      </div>

      {/* Spaced-repetition review */}
      {studyMode === 'sr' && (
        <SrReview
          deck={deck}
          loading={deckLoading}
          index={srIndex}
          revealed={srRevealed}
          onReveal={() => setSrRevealed(true)}
          onGrade={gradeCurrent}
          onRestart={() => void loadDeck()}
        />
      )}

      {studyMode !== 'sr' && (
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {visibleCards.map(({ idx, q, a }) => {
          const flipped = showBack[idx]
          return (
            <div
              key={idx}
              className="group relative perspective"
            >
              {/* Card */}
              <div className="card-surface accent-top relative h-44 w-full transition-transform duration-500 [transform-style:preserve-3d] rounded-xl p-4 hover:border-cyan-400/30"
                   style={{ transform: (studyMode==='all' && flipped) ? 'rotateY(180deg)' : 'none' }}
                   onClick={() => studyMode==='all' && setShowBack(s => ({...s, [idx]: !s[idx]}))}
              >
                {/* Front */}
                <div className="absolute inset-0 p-4 backface-hidden">
                  <div className="text-xs font-semibold uppercase tracking-widest text-gradient-brand mb-1.5">Question</div>
                  <div className="text-zinc-50 font-medium leading-snug overflow-hidden max-h-32"><Markdown content={q} /></div>
                  {studyMode==='all' && (
                    <div className="absolute bottom-3 right-3 text-xs text-zinc-500">Click to flip</div>
                  )}
                </div>

                {/* Back (answer) */}
                <div className="absolute inset-0 p-4 rotate-y-180 backface-hidden">
                  <div className="text-xs font-semibold uppercase tracking-widest text-gradient-brand mb-1.5">Answer</div>
                  <div className="text-zinc-50 leading-snug overflow-hidden max-h-32"><Markdown content={a} /></div>
                </div>
              </div>

              {/* Hide-answers mode: show only prompts */}
              {studyMode==='hide-answers' && (
                <div className="absolute inset-0 rounded-xl border border-dashed border-white/15 bg-white/[0.04] p-4">
                  <div className="text-xs font-semibold uppercase tracking-widest text-zinc-400 mb-1.5">Prompt</div>
                  <div className="text-zinc-200 font-medium leading-snug overflow-hidden max-h-32"><Markdown content={q} /></div>
                </div>
              )}

              {/* Typing practice mode */}
              {studyMode==='typing' && (
                <div className="absolute inset-0 rounded-xl border border-cyan-400/30 bg-gradient-brand-soft p-4 flex flex-col">
                  <div className="text-xs font-semibold uppercase tracking-widest text-cyan-300 mb-1.5">Type your answer</div>
                  <textarea
                    className="flex-1 resize-none rounded-lg bg-white/[0.04] border border-cyan-400/30 px-3 py-2 text-sm text-zinc-50 outline-none focus:border-cyan-400/60 focus:ring-2 focus:ring-cyan-500/25 transition-colors"
                    value={typingAnswers[idx] || ''}
                    onChange={e => setTypingAnswers(s => ({...s, [idx]: e.target.value}))}
                    placeholder="Write your answer here…"
                  />
                  {typingAnswers[idx] && (
                    <div className="mt-2 text-xs text-zinc-400">
                      Correct answer: <span className="text-zinc-50">{a}</span>
                    </div>
                  )}
                </div>
              )}
            </div>
          )
        })}
      </div>
      )}
    </Card>
  )
}

function SrReview({
  deck,
  loading,
  index,
  revealed,
  onReveal,
  onGrade,
  onRestart,
}: {
  deck: DeckCard[] | null
  loading: boolean
  index: number
  revealed: boolean
  onReveal: () => void
  onGrade: (grade: number) => void
  onRestart: () => void
}) {
  if (loading) {
    return (
      <div className="text-center py-12 text-zinc-400">
        <div className="w-8 h-8 border-2 border-white/10 border-t-cyan-400 rounded-full animate-spin mx-auto mb-3" />
        Loading your due cards…
      </div>
    )
  }

  if (!deck) return null

  if (deck.length === 0 || index >= deck.length) {
    return (
      <div className="text-center py-12">
        <CheckCircle className="w-12 h-12 text-emerald-400 mx-auto mb-3" />
        <p className="text-emerald-400 font-medium">
          {deck.length === 0 ? 'No cards due right now' : 'All caught up!'}
        </p>
        <p className="text-zinc-500 text-sm mb-4">
          {deck.length === 0
            ? 'Save cards to your deck, then come back when they’re due.'
            : `You reviewed ${deck.length} card${deck.length === 1 ? '' : 's'}.`}
        </p>
        <Button variant="secondary" size="sm" onClick={onRestart}>
          Reload deck
        </Button>
      </div>
    )
  }

  const card = deck[index]
  return (
    <div>
      <div className="flex items-center justify-between mb-3 text-xs text-zinc-500">
        <span>Card {index + 1} of {deck.length} due</span>
      </div>
      <Card accent padding="lg" className="min-h-44">
        <div className="text-xs font-semibold uppercase tracking-widest text-gradient-brand mb-1.5">Question</div>
        <div className="text-zinc-50 font-medium leading-snug mb-4"><Markdown content={card.q} /></div>
        {revealed && (
          <>
            <div className="border-t border-white/10 my-3" />
            <div className="text-xs font-semibold uppercase tracking-widest text-gradient-brand mb-1.5">Answer</div>
            <div className="text-zinc-200 leading-snug"><Markdown content={card.a} /></div>
          </>
        )}
      </Card>

      {!revealed ? (
        <Button variant="secondary" onClick={onReveal} className="mt-4 w-full">
          Show answer
        </Button>
      ) : (
        <div className="mt-4 grid grid-cols-4 gap-2">
          {GRADES.map((g) => (
            <Button
              key={g.grade}
              variant={g.variant}
              onClick={() => onGrade(g.grade)}
              className={g.className}
            >
              {g.label}
            </Button>
          ))}
        </div>
      )}
    </div>
  )
}
