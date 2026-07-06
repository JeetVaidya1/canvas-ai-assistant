import { useMemo, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { Shuffle, RotateCcw, Download, Save, Brain, CheckCircle } from 'lucide-react'
import { Markdown } from '@/components/ui/Markdown'
import { Button } from '@/components/ui/Button'
import { Card } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { Select } from '@/components/ui/Select'
import { ProgressBar } from '@/components/ui/Progress'
import {
  exportFlashcardsAnki,
  type DeckCard,
} from '@/lib/api'
import { flashcardDeckOptions, useSaveFlashcards, useReviewFlashcard } from '@/hooks/useFlashcards'
import { showError, showSuccess } from '@/lib/toast'

export type Flashcard = { q: string; a: string }

type StudyMode = 'all' | 'hide-answers' | 'typing' | 'sr'

const STUDY_MODE_OPTIONS = [
  { value: 'all', label: 'Flip to reveal' },
  { value: 'hide-answers', label: 'Prompt only' },
  { value: 'typing', label: 'Typing practice' },
]

type GradeVariant = 'primary' | 'secondary' | 'ghost' | 'danger'

// SM-2 recall grades surfaced as four buttons — semantic ink text on wash tints.
const GRADES: { label: string; grade: number; variant: GradeVariant; className: string }[] = [
  { label: 'Again', grade: 1, variant: 'secondary', className: '!bg-danger-wash !border-danger/25 !text-danger hover:!border-danger/50' },
  { label: 'Hard', grade: 3, variant: 'secondary', className: '!bg-warning-wash !border-warning/25 !text-warning hover:!border-warning/50' },
  { label: 'Good', grade: 4, variant: 'secondary', className: '!bg-accent-wash !border-accent-line !text-accent-deep hover:!border-accent/50' },
  { label: 'Easy', grade: 5, variant: 'secondary', className: '!bg-success-wash !border-success/25 !text-success hover:!border-success/50' },
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
    <Card padding="lg">
      <div className="flex items-center justify-between mb-4 flex-wrap gap-3">
        <div className="flex items-center gap-2.5">
          <h3 className="text-lg font-semibold text-ink">{title}</h3>
          <Badge tone="neutral">{cards.length} card{cards.length === 1 ? '' : 's'}</Badge>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          <Select
            value={studyMode === 'sr' ? '' : studyMode}
            options={STUDY_MODE_OPTIONS}
            onChange={(v) => { setStudyMode(v as StudyMode); setShowBack({}); setTypingAnswers({}) }}
            placeholder="Study mode"
            ariaLabel="Study mode"
            className="w-44"
          />

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
              {/* Card — white index-card sheet with hairline border */}
              <div className="card-surface relative h-44 w-full transition-transform duration-500 [transform-style:preserve-3d] rounded-xl p-4 hover:border-line-strong"
                   style={{ transform: (studyMode==='all' && flipped) ? 'rotateY(180deg)' : 'none' }}
                   onClick={() => studyMode==='all' && setShowBack(s => ({...s, [idx]: !s[idx]}))}
              >
                {/* Front */}
                <div className="absolute inset-0 p-4 backface-hidden">
                  <div className="font-mono text-[11px] font-medium uppercase tracking-[0.08em] text-ink-faint mb-1.5">Question</div>
                  <div className="text-ink font-medium leading-snug overflow-hidden max-h-32"><Markdown content={q} /></div>
                  {studyMode==='all' && (
                    <div className="absolute bottom-3 right-3 text-xs text-ink-faint">Click to flip</div>
                  )}
                </div>

                {/* Back (answer) */}
                <div className="absolute inset-0 p-4 rotate-y-180 backface-hidden">
                  <div className="font-mono text-[11px] font-medium uppercase tracking-[0.08em] text-accent-deep mb-1.5">Answer</div>
                  <div className="text-ink leading-snug overflow-hidden max-h-32"><Markdown content={a} /></div>
                </div>
              </div>

              {/* Hide-answers mode: show only prompts */}
              {studyMode==='hide-answers' && (
                <div className="absolute inset-0 rounded-xl border border-dashed border-line-strong bg-surface p-4">
                  <div className="font-mono text-[11px] font-medium uppercase tracking-[0.08em] text-ink-faint mb-1.5">Prompt</div>
                  <div className="text-ink font-medium leading-snug overflow-hidden max-h-32"><Markdown content={q} /></div>
                </div>
              )}

              {/* Typing practice mode */}
              {studyMode==='typing' && (
                <div className="absolute inset-0 rounded-xl border border-accent-line bg-accent-wash p-4 flex flex-col">
                  <div className="font-mono text-[11px] font-medium uppercase tracking-[0.08em] text-accent-deep mb-1.5">Type your answer</div>
                  <textarea
                    className="flex-1 resize-none rounded-lg bg-surface border border-line px-3 py-2 text-sm text-ink placeholder-ink-faint outline-none focus:border-accent focus:ring-2 focus:ring-accent/20 transition-colors"
                    value={typingAnswers[idx] || ''}
                    onChange={e => setTypingAnswers(s => ({...s, [idx]: e.target.value}))}
                    placeholder="Write your answer here…"
                  />
                  {typingAnswers[idx] && (
                    <div className="mt-2 text-xs text-ink-soft">
                      Correct answer: <span className="text-ink font-medium">{a}</span>
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
      <div className="text-center py-12 text-ink-soft">
        <div className="w-8 h-8 border-2 border-line border-t-accent rounded-full animate-spin mx-auto mb-3" />
        Loading your due cards…
      </div>
    )
  }

  if (!deck) return null

  if (deck.length === 0 || index >= deck.length) {
    return (
      <div className="text-center py-12">
        <CheckCircle className="w-12 h-12 text-success mx-auto mb-3" />
        <p className="text-success font-medium">
          {deck.length === 0 ? 'No cards due right now' : 'All caught up!'}
        </p>
        <p className="text-ink-faint text-sm mb-4">
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
      <div className="flex items-center justify-between gap-3 mb-2 text-xs text-ink-faint">
        <span className="tnum">Card {index + 1} of {deck.length} due</span>
        <Badge tone="warning">{deck.length - index} left</Badge>
      </div>
      {/* Session progress — completed reviews out of the due snapshot. */}
      <ProgressBar
        value={(index / deck.length) * 100}
        color="#2b4acb"
        label="Review session progress"
        className="mb-4"
      />
      <Card padding="lg" className="min-h-44">
        <div className="font-mono text-[11px] font-medium uppercase tracking-[0.08em] text-ink-faint mb-1.5">Question</div>
        <div className="text-ink font-medium leading-snug mb-4"><Markdown content={card.q} /></div>
        {revealed && (
          <>
            <div className="border-t border-line my-3" />
            <div className="font-mono text-[11px] font-medium uppercase tracking-[0.08em] text-accent-deep mb-1.5">Answer</div>
            <div className="text-ink leading-snug"><Markdown content={card.a} /></div>
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
