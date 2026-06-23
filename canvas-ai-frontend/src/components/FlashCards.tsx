import { useMemo, useState } from 'react'
import { Shuffle, RotateCcw, Download, Save, Brain, CheckCircle } from 'lucide-react'
import { Markdown } from '@/components/ui/Markdown'
import {
  saveFlashcards,
  getFlashcardDeck,
  reviewFlashcard,
  exportFlashcardsAnki,
  type DeckCard,
} from '@/lib/api'
import { showError, showSuccess } from '@/lib/toast'

export type Flashcard = { q: string; a: string }

// SM-2 recall grades surfaced as four buttons.
const GRADES: { label: string; grade: number; tone: string }[] = [
  { label: 'Again', grade: 1, tone: 'bg-red-600 hover:bg-red-500' },
  { label: 'Hard', grade: 3, tone: 'bg-amber-600 hover:bg-amber-500' },
  { label: 'Good', grade: 4, tone: 'bg-cyan-600 hover:bg-cyan-500' },
  { label: 'Easy', grade: 5, tone: 'bg-emerald-600 hover:bg-emerald-500' },
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
  const [studyMode, setStudyMode] = useState<'all'|'hide-answers'|'typing'|'sr'>('all')
  const [typingAnswers, setTypingAnswers] = useState<Record<number,string>>({})

  // Spaced-repetition review state (only used when courseId is provided).
  const [saving, setSaving] = useState(false)
  const [deck, setDeck] = useState<DeckCard[] | null>(null)
  const [deckLoading, setDeckLoading] = useState(false)
  const [srIndex, setSrIndex] = useState(0)
  const [srRevealed, setSrRevealed] = useState(false)

  const handleSaveToDeck = async () => {
    if (!courseId) return
    setSaving(true)
    try {
      const res = await saveFlashcards(courseId, cards)
      showSuccess(`Saved ${res.saved} card${res.saved === 1 ? '' : 's'} to your deck${res.skipped ? ` (${res.skipped} already there)` : ''}`)
    } catch (e) {
      showError(e instanceof Error ? e.message : 'Failed to save cards')
    } finally {
      setSaving(false)
    }
  }

  const loadDeck = async () => {
    if (!courseId) return
    setDeckLoading(true)
    try {
      const d = await getFlashcardDeck(courseId, userId)
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
      await reviewFlashcard(card.id, grade, userId)
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
    <div className="bg-zinc-900/60 rounded-lg border border-zinc-800 p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-bold text-zinc-50">{title} ({cards.length})</h3>
        <div className="flex items-center gap-2">
          <select
            value={studyMode}
            onChange={e => { setStudyMode(e.target.value as any); setShowBack({}); setTypingAnswers({}) }}
            className="bg-zinc-800 border border-zinc-700 text-zinc-50 rounded-lg px-3 py-2 text-sm focus:ring-cyan-500/50"
            title="Study mode"
          >
            <option value="all">Flip to reveal</option>
            <option value="hide-answers">Prompt only</option>
            <option value="typing">Typing practice</option>
          </select>

          <button onClick={shuffled} className="px-3 py-2 border border-zinc-700 rounded-lg text-sm text-zinc-400 hover:bg-zinc-800 flex items-center gap-2">
            <Shuffle className="w-4 h-4" /> Shuffle
          </button>
          <button onClick={reset} className="px-3 py-2 border border-zinc-700 rounded-lg text-sm text-zinc-400 hover:bg-zinc-800 flex items-center gap-2">
            <RotateCcw className="w-4 h-4" /> Reset
          </button>
          <button onClick={() => downloadCSV(cards)} className="px-3 py-2 border border-zinc-700 rounded-lg text-sm text-zinc-400 hover:bg-zinc-800 flex items-center gap-2">
            <Download className="w-4 h-4" /> CSV
          </button>
          {courseId && (
            <>
              <button
                onClick={() => void handleAnkiExport()}
                className="px-3 py-2 border border-zinc-700 rounded-lg text-sm text-zinc-400 hover:bg-zinc-800 flex items-center gap-2"
                title="Export your saved deck to Anki, keeping spaced-repetition state"
              >
                <Download className="w-4 h-4" /> Anki
              </button>
              <button
                onClick={() => void handleSaveToDeck()}
                disabled={saving}
                className="px-3 py-2 border border-zinc-700 rounded-lg text-sm text-zinc-400 hover:bg-zinc-800 disabled:opacity-50 flex items-center gap-2"
              >
                <Save className="w-4 h-4" /> {saving ? 'Saving…' : 'Save to deck'}
              </button>
              <button
                onClick={() => (studyMode === 'sr' ? setStudyMode('all') : enterSrMode())}
                className={`px-3 py-2 rounded-lg text-sm flex items-center gap-2 ${
                  studyMode === 'sr' ? 'bg-cyan-600 text-white' : 'border border-zinc-700 text-zinc-400 hover:bg-zinc-800'
                }`}
              >
                <Brain className="w-4 h-4" /> Review
              </button>
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
              <div className="relative h-44 w-full transition-transform duration-500 [transform-style:preserve-3d] rounded-xl border border-zinc-700 bg-zinc-900 p-4"
                   style={{ transform: (studyMode==='all' && flipped) ? 'rotateY(180deg)' : 'none' }}
                   onClick={() => studyMode==='all' && setShowBack(s => ({...s, [idx]: !s[idx]}))}
              >
                {/* Front */}
                <div className="absolute inset-0 backface-hidden">
                  <div className="text-xs text-zinc-400 mb-1">Question</div>
                  <div className="text-zinc-50 font-medium leading-snug overflow-hidden max-h-32"><Markdown content={q} /></div>
                  {studyMode==='all' && (
                    <div className="absolute bottom-3 right-3 text-xs text-zinc-400">Click to flip</div>
                  )}
                </div>

                {/* Back (answer) */}
                <div className="absolute inset-0 rotate-y-180 backface-hidden">
                  <div className="text-xs text-zinc-400 mb-1">Answer</div>
                  <div className="text-zinc-50 leading-snug overflow-hidden max-h-32"><Markdown content={a} /></div>
                </div>
              </div>

              {/* Hide-answers mode: show only prompts */}
              {studyMode==='hide-answers' && (
                <div className="absolute inset-0 rounded-xl border border-dashed border-zinc-700 bg-zinc-800/60 p-4">
                  <div className="text-xs text-zinc-400 mb-1">Prompt</div>
                  <div className="text-zinc-200 font-medium leading-snug overflow-hidden max-h-32"><Markdown content={q} /></div>
                </div>
              )}

              {/* Typing practice mode */}
              {studyMode==='typing' && (
                <div className="absolute inset-0 rounded-xl border border-cyan-500/30 bg-cyan-500/10 p-4 flex flex-col">
                  <div className="text-xs text-cyan-400 mb-1">Type your answer</div>
                  <textarea
                    className="flex-1 resize-none rounded-lg bg-zinc-800 border border-cyan-500/30 px-3 py-2 text-sm text-zinc-50 focus:ring-2 focus:ring-cyan-500/50"
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
    </div>
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
        <div className="w-8 h-8 border-2 border-zinc-700 border-t-cyan-500 rounded-full animate-spin mx-auto mb-3" />
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
        <button onClick={onRestart} className="px-4 py-2 border border-zinc-700 rounded-lg text-sm text-zinc-300 hover:bg-zinc-800">
          Reload deck
        </button>
      </div>
    )
  }

  const card = deck[index]
  return (
    <div>
      <div className="flex items-center justify-between mb-3 text-xs text-zinc-500">
        <span>Card {index + 1} of {deck.length} due</span>
      </div>
      <div className="rounded-xl border border-zinc-700 bg-zinc-900 p-6 min-h-44">
        <div className="text-xs text-zinc-400 mb-1">Question</div>
        <div className="text-zinc-50 font-medium leading-snug mb-4"><Markdown content={card.q} /></div>
        {revealed && (
          <>
            <div className="border-t border-zinc-800 my-3" />
            <div className="text-xs text-zinc-400 mb-1">Answer</div>
            <div className="text-zinc-200 leading-snug"><Markdown content={card.a} /></div>
          </>
        )}
      </div>

      {!revealed ? (
        <button
          onClick={onReveal}
          className="mt-4 w-full bg-zinc-800 border border-zinc-700 text-zinc-200 py-2.5 rounded-lg hover:bg-zinc-700 text-sm font-medium"
        >
          Show answer
        </button>
      ) : (
        <div className="mt-4 grid grid-cols-4 gap-2">
          {GRADES.map((g) => (
            <button
              key={g.grade}
              onClick={() => onGrade(g.grade)}
              className={`py-2.5 rounded-lg text-white text-sm font-medium transition-colors ${g.tone}`}
            >
              {g.label}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
