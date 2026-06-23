import { useMemo, useState } from 'react'
import { Shuffle, RotateCcw, Download } from 'lucide-react'
import { Markdown } from '@/components/ui/Markdown'

export type Flashcard = { q: string; a: string }

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
  title = 'Flashcards'
}: { cards: Flashcard[]; title?: string }) {
  const [order, setOrder] = useState<number[]>(() => cards.map((_, i) => i))
  const [showBack, setShowBack] = useState<Record<number, boolean>>({})
  const [studyMode, setStudyMode] = useState<'all'|'hide-answers'|'typing'>('all')
  const [typingAnswers, setTypingAnswers] = useState<Record<number,string>>({})

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
        </div>
      </div>

      {/* Grid of cards */}
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
    </div>
  )
}
