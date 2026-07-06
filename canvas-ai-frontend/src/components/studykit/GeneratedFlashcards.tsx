// src/components/studykit/GeneratedFlashcards.tsx — the auto-generated deck: flip cards + spaced repetition entry
import { useEffect, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { BookOpen, ChevronLeft, ChevronRight, Layers } from 'lucide-react'
import { Badge } from '@/components/ui/Badge'
import { Button } from '@/components/ui/Button'
import { Card } from '@/components/ui/Card'
import { Markdown } from '@/components/ui/Markdown'
import { EmptyState } from '@/components/ui/States'
import Flashcards from '@/components/FlashCards'
import type { Flashcard } from './noteUtils'

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
        {/* Front — Question: white index-card sheet */}
        <div className="absolute inset-0 [backface-visibility:hidden] rounded-2xl card-surface p-7 flex flex-col">
          <div className="flex items-center gap-2 font-mono text-[11px] font-medium uppercase tracking-[0.08em] text-ink-faint">
            Question
          </div>
          <div className="flex-1 flex items-center justify-center text-center px-2">
            <p className="text-lg font-medium text-ink leading-relaxed">{card.q}</p>
          </div>
          <p className="text-xs text-ink-faint text-center">Tap to reveal answer</p>
        </div>
        {/* Back — Answer: pen-blue-washed sheet */}
        <div className="absolute inset-0 [backface-visibility:hidden] [transform:rotateY(180deg)] rounded-2xl bg-accent-wash border border-accent-line p-7 flex flex-col">
          <div className="flex items-center gap-2 font-mono text-[11px] font-medium uppercase tracking-[0.08em] text-accent-deep">
            Answer
          </div>
          <div className="flex-1 flex items-center justify-center text-center overflow-auto px-2">
            <div className="text-sm text-ink max-w-none">
              <Markdown content={card.a} />
            </div>
          </div>
          <p className="text-xs text-accent-deep/70 text-center">Tap to flip back</p>
        </div>
      </motion.div>
    </button>
  )
}

/** Prominent deck presentation with navigation + progress. */
export function FlashcardDeck({ cards }: { cards: Flashcard[] }) {
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
        <div className="flex items-center gap-2 text-sm text-ink-soft">
          <Layers className="w-4 h-4 text-accent" />
          Card <span className="text-ink font-semibold tnum">{safeIndex + 1}</span> of {cards.length}
        </div>
        <div className="flex items-center gap-1.5">
          {cards.map((_, i) => (
            <span
              key={i}
              className={`h-1.5 rounded-full transition-all ${
                i === safeIndex
                  ? 'w-6 bg-accent'
                  : seen.has(i)
                  ? 'w-1.5 bg-accent/40'
                  : 'w-1.5 bg-line-strong'
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
        <span className="text-xs text-ink-faint tnum">{seen.size}/{cards.length} reviewed</span>
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

interface GeneratedFlashcardsProps {
  cards: Flashcard[]
}

/** Sidebar panel presenting the deck generated alongside the current note. */
export function GeneratedFlashcardsPanel({ cards }: GeneratedFlashcardsProps) {
  if (cards.length === 0) {
    return (
      <Card padding="lg">
        <EmptyState
          icon={<Layers />}
          title="No flashcards for this note"
          description="This note didn't include a deck. Generate a fresh note to get one."
        />
      </Card>
    )
  }
  return (
    <Card padding="lg" className="sticky top-6">
      <h3 className="text-base font-semibold text-ink mb-1 flex items-center gap-2">
        <Layers className="w-5 h-5 text-accent" />
        Flashcard Deck
        <Badge tone="marker" className="ml-1">{cards.length} cards</Badge>
      </h3>
      <p className="text-xs text-ink-soft mb-5">Auto-generated from your notes. Flip to study.</p>
      <FlashcardDeck cards={cards} />
    </Card>
  )
}

interface SpacedRepetitionPanelProps {
  cards: Flashcard[]
  courseId: string
  userId: string
}

/** Save-to-deck + SM-2 review entry — data layer lives in FlashCards.tsx, unchanged. */
export function SpacedRepetitionPanel({ cards, courseId, userId }: SpacedRepetitionPanelProps) {
  if (cards.length === 0) return null
  return (
    <Card padding="lg">
      <h3 className="text-base font-semibold text-ink mb-1 flex items-center gap-2">
        <BookOpen className="w-5 h-5 text-accent" />
        Spaced Repetition
      </h3>
      <p className="text-xs text-ink-soft mb-5">
        Save this deck and review it on an SM-2 schedule to lock it into memory.
      </p>
      <Flashcards cards={cards} courseId={courseId} userId={userId} />
    </Card>
  )
}
