import { motion } from 'motion/react'
import { CheckCircle, XCircle } from 'lucide-react'
import { cn } from '@/lib/utils'
import type { OptionRowState } from './types'

const rowStyles: Record<OptionRowState, string> = {
  idle: 'border-line bg-surface text-ink hover:border-accent-line hover:bg-accent-wash/50',
  selected: 'border-accent bg-accent-wash text-accent-deep ring-1 ring-inset ring-accent/25',
  correct: 'border-success/40 bg-success-wash text-success',
  incorrect: 'border-danger/40 bg-danger-wash text-danger',
  dimmed: 'border-line bg-paper-deep text-ink-faint',
}

const letterStyles: Record<OptionRowState, string> = {
  idle: 'bg-paper-deep text-ink-soft border border-line',
  selected: 'bg-accent text-white',
  correct: 'bg-success text-white',
  incorrect: 'bg-danger text-white',
  dimmed: 'bg-line text-ink-faint',
}

interface OptionRowProps {
  letter: string
  text: string
  state: OptionRowState
  /** True once feedback is showing — rows lock and recolor semantically. */
  revealed: boolean
  onSelect: () => void
}

/** One shared answer-row treatment for quiz + problem sets (hover / selected / correct / incorrect). */
export function OptionRow({ letter, text, state, revealed, onSelect }: OptionRowProps) {
  return (
    <motion.button
      type="button"
      whileTap={!revealed ? { scale: 0.99 } : undefined}
      onClick={() => {
        if (!revealed) onSelect()
      }}
      disabled={revealed}
      aria-pressed={state === 'selected'}
      className={cn(
        'w-full p-4 border rounded-xl text-left transition-all text-[15px] disabled:cursor-default outline-none',
        'focus-visible:ring-2 focus-visible:ring-accent/60 focus-visible:ring-offset-2 focus-visible:ring-offset-paper',
        rowStyles[state],
      )}
    >
      <div className="flex items-center gap-3">
        <div
          className={cn(
            'w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0',
            letterStyles[state],
          )}
        >
          {state === 'correct' ? (
            <CheckCircle className="w-3.5 h-3.5" />
          ) : state === 'incorrect' ? (
            <XCircle className="w-3.5 h-3.5" />
          ) : (
            letter
          )}
        </div>
        <span className="flex-1">{text}</span>
      </div>
    </motion.button>
  )
}
