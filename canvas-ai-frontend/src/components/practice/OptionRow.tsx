import { motion } from 'motion/react'
import { CheckCircle, XCircle } from 'lucide-react'
import { cn } from '@/lib/utils'
import type { OptionRowState } from './types'

const rowStyles: Record<OptionRowState, string> = {
  idle: 'border-zinc-700 text-zinc-200 hover:border-cyan-400/40 hover:bg-cyan-400/5',
  selected: 'border-transparent bg-gradient-brand-soft text-cyan-200 ring-2 ring-cyan-400/40',
  correct: 'border-emerald-500/70 bg-emerald-500/10 text-emerald-300',
  incorrect: 'border-rose-500/70 bg-rose-500/10 text-rose-300',
  dimmed: 'border-zinc-700/60 bg-zinc-800/40 text-zinc-500',
}

const letterStyles: Record<OptionRowState, string> = {
  idle: 'bg-zinc-700 text-zinc-400',
  selected: 'bg-cyan-500 text-white',
  correct: 'bg-emerald-500 text-white',
  incorrect: 'bg-rose-500 text-white',
  dimmed: 'bg-zinc-700 text-zinc-400',
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
        'focus-visible:ring-2 focus-visible:ring-cyan-400/70 focus-visible:ring-offset-2 focus-visible:ring-offset-[#0a0c14]',
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
