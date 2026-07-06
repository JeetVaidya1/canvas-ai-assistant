// src/components/studykit/GenerationProgress.tsx — honest staged in-flight state (no fake percentages)
import { AnimatePresence, motion } from 'motion/react'
import { Check, Loader2, Sparkles } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Card } from '@/components/ui/Card'
import { LOADING_STAGES } from './noteUtils'

interface GenerationProgressProps {
  /** Index into LOADING_STAGES — advanced on a timer while the request runs. */
  stage: number
  fileCount: number
}

/**
 * Staged, truthful generation state: a checklist of what the backend is doing
 * right now. Deliberately no progress bars or percentages — we don't know how
 * long the model will take, so we never pretend to.
 */
export function GenerationProgress({ stage, fileCount }: GenerationProgressProps) {
  return (
    <Card accent padding="lg" className="glow-brand">
      <div className="mx-auto max-w-md py-8 text-center">
        <div className="relative mx-auto mb-6 h-16 w-16">
          <div className="absolute inset-0 animate-pulse rounded-full bg-gradient-brand opacity-20 blur-xl" />
          <div className="absolute inset-0 animate-spin rounded-full border-4 border-cyan-400/15 border-t-cyan-400" />
          <div className="absolute inset-0 flex items-center justify-center">
            <Sparkles className="h-6 w-6 text-cyan-300" />
          </div>
        </div>
        <h3 className="text-lg font-semibold text-zinc-100">Crafting your study kit</h3>
        <p className="mt-1 text-xs text-zinc-500">
          Grounded in {fileCount} source file{fileCount === 1 ? '' : 's'}
        </p>

        <ul className="mx-auto mt-6 max-w-xs space-y-2.5 text-left" aria-live="polite">
          {LOADING_STAGES.map((label, i) => {
            const done = i < stage
            const active = i === stage
            return (
              <motion.li
                key={label}
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.25, delay: i * 0.05 }}
                className="flex items-center gap-2.5 text-sm"
              >
                <span
                  className={cn(
                    'flex h-5 w-5 flex-shrink-0 items-center justify-center rounded-full border',
                    done && 'border-emerald-400/40 bg-emerald-500/15 text-emerald-300',
                    active && 'border-cyan-400/40 bg-cyan-500/10 text-cyan-300',
                    !done && !active && 'border-white/10 text-zinc-600',
                  )}
                >
                  <AnimatePresence mode="wait" initial={false}>
                    {done ? (
                      <motion.span key="done" initial={{ scale: 0.5, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ duration: 0.15 }}>
                        <Check className="h-3 w-3" />
                      </motion.span>
                    ) : active ? (
                      <motion.span key="active" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.15 }}>
                        <Loader2 className="h-3 w-3 animate-spin" />
                      </motion.span>
                    ) : (
                      <span key="pending" className="h-1 w-1 rounded-full bg-current" />
                    )}
                  </AnimatePresence>
                </span>
                <span className={cn(done ? 'text-zinc-400' : active ? 'text-zinc-100' : 'text-zinc-500')}>
                  {label}
                </span>
              </motion.li>
            )
          })}
        </ul>
      </div>
    </Card>
  )
}
