import type { ReactNode } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { CheckCircle, XCircle } from 'lucide-react'
import { Markdown } from '@/components/ui/Markdown'
import { cn } from '@/lib/utils'

interface FeedbackPanelProps {
  show: boolean
  correct: boolean
  explanation: string
  /** Extra grounded content (mistake analysis, source/concept tags). */
  children?: ReactNode
}

/** Post-answer verdict panel — success ink for correct, danger ink for incorrect. */
export function FeedbackPanel({ show, correct, explanation, children }: FeedbackPanelProps) {
  return (
    <AnimatePresence>
      {show && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          transition={{ duration: 0.3, ease: 'easeOut' }}
          className="overflow-hidden"
        >
          <div
            role="status"
            className={cn(
              'rounded-xl border p-4 mb-5',
              correct ? 'bg-success-wash border-success/25' : 'bg-danger-wash border-danger/25',
            )}
          >
            <div className="flex items-start gap-2.5">
              <motion.div
                initial={{ scale: 0.6, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ type: 'spring', stiffness: 220, damping: 16 }}
                className="flex-shrink-0 mt-0.5"
              >
                {correct ? (
                  <CheckCircle className="w-5 h-5 text-success" />
                ) : (
                  <XCircle className="w-5 h-5 text-danger" />
                )}
              </motion.div>
              <div className="min-w-0 flex-1">
                <h4 className={cn('text-sm font-semibold mb-1', correct ? 'text-success' : 'text-danger')}>
                  {correct ? 'Correct!' : 'Not quite right'}
                </h4>
                <div className="text-sm text-ink-soft">
                  <Markdown content={explanation} />
                </div>
                {children}
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
