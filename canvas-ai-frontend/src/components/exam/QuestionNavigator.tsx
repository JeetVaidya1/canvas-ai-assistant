// Slide-over question navigator: numbered grid with answered/current legend
// and a finish CTA in the footer.
import { motion, AnimatePresence } from 'motion/react'
import { Flag, GraduationCap, LayoutGrid, X } from 'lucide-react'
import { Button } from '@/components/ui/Button'
import type { ExamSession } from './types'

interface QuestionNavigatorProps {
  open: boolean
  onClose: () => void
  session: ExamSession
  onGoToQuestion: (index: number) => void
  onRequestSubmit: () => void
}

export function QuestionNavigator({ open, onClose, session, onGoToQuestion, onRequestSubmit }: QuestionNavigatorProps) {
  const answeredCount = Object.keys(session.userAnswers).filter((k) => session.userAnswers[k]).length

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 z-30 bg-black/50 backdrop-blur-sm"
          />
          <motion.aside
            initial={{ x: 360 }}
            animate={{ x: 0 }}
            exit={{ x: 360 }}
            transition={{ type: 'spring', stiffness: 320, damping: 34 }}
            className="fixed inset-y-0 right-0 z-40 flex w-[340px] max-w-[88vw] flex-col border-l border-white/10 bg-[#0c0f18]"
          >
            <div className="flex h-14 items-center justify-between px-4 border-b border-white/10">
              <div className="flex items-center gap-2">
                <LayoutGrid className="h-4 w-4 text-cyan-300" />
                <span className="text-sm font-semibold text-zinc-100">Questions</span>
              </div>
              <button
                onClick={onClose}
                className="rounded-lg p-1.5 text-zinc-400 transition-colors hover:bg-white/[0.06] hover:text-zinc-100"
                aria-label="Close navigator"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            {/* Legend */}
            <div className="flex items-center gap-3 px-4 py-3 text-[11px] text-zinc-400 border-b border-white/10">
              <span className="inline-flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-sm bg-gradient-brand" /> Current
              </span>
              <span className="inline-flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-sm bg-emerald-500/40 border border-emerald-500/40" /> Answered
              </span>
              <span className="inline-flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-sm bg-white/[0.04] border border-white/10" /> Empty
              </span>
            </div>

            {/* Numbered grid */}
            <div className="flex-1 overflow-y-auto p-4">
              <div className="grid grid-cols-6 gap-2">
                {session.questions.map((q, index) => {
                  const isAnswered = !!session.userAnswers[q.id]
                  const isCurrent = index === session.currentQuestion
                  return (
                    <button
                      key={q.id}
                      onClick={() => { onGoToQuestion(index); onClose() }}
                      aria-label={`Go to question ${index + 1}${isAnswered ? ', answered' : ', unanswered'}${isCurrent ? ', current' : ''}`}
                      className={`aspect-square rounded-lg text-sm font-bold transition-all ${
                        isCurrent
                          ? 'bg-gradient-brand text-white ring-1 ring-inset ring-cyan-400/30 glow-brand-sm'
                          : isAnswered
                          ? 'bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 hover:bg-emerald-500/20'
                          : 'bg-white/[0.04] border border-white/10 text-zinc-400 hover:border-cyan-400/30 hover:text-zinc-200'
                      }`}
                    >
                      {index + 1}
                    </button>
                  )
                })}
              </div>
            </div>

            {/* Footer — progress + submit */}
            <div className="border-t border-white/10 p-4 space-y-3">
              <div className="flex items-center gap-2 text-xs text-zinc-400">
                <GraduationCap className="h-3.5 w-3.5" />
                {answeredCount}/{session.questions.length} answered
              </div>
              <Button
                onClick={() => { onClose(); onRequestSubmit() }}
                leftIcon={<Flag className="w-3.5 h-3.5" />}
                className="w-full !bg-emerald-600 hover:!bg-emerald-500"
              >
                Finish &amp; submit
              </Button>
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  )
}
