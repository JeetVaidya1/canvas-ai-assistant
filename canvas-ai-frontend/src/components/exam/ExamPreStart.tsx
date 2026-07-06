// Pre-start confirmation / resume screen shown once an exam exists but the
// clock hasn't started (or the user came back to an in-progress exam).
import { motion } from 'motion/react'
import { Play } from 'lucide-react'
import { BrandMark } from '@/components/ui/BrandMark'
import { Button } from '@/components/ui/Button'
import type { ExamSession } from './types'

interface ExamPreStartProps {
  session: ExamSession
  /** Remaining seconds restored from a previous run (0 for a fresh exam). */
  timeRemaining: number
  onStart: () => void
  onAbandon: () => void
}

export function ExamPreStart({ session, timeRemaining, onStart, onAbandon }: ExamPreStartProps) {
  const totalPoints = session.questions.reduce((a, q) => a + (q.points ?? 0), 0)
  const answeredCount = Object.keys(session.userAnswers).length
  const isResume = answeredCount > 0 || timeRemaining > 0

  const stats = [
    { label: 'Questions', value: session.questions.length },
    { label: 'Points', value: totalPoints },
    { label: 'Time', value: `${session.timeLimit}m` },
  ]

  return (
    <div className="flex min-h-full flex-col items-center justify-center px-4 py-10">
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
        className="w-full max-w-md"
      >
        <div className="mb-7 text-center">
          <BrandMark className="mx-auto mb-5 h-14 w-14 glow-brand-sm" />
          <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-gradient-brand mb-1.5">
            {isResume ? 'Resume your exam' : 'Ready when you are'}
          </p>
          <h1 className="text-2xl font-semibold tracking-tight text-zinc-50 truncate">{session.examName}</h1>
          {isResume && (
            <p className="text-xs text-amber-400/90 mt-2">
              In progress — {answeredCount}/{session.questions.length} answered, clock resumes where you left off.
            </p>
          )}
        </div>

        <div className="grid grid-cols-3 gap-3 mb-7">
          {stats.map((s) => (
            <div key={s.label} className="rounded-2xl border border-white/10 bg-white/[0.03] px-3 py-4 text-center">
              <div className="text-xl font-bold text-zinc-100">{s.value}</div>
              <div className="text-[11px] text-zinc-400 mt-0.5">{s.label}</div>
            </div>
          ))}
        </div>

        <Button
          size="lg"
          onClick={onStart}
          leftIcon={<Play className="w-4 h-4" />}
          className="w-full !py-3.5 !text-base"
        >
          {isResume ? 'Resume exam' : 'Begin exam'}
        </Button>
        <button
          onClick={onAbandon}
          className="mt-4 w-full text-center text-xs text-zinc-500 transition-colors hover:text-zinc-300"
        >
          ← Back to setup
        </button>
      </motion.div>
    </div>
  )
}
