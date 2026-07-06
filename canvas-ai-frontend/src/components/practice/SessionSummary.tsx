import { motion } from 'motion/react'
import { Trophy, Clock, CheckCircle, Target, BookOpen, RotateCcw, Brain } from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { Card } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { ProgressRing } from '@/components/ui/Progress'
import { scoreTone } from '@/lib/score'
import { cn } from '@/lib/utils'
import { formatTime } from './format'
import type { ModeChangeHandler } from './types'
import type { PracticeController } from './usePracticeSession'

interface SessionSummaryProps {
  practice: PracticeController
  onModeChange?: ModeChangeHandler
}

/** Problem-set summary: semantic score ring, verdict banner, next actions. */
export function SessionSummary({ practice, onModeChange }: SessionSummaryProps) {
  const session = practice.session
  if (!session) return null

  const correctCount = session.userAnswers.filter(
    (a, i) => a === session.problems[i].correct_answer,
  ).length
  const tone = scoreTone(session.score)

  const goAnalytics = () =>
    onModeChange
      ? onModeChange('analytics')
      : window.dispatchEvent(new CustomEvent('navigateToAnalytics'))

  return (
    <div className="max-w-3xl mx-auto px-5 py-8">
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, ease: 'easeOut' }}
      >
        <Card accent padding="lg">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-11 h-11 rounded-xl bg-gradient-brand-soft border border-cyan-400/15 flex items-center justify-center flex-shrink-0 glow-brand-sm">
              <Trophy className="w-5 h-5 text-cyan-300" />
            </div>
            <div className="min-w-0">
              <h2 className="text-xl font-semibold text-zinc-50 mb-1">Practice complete</h2>
              <div className="flex flex-wrap items-center gap-1.5">
                <Badge tone="accent">{practice.selectedTopic}</Badge>
                <Badge tone="neutral">{practice.difficulty}</Badge>
              </div>
            </div>
          </div>

          {/* Score ring + stats */}
          <div className="flex flex-col sm:flex-row items-center gap-6 mb-6">
            <ProgressRing value={session.score} size={132} strokeWidth={10}>
              <span className={cn('text-3xl font-bold leading-none', tone.text)}>
                {session.score}%
              </span>
              <span className="text-xs text-zinc-500 mt-1">{tone.label}</span>
            </ProgressRing>

            <div className="grid grid-cols-2 gap-3 flex-1 w-full">
              <div className="bg-white/[0.04] border border-white/10 rounded-lg p-4 text-center">
                <div className="text-2xl font-bold mb-0.5 text-emerald-400">
                  {correctCount}/{session.problems.length}
                </div>
                <div className="text-xs text-zinc-500">Correct</div>
              </div>
              <div className="bg-white/[0.04] border border-white/10 rounded-lg p-4 text-center">
                <div className="text-2xl font-bold mb-0.5 text-zinc-100 flex items-center justify-center gap-1.5">
                  <Clock className="w-4 h-4 text-cyan-300" />
                  {formatTime(practice.timeElapsed)}
                </div>
                <div className="text-xs text-zinc-500">Time</div>
              </div>
            </div>
          </div>

          {/* Verdict banner — semantic tones only */}
          {session.score >= 80 ? (
            <div className="mb-6 bg-emerald-500/10 border border-emerald-500/20 rounded-lg p-3 flex items-center gap-3">
              <CheckCircle className="w-5 h-5 text-emerald-400 flex-shrink-0" />
              <p className="text-sm text-emerald-400">
                Strong mastery of {practice.selectedTopic}. Try harder difficulty or new topics.
              </p>
            </div>
          ) : session.score >= 60 ? (
            <div className="mb-6 bg-amber-500/10 border border-amber-500/20 rounded-lg p-3 flex items-center gap-3">
              <Target className="w-5 h-5 text-amber-400 flex-shrink-0" />
              <p className="text-sm text-amber-400">
                Good progress on {practice.selectedTopic}. A bit more practice will help.
              </p>
            </div>
          ) : (
            <div className="mb-6 bg-rose-500/10 border border-rose-500/20 rounded-lg p-3 flex items-center gap-3">
              <BookOpen className="w-5 h-5 text-rose-400 flex-shrink-0" />
              <p className="text-sm text-rose-400">
                Review {practice.selectedTopic} and try easier problems first.
              </p>
            </div>
          )}

          <div className="flex flex-wrap gap-3">
            <Button onClick={practice.resetSession} leftIcon={<RotateCcw className="w-4 h-4" />}>
              Practice again
            </Button>
            <Button variant="secondary" onClick={goAnalytics} leftIcon={<Brain className="w-4 h-4" />}>
              View analytics
            </Button>
          </div>
        </Card>
      </motion.div>
    </div>
  )
}
