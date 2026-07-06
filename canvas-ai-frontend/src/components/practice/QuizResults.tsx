import { motion } from 'motion/react'
import { Trophy, Clock, RotateCcw, BookOpen, Brain, Target } from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { Card } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { ProgressRing, ProgressBar } from '@/components/ui/Progress'
import { scoreTone } from '@/lib/score'
import { cn } from '@/lib/utils'
import { formatTime } from './format'
import type { ModeChangeHandler } from './types'
import type { QuizController } from './useQuizRun'

interface QuizResultsProps {
  quiz: QuizController
  onModeChange?: ModeChangeHandler
}

/** Quiz summary: semantic score ring, per-topic breakdown, weak-area chips, next actions. */
export function QuizResults({ quiz, onModeChange }: QuizResultsProps) {
  const result = quiz.result
  if (!result) return null

  const scorePct = result.score.pct
  const tone = scoreTone(scorePct)
  const sortedTopics = [...result.by_topic].sort((a, b) => a.pct - b.pct)
  const headline =
    scorePct >= 85 ? 'Outstanding work' : scorePct >= 60 ? 'Solid effort' : 'Good start — keep going'

  const goPractice = () =>
    onModeChange
      ? onModeChange('practice')
      : window.dispatchEvent(new CustomEvent('navigateToPractice'))
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
        <Card accent>
          <div className="flex items-center gap-3 mb-6">
            <div className="w-11 h-11 rounded-xl bg-gradient-brand-soft border border-cyan-400/15 flex items-center justify-center flex-shrink-0 glow-brand-sm">
              <Trophy className="w-5 h-5 text-cyan-300" />
            </div>
            <div>
              <h2 className="text-xl font-semibold text-zinc-50 mb-0.5">{headline}</h2>
              <p className="text-sm text-zinc-400">
                {quiz.selectedTopic} &middot; {quiz.difficulty}
              </p>
            </div>
          </div>

          {/* Score ring + stats */}
          <div className="flex flex-col sm:flex-row items-center gap-6 mb-6">
            <ProgressRing value={scorePct} size={132} strokeWidth={10}>
              <span className={cn('text-3xl font-bold leading-none', tone.text)}>{scorePct}%</span>
              <span className="text-xs text-zinc-500 mt-1">{tone.label}</span>
            </ProgressRing>

            <div className="grid grid-cols-2 gap-3 flex-1 w-full">
              <div className="bg-white/[0.04] border border-white/10 rounded-lg p-4 text-center">
                <div className="text-2xl font-bold mb-0.5 text-emerald-400">
                  {result.score.correct}/{result.score.total}
                </div>
                <div className="text-xs text-zinc-500">Correct</div>
              </div>
              <div className="bg-white/[0.04] border border-white/10 rounded-lg p-4 text-center">
                <div className="text-2xl font-bold mb-0.5 text-zinc-100 flex items-center justify-center gap-1.5">
                  <Clock className="w-4 h-4 text-cyan-300" />
                  {formatTime(quiz.timeElapsed)}
                </div>
                <div className="text-xs text-zinc-500">Time</div>
              </div>
            </div>
          </div>

          {/* By-topic breakdown */}
          {sortedTopics.length > 0 && (
            <div className="mb-6">
              <h3 className="text-sm font-medium text-zinc-300 mb-3">Breakdown by topic</h3>
              <div className="space-y-2.5">
                {sortedTopics.map((t) => (
                  <div key={t.topic}>
                    <div className="flex items-center justify-between mb-1 text-sm">
                      <span className="text-zinc-300 truncate pr-3">{t.topic}</span>
                      <span className="text-zinc-500 flex-shrink-0">
                        {t.correct}/{t.total} &middot; {t.pct}%
                      </span>
                    </div>
                    <ProgressBar value={t.pct} className="h-2" label={`${t.topic} score`} />
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Weak areas */}
          {result.weak_areas.length > 0 && (
            <div className="mb-6 bg-amber-500/10 border border-amber-500/20 rounded-lg p-4">
              <div className="flex items-start gap-2.5">
                <Target className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
                <div>
                  <h4 className="text-sm font-medium text-amber-400 mb-1.5">Worth another look</h4>
                  <div className="flex flex-wrap gap-1.5">
                    {result.weak_areas.map((area) => (
                      <Badge key={area} tone="warning">
                        {area}
                      </Badge>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          <div className="flex flex-wrap gap-3">
            <Button onClick={quiz.resetQuiz} leftIcon={<RotateCcw className="w-4 h-4" />}>
              Drill again
            </Button>
            <Button variant="secondary" onClick={goPractice} leftIcon={<BookOpen className="w-4 h-4" />}>
              Practice weak areas
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
