// Graded report: score ring, letter grade + time efficiency, per-concept
// breakdown, and per-question AI-judge verdicts.
import { motion } from 'motion/react'
import { RotateCcw, Trophy } from 'lucide-react'
import { Badge } from '@/components/ui/Badge'
import { Button } from '@/components/ui/Button'
import { Card } from '@/components/ui/Card'
import { Markdown } from '@/components/ui/Markdown'
import { ProgressBar, ProgressRing } from '@/components/ui/Progress'
import { scoreTone } from '@/lib/score'
import { readinessLabel, verdictMeta } from './examMeta'
import type { BreakdownItem, ExamResults as ExamResultsData, Verdict } from './types'

interface ExamResultsProps {
  results: ExamResultsData
  onNewExam: () => void
}

const verdictOf = (b: BreakdownItem): Verdict =>
  (b.verdict as Verdict) ??
  ((b.pointsEarned ?? 0) >= (b.points ?? 0) ? 'correct' : (b.pointsEarned ?? 0) > 0 ? 'partial' : 'incorrect')

interface TopicRow {
  topic: string
  earned: number
  possible: number
  pct: number
}

/** Per-topic performance: prefer the server payload, else derive from breakdown. */
function buildTopicRows(r: ExamResultsData, breakdown: BreakdownItem[]): TopicRow[] {
  if (r.topicPerformance && typeof r.topicPerformance === 'object') {
    return Object.entries(r.topicPerformance)
      .map(([topic, v]) => {
        const earned = v?.earned ?? 0
        const possible = v?.possible ?? 0
        const pct = v?.percentage ?? (possible > 0 ? Math.round((earned / possible) * 100) : 0)
        return { topic, earned, possible, pct }
      })
      .sort((a, b) => a.pct - b.pct)
  }
  const acc = breakdown.reduce<Record<string, { earned: number; possible: number }>>((map, b) => {
    const t = b.topic ?? 'General'
    const prev = map[t] ?? { earned: 0, possible: 0 }
    return { ...map, [t]: { earned: prev.earned + (b.pointsEarned ?? 0), possible: prev.possible + (b.points ?? 0) } }
  }, {})
  return Object.entries(acc)
    .map(([topic, v]) => ({
      topic,
      earned: v.earned,
      possible: v.possible,
      pct: v.possible > 0 ? Math.round((v.earned / v.possible) * 100) : 0,
    }))
    .sort((a, b) => a.pct - b.pct)
}

export function ExamResults({ results: r, onNewExam }: ExamResultsProps) {
  const breakdown: BreakdownItem[] = Array.isArray(r.breakdown) ? r.breakdown : []
  const tallies = breakdown.reduce(
    (acc, b) => ({ ...acc, [verdictOf(b)]: acc[verdictOf(b)] + 1 }),
    { correct: 0, partial: 0, incorrect: 0 } as Record<Verdict, number>,
  )
  const topicRows = buildTopicRows(r, breakdown)
  const ringPct = Math.max(0, Math.min(100, r.percentage))
  const tone = scoreTone(ringPct)

  const stats = [
    { value: `${r.percentage}%`, label: 'Score', cls: tone.text },
    { value: `${r.correctAnswers}/${r.totalQuestions}`, label: 'Correct', cls: 'text-success' },
    { value: `${r.earnedPoints}/${r.totalPoints}`, label: 'Points', cls: 'text-ink' },
    { value: `${r.timeSpent}m`, label: 'Time', cls: 'text-ink' },
  ]

  return (
    <div className="max-w-3xl mx-auto px-5 py-8 space-y-6">
      {/* Graded-report hero — big score ring, verdict, letter grade, tally */}
      <motion.div
        initial={{ opacity: 0, y: 14 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
      >
        <div className="mb-6 text-center">
          <p className="text-[11px] font-medium uppercase tracking-[0.16em] text-ink-faint mb-1.5">
            Your graded report
          </p>
          <h1 className="font-display text-[28px] font-semibold tracking-tight text-ink">{readinessLabel(r.percentage)}</h1>
        </div>

        <Card padding="lg">
          <div className="flex flex-col sm:flex-row items-center gap-7">
            {/* Big radial score — semantic scoreTone color */}
            <ProgressRing value={ringPct} size={156} strokeWidth={10}>
              <span className={`text-4xl font-bold tracking-tight tnum ${tone.text}`}>{r.percentage}%</span>
              <span className="text-[10px] text-ink-faint mt-0.5">Readiness</span>
            </ProgressRing>

            <div className="flex-1 text-center sm:text-left">
              <div className="flex items-center justify-center sm:justify-start gap-2.5 mb-2">
                <Trophy className="w-4 h-4 text-accent" />
                <span className="text-sm font-semibold text-ink">{readinessLabel(r.percentage)}</span>
                {/* Letter grade — serif display with the page's one highlighter mark */}
                {r.letterGrade && (
                  <span className="hl font-display text-2xl font-semibold leading-none text-ink px-1">
                    {r.letterGrade}
                  </span>
                )}
              </div>
              <p className="text-sm text-ink-soft">
                {r.correctAnswers} of {r.totalQuestions} correct &middot; {r.earnedPoints}/{r.totalPoints} points
              </p>
              {typeof r.timeEfficiency === 'string' && (
                <p className="text-xs text-ink-faint mt-1">{r.timeEfficiency}</p>
              )}
              {/* Verdict tally */}
              <div className="mt-4 flex flex-wrap items-center gap-2 justify-center sm:justify-start">
                {(['correct', 'partial', 'incorrect'] as Verdict[]).map((v) => {
                  const m = verdictMeta(v)
                  return (
                    <Badge key={v} tone={m.badgeTone} icon={<m.Icon />}>
                      {tallies[v]} {m.label.toLowerCase()}
                    </Badge>
                  )
                })}
              </div>
            </div>
          </div>
        </Card>
      </motion.div>

      {/* Quick stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {stats.map((s, i) => (
          <motion.div
            key={s.label}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.05 * i }}
          >
            <Card padding="sm" className="text-center">
              <div className={`font-display text-2xl font-semibold tnum mb-0.5 ${s.cls}`}>{s.value}</div>
              <div className="text-xs text-ink-faint">{s.label}</div>
            </Card>
          </motion.div>
        ))}
      </div>

      {/* Per-concept performance */}
      {topicRows.length > 0 && (
        <Card padding="md">
          <div className="section-head mb-3.5">
            <span className="section-num">01</span>
            <h3 className="text-sm font-semibold text-ink">Performance by concept</h3>
          </div>
          <div className="space-y-3">
            {topicRows.map((t) => (
              <div key={t.topic}>
                <div className="flex items-center justify-between mb-1.5 text-xs">
                  <span className="text-ink font-medium">{t.topic}</span>
                  <span className="text-ink-faint tnum">
                    {t.earned}/{t.possible} pts &middot; <span className="text-ink">{t.pct}%</span>
                  </span>
                </div>
                <ProgressBar value={t.pct} label={`${t.topic} score`} />
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Per-question AI-judge verdicts */}
      {breakdown.length > 0 && (
        <div className="space-y-2.5">
          <div className="section-head px-1 mb-3.5">
            <span className="section-num">02</span>
            <h3 className="text-sm font-semibold text-ink">Question-by-question review</h3>
          </div>
          {breakdown.map((b, i) => {
            const v = verdictOf(b)
            const m = verdictMeta(v)
            return (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.25, delay: Math.min(0.02 * i, 0.1) }}
                className={`border rounded-xl p-4 ${m.tone}`}
              >
                <div className="flex items-start justify-between gap-3 mb-2">
                  <span className="inline-flex items-center gap-2">
                    <span className="section-num">Question {String(i + 1).padStart(2, '0')}</span>
                    <Badge tone="neutral">{b.topic ?? 'General'}</Badge>
                  </span>
                  <span className={`inline-flex items-center gap-1.5 text-xs font-semibold tnum ${m.text}`}>
                    <m.Icon className="w-3.5 h-3.5" />
                    {m.label} &middot; {b.pointsEarned ?? 0}/{b.points ?? 0} pts
                    {typeof b.timeSpent === 'number' ? ` · ${b.timeSpent}s` : ''}
                  </span>
                </div>
                <div className="text-sm text-ink mb-1.5 leading-relaxed">
                  <Markdown content={b.question} />
                </div>
                {b.userAnswer && (
                  <p className="text-xs text-ink-soft mb-1">
                    <span className="text-ink-faint">Your answer: </span>{b.userAnswer}
                  </p>
                )}
                {/* AI-judge explanation — typeset as prose on the sheet */}
                {b.gradeReason && (
                  <div className="text-xs italic text-ink-soft">
                    <Markdown content={b.gradeReason} className="text-xs" />
                  </div>
                )}
                {b.mistakeExplanation && (
                  <div className="mt-2 flex items-start gap-1.5 rounded-lg bg-warning-wash border border-warning/25 p-2.5">
                    <div className="min-w-0 flex-1 text-xs">
                      <span className="block font-medium text-warning mb-0.5">Where it went wrong</span>
                      <Markdown content={b.mistakeExplanation} className="text-xs" />
                    </div>
                  </div>
                )}
              </motion.div>
            )
          })}
        </div>
      )}

      <div className="flex gap-3 pt-1">
        <Button size="lg" onClick={onNewExam} leftIcon={<RotateCcw className="w-4 h-4" />}>
          New Exam
        </Button>
      </div>
    </div>
  )
}
