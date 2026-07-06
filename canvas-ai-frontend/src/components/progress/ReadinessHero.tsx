import type { Readiness } from '@/lib/api'
import { scoreTone } from '@/lib/score'
import { Card } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { ProgressRing } from '@/components/ui/Progress'

interface ReadinessHeroProps {
  readiness: Readiness
}

/**
 * Exam-readiness hero — semantic ProgressRing + tone label + the weak topics
 * dragging the score down. Carries the page's one highlighter mark (the
 * headline %) and opens the numbered report-card sections ("01").
 */
export function ReadinessHero({ readiness }: ReadinessHeroProps) {
  const score = Math.round(readiness.score_pct)
  const tone = scoreTone(score)

  return (
    <Card padding="lg" elevation={2} className="flex flex-col md:flex-row items-center gap-6">
      <ProgressRing value={score} size={116} strokeWidth={9}>
        <span className="hl font-display text-3xl font-semibold tnum text-ink">{score}%</span>
        <span className="text-[10px] text-ink-faint uppercase tracking-widest mt-0.5">ready</span>
      </ProgressRing>

      <div className="flex-1 min-w-0 w-full text-center md:text-left">
        <div className="section-head mb-3">
          <span className="section-num">01</span>
          <h2 className="text-sm font-semibold text-ink tracking-tight">Exam readiness</h2>
        </div>
        <h3 className={`font-display text-xl font-semibold tracking-tight mb-1.5 ${tone.text}`}>{tone.label}</h3>
        <p className="text-sm text-ink-soft mb-3">
          {readiness.has_past_papers
            ? 'Weighted by how often each topic shows up on your past papers.'
            : 'Based on your topic mastery. Upload a past paper to weight by what’s actually tested.'}
          {readiness.confidence === 'low' && ' Study more to sharpen this estimate.'}
        </p>
        {readiness.gaps.length > 0 ? (
          <div className="flex flex-wrap gap-2 justify-center md:justify-start items-center">
            <span className="text-xs text-ink-soft">Biggest gaps:</span>
            {readiness.gaps.map((gap) => (
              <Badge key={gap} tone="warning">{gap}</Badge>
            ))}
          </div>
        ) : (
          <p className="text-xs text-success">No major gaps — keep reviewing to hold your edge.</p>
        )}
      </div>
    </Card>
  )
}
