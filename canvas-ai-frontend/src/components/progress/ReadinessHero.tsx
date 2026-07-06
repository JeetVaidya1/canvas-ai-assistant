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
 * dragging the score down. The single uppercase eyebrow on the Analytics tab.
 */
export function ReadinessHero({ readiness }: ReadinessHeroProps) {
  const score = Math.round(readiness.score_pct)
  const tone = scoreTone(score)

  return (
    <Card accent padding="lg" elevation={2} className="flex flex-col md:flex-row items-center gap-6">
      <ProgressRing value={score} size={116} strokeWidth={9}>
        <span className={`text-3xl font-bold ${tone.text}`}>{score}%</span>
        <span className="text-[10px] text-zinc-500 uppercase tracking-widest mt-0.5">ready</span>
      </ProgressRing>

      <div className="flex-1 text-center md:text-left">
        <p className="text-xs font-semibold uppercase tracking-widest text-gradient-brand mb-1.5">
          Exam readiness
        </p>
        <h2 className={`text-xl font-semibold tracking-tight mb-1.5 ${tone.text}`}>{tone.label}</h2>
        <p className="text-sm text-zinc-400 mb-3">
          {readiness.has_past_papers
            ? 'Weighted by how often each topic shows up on your past papers.'
            : 'Based on your topic mastery. Upload a past paper to weight by what’s actually tested.'}
          {readiness.confidence === 'low' && ' Study more to sharpen this estimate.'}
        </p>
        {readiness.gaps.length > 0 ? (
          <div className="flex flex-wrap gap-2 justify-center md:justify-start items-center">
            <span className="text-xs text-zinc-400">Biggest gaps:</span>
            {readiness.gaps.map((gap) => (
              <Badge key={gap} tone="warning">{gap}</Badge>
            ))}
          </div>
        ) : (
          <p className="text-xs text-emerald-300">No major gaps — keep reviewing to hold your edge.</p>
        )}
      </div>
    </Card>
  )
}
