import { useNavigate } from 'react-router-dom'
import { useReadiness } from '@/hooks/useReadiness'
import { useUser } from '@/hooks/useUser'
import { scoreTone } from '@/lib/score'
import { Tooltip } from '@/components/ui/Tooltip'

/**
 * Exam-readiness chip for the TopBar: tone dot + tabular % — rendered only
 * once readiness data is in the cache (no spinner, no layout shift games).
 * Click jumps to the course Progress page.
 */
export default function ReadinessChip({ courseId }: { courseId: string }) {
  const userId = useUser()
  const navigate = useNavigate()
  const { data } = useReadiness(courseId, userId, { enabled: !!courseId })

  if (!data) return null

  const score = Math.round(data.score_pct)
  const tone = scoreTone(score)

  return (
    <Tooltip content={`Exam readiness — ${tone.label}`} side="bottom" delay={200}>
      <button
        onClick={() => navigate(`/course/${courseId}/progress`)}
        aria-label={`Exam readiness ${score} percent — ${tone.label}`}
        className="flex items-center gap-1.5 h-7 px-2.5 rounded-full bg-surface border border-line text-xs font-medium text-ink-soft hover:text-ink transition-colors focus-ring"
      >
        <span
          aria-hidden="true"
          className="w-1.5 h-1.5 rounded-full flex-shrink-0"
          style={{ backgroundColor: tone.stroke }}
        />
        <span className="tnum">{score}%</span>
      </button>
    </Tooltip>
  )
}
