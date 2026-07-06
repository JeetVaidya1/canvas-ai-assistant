import { useMemo, useState } from 'react'
import { Brain, CheckCircle, Zap } from 'lucide-react'
import { Markdown } from '@/components/ui/Markdown'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import { ErrorState } from '@/components/ui/States'
import { type ReviewItem } from '@/lib/api'
import { useReviewQueue, useGradeReview } from '@/hooks/useReviews'
import { showError } from '@/lib/toast'

// Semantic grade tones: danger (again) / warning (hard) / accent (good) / success (easy).
const GRADES: { label: string; grade: number; tone: string }[] = [
  { label: 'Again', grade: 1, tone: 'bg-danger hover:bg-[#a53a3a]' },
  { label: 'Hard', grade: 3, tone: 'bg-warning hover:bg-[#916312]' },
  { label: 'Good', grade: 4, tone: 'bg-accent hover:bg-accent-deep' },
  { label: 'Easy', grade: 5, tone: 'bg-success hover:bg-[#276b4e]' },
]

interface ReviewPanelProps {
  courseId: string
  userId: string
}

export default function ReviewPanel({ courseId, userId }: ReviewPanelProps) {
  const queueQuery = useReviewQueue(courseId, userId)
  const gradeMutation = useGradeReview(courseId, userId)

  // Active review sessions run off a local snapshot so background refetches
  // (grading invalidates the queue) never reshuffle the cards mid-session.
  const [sessionItems, setSessionItems] = useState<ReviewItem[] | null>(null)
  const [index, setIndex] = useState(0)
  const [revealed, setRevealed] = useState(false)

  const dueItems = useMemo(
    () => (queueQuery.data?.items ?? []).filter((i) => i.due),
    [queueQuery.data],
  )

  const start = () => {
    setSessionItems(dueItems)
    setIndex(0)
    setRevealed(false)
  }

  const finish = () => {
    setSessionItems(null)
    void queueQuery.refetch()
  }

  const grade = async (g: number) => {
    if (!sessionItems || !sessionItems[index]) return
    try {
      await gradeMutation.mutateAsync({ itemId: sessionItems[index].id, grade: g })
    } catch (e) {
      showError(e instanceof Error ? e.message : 'Failed to record review')
    }
    setRevealed(false)
    setIndex((i) => i + 1)
  }

  // ── Collapsed states (no active session) ─────────────────────────────
  if (!sessionItems) {
    if (queueQuery.isError) {
      return (
        <ErrorState
          compact
          title="Couldn't load your review queue."
          onRetry={() => void queueQuery.refetch()}
          retrying={queueQuery.isRefetching}
        />
      )
    }
    if (queueQuery.isPending) return null // stays out of the way while loading
    if (dueItems.length === 0) return null // nothing due — stay out of the way

    const dueCount = dueItems.length
    return (
      <Card accent padding="md" className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <div className="w-11 h-11 rounded-xl bg-accent-wash border border-accent-line flex items-center justify-center flex-shrink-0">
            <Zap className="w-5 h-5 text-accent" />
          </div>
          <div>
            <h2 className="text-sm font-semibold text-ink">
              <span className="text-accent-deep tnum">{dueCount} review{dueCount === 1 ? '' : 's'}</span> due
            </h2>
            <p className="text-xs text-ink-soft mt-0.5">Questions you missed, resurfaced on schedule. Clear them to raise your readiness.</p>
          </div>
        </div>
        <Button onClick={start} leftIcon={<Brain className="w-4 h-4" />} className="flex-shrink-0">
          Review now
        </Button>
      </Card>
    )
  }

  // ── Finished ──────────────────────────────────────────────────────────
  if (index >= sessionItems.length) {
    return (
      <Card padding="lg" className="text-center">
        <div className="w-14 h-14 rounded-2xl bg-success-wash border border-success/25 flex items-center justify-center mx-auto mb-4">
          <CheckCircle className="w-7 h-7 text-success" />
        </div>
        <p className="text-success font-semibold">Review cleared</p>
        <p className="text-ink-soft text-sm mb-5">You worked through {sessionItems.length} item{sessionItems.length === 1 ? '' : 's'}.</p>
        <Button variant="secondary" onClick={finish}>
          Done
        </Button>
      </Card>
    )
  }

  const item = sessionItems[index]
  return (
    <Card padding="lg">
      <div className="flex items-center justify-between mb-4 text-xs">
        <span className="font-medium text-ink-soft tnum">Review {index + 1} of {sessionItems.length}</span>
        <Badge tone="accent">{item.concept}</Badge>
      </div>
      <div className="rounded-xl border border-line bg-paper-deep p-5">
        <div className="text-[11px] font-medium text-ink-faint mb-1.5">From a {item.source} you missed</div>
        <div className="text-ink font-medium leading-snug mb-3"><Markdown content={item.prompt} /></div>
        {revealed && (
          <>
            <div className="border-t border-line my-3" />
            <div className="text-[11px] font-medium text-ink-faint mb-1.5">Answer</div>
            <div className="text-success leading-snug mb-2"><Markdown content={item.answer} /></div>
            {item.explanation && <div className="text-sm text-ink-soft"><Markdown content={item.explanation} /></div>}
          </>
        )}
      </div>

      {!revealed ? (
        <Button variant="secondary" onClick={() => setRevealed(true)} className="mt-4 w-full">
          Show answer
        </Button>
      ) : (
        <div className="mt-4 grid grid-cols-4 gap-2">
          {GRADES.map((g) => (
            <button
              key={g.grade}
              onClick={() => void grade(g.grade)}
              disabled={gradeMutation.isPending}
              className={`py-2.5 rounded-lg text-white text-sm font-medium transition-all active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed ${g.tone}`}
            >
              {g.label}
            </button>
          ))}
        </div>
      )}
    </Card>
  )
}
