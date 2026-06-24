import { useEffect, useState } from 'react'
import { Brain, CheckCircle, Zap } from 'lucide-react'
import { Markdown } from '@/components/ui/Markdown'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { getReviewQueue, gradeReview, type ReviewItem } from '@/lib/api'
import { showError } from '@/lib/toast'

const GRADES: { label: string; grade: number; tone: string }[] = [
  { label: 'Again', grade: 1, tone: 'bg-red-600 hover:bg-red-500' },
  { label: 'Hard', grade: 3, tone: 'bg-amber-600 hover:bg-amber-500' },
  { label: 'Good', grade: 4, tone: 'bg-cyan-600 hover:bg-cyan-500' },
  { label: 'Easy', grade: 5, tone: 'bg-emerald-600 hover:bg-emerald-500' },
]

interface ReviewPanelProps {
  courseId: string
  userId: string
}

export default function ReviewPanel({ courseId, userId }: ReviewPanelProps) {
  const [items, setItems] = useState<ReviewItem[] | null>(null)
  const [active, setActive] = useState(false)
  const [index, setIndex] = useState(0)
  const [revealed, setRevealed] = useState(false)

  const load = async () => {
    try {
      const q = await getReviewQueue(courseId, userId)
      setItems(q.items.filter((i) => i.due))
    } catch {
      setItems([])
    }
  }

  useEffect(() => {
    void load()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [courseId, userId])

  const start = () => {
    setActive(true)
    setIndex(0)
    setRevealed(false)
  }

  const grade = async (g: number) => {
    if (!items || !items[index]) return
    try {
      await gradeReview(items[index].id, g, userId)
    } catch (e) {
      showError(e instanceof Error ? e.message : 'Failed to record review')
    }
    setRevealed(false)
    setIndex((i) => i + 1)
  }

  if (!items || items.length === 0) return null // nothing due — stay out of the way

  const dueCount = items.length

  // Collapsed prompt
  if (!active) {
    return (
      <Card accent padding="md" className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <div className="w-11 h-11 rounded-xl bg-gradient-brand-soft border border-cyan-500/20 flex items-center justify-center flex-shrink-0">
            <Zap className="w-5 h-5 text-cyan-300" />
          </div>
          <div>
            <h2 className="text-sm font-semibold text-zinc-100">
              <span className="text-gradient-brand">{dueCount} review{dueCount === 1 ? '' : 's'}</span> due
            </h2>
            <p className="text-xs text-zinc-500 mt-0.5">Questions you missed, resurfaced on schedule. Clear them to raise your readiness.</p>
          </div>
        </div>
        <Button onClick={start} leftIcon={<Brain className="w-4 h-4" />} className="flex-shrink-0">
          Review now
        </Button>
      </Card>
    )
  }

  // Finished
  if (index >= items.length) {
    return (
      <Card padding="lg" className="text-center">
        <div className="w-14 h-14 rounded-2xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center mx-auto mb-4">
          <CheckCircle className="w-7 h-7 text-emerald-400" />
        </div>
        <p className="text-emerald-400 font-semibold">Review cleared</p>
        <p className="text-zinc-500 text-sm mb-5">You worked through {items.length} item{items.length === 1 ? '' : 's'}.</p>
        <Button variant="secondary" onClick={() => { setActive(false); void load() }}>
          Done
        </Button>
      </Card>
    )
  }

  const item = items[index]
  return (
    <Card padding="lg">
      <div className="flex items-center justify-between mb-4 text-xs">
        <span className="font-medium text-zinc-500">Review {index + 1} of {items.length}</span>
        <span className="text-cyan-300 font-medium bg-gradient-brand-soft border border-cyan-500/15 rounded-full px-2.5 py-0.5">{item.concept}</span>
      </div>
      <div className="rounded-xl border border-zinc-800 bg-zinc-950/60 p-5">
        <div className="text-[11px] font-medium uppercase tracking-wide text-zinc-500 mb-1.5">From a {item.source} you missed</div>
        <div className="text-zinc-50 font-medium leading-snug mb-3"><Markdown content={item.prompt} /></div>
        {revealed && (
          <>
            <div className="border-t border-zinc-800 my-3" />
            <div className="text-[11px] font-medium uppercase tracking-wide text-zinc-500 mb-1.5">Answer</div>
            <div className="text-emerald-300 leading-snug mb-2"><Markdown content={item.answer} /></div>
            {item.explanation && <div className="text-sm text-zinc-400"><Markdown content={item.explanation} /></div>}
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
              className={`py-2.5 rounded-lg text-white text-sm font-medium transition-all active:scale-[0.98] ${g.tone}`}
            >
              {g.label}
            </button>
          ))}
        </div>
      )}
    </Card>
  )
}
