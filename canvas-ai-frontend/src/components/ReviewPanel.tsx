import { useEffect, useState } from 'react'
import { Brain, CheckCircle, Zap } from 'lucide-react'
import { Markdown } from '@/components/ui/Markdown'
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
      <div className="bg-gradient-to-r from-amber-500/10 to-zinc-800/40 border border-amber-500/20 rounded-xl p-5 flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-amber-500/15 flex items-center justify-center flex-shrink-0">
            <Zap className="w-5 h-5 text-amber-400" />
          </div>
          <div>
            <h2 className="text-sm font-semibold text-zinc-100">
              {dueCount} review{dueCount === 1 ? '' : 's'} due
            </h2>
            <p className="text-xs text-zinc-400">Questions you missed, resurfaced on schedule. Clear them to raise your readiness.</p>
          </div>
        </div>
        <button
          onClick={start}
          className="bg-amber-600 text-white px-4 py-2 rounded-lg hover:bg-amber-500 text-sm font-medium flex items-center gap-2 transition-colors flex-shrink-0"
        >
          <Brain className="w-4 h-4" /> Review now
        </button>
      </div>
    )
  }

  // Finished
  if (index >= items.length) {
    return (
      <div className="bg-zinc-800/60 border border-zinc-700/40 rounded-xl p-6 text-center">
        <CheckCircle className="w-12 h-12 text-emerald-400 mx-auto mb-3" />
        <p className="text-emerald-400 font-medium">Review cleared</p>
        <p className="text-zinc-500 text-sm mb-4">You worked through {items.length} item{items.length === 1 ? '' : 's'}.</p>
        <button
          onClick={() => { setActive(false); void load() }}
          className="px-4 py-2 border border-zinc-700 rounded-lg text-sm text-zinc-300 hover:bg-zinc-800"
        >
          Done
        </button>
      </div>
    )
  }

  const item = items[index]
  return (
    <div className="bg-zinc-800/60 border border-zinc-700/40 rounded-xl p-6">
      <div className="flex items-center justify-between mb-3 text-xs text-zinc-500">
        <span>Review {index + 1} of {items.length}</span>
        <span className="text-amber-400">{item.concept}</span>
      </div>
      <div className="rounded-xl border border-zinc-700 bg-zinc-900 p-5">
        <div className="text-xs text-zinc-400 mb-1">From a {item.source} you missed</div>
        <div className="text-zinc-50 font-medium leading-snug mb-3"><Markdown content={item.prompt} /></div>
        {revealed && (
          <>
            <div className="border-t border-zinc-800 my-3" />
            <div className="text-xs text-zinc-400 mb-1">Answer</div>
            <div className="text-emerald-300 leading-snug mb-2"><Markdown content={item.answer} /></div>
            {item.explanation && <div className="text-sm text-zinc-400"><Markdown content={item.explanation} /></div>}
          </>
        )}
      </div>

      {!revealed ? (
        <button
          onClick={() => setRevealed(true)}
          className="mt-4 w-full bg-zinc-800 border border-zinc-700 text-zinc-200 py-2.5 rounded-lg hover:bg-zinc-700 text-sm font-medium"
        >
          Show answer
        </button>
      ) : (
        <div className="mt-4 grid grid-cols-4 gap-2">
          {GRADES.map((g) => (
            <button
              key={g.grade}
              onClick={() => void grade(g.grade)}
              className={`py-2.5 rounded-lg text-white text-sm font-medium transition-colors ${g.tone}`}
            >
              {g.label}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
