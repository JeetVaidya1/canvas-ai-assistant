import { useParams } from 'react-router-dom'
import { Calendar } from 'lucide-react'

export default function PlannerPage() {
  const { courseId: _courseId } = useParams<{ courseId: string }>()

  return (
    <div className="max-w-4xl mx-auto px-6 py-8 space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-zinc-100">Study Planner</h1>
        <p className="text-sm text-zinc-500 mt-1">AI-powered study schedules with spaced repetition</p>
      </div>

      <div className="bg-zinc-800/60 border border-zinc-700/40 rounded-xl p-8 text-center">
        <div className="w-14 h-14 rounded-full bg-zinc-800 flex items-center justify-center mx-auto mb-4">
          <Calendar className="w-7 h-7 text-zinc-500" />
        </div>
        <div className="inline-flex items-center gap-1.5 px-2.5 py-1 bg-cyan-500/10 text-cyan-400 text-xs font-medium rounded-full mb-3">
          Coming soon
        </div>
        <h3 className="text-lg font-semibold text-zinc-200 mb-2">Smart study planning is on the way</h3>
        <p className="text-sm text-zinc-500 max-w-md mx-auto">
          We're building personalized study calendars with spaced repetition scheduling to help you retain more and study efficiently.
        </p>
      </div>
    </div>
  )
}
