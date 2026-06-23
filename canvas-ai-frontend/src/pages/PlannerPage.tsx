import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { Calendar, Download, Sparkles, BookOpen, RefreshCw, Dumbbell } from 'lucide-react'
import {
  generateStudyPlan,
  getStudyPlan,
  exportPlannerIcal,
  type StudyPlan,
} from '@/lib/api'
import { showError, showSuccess } from '@/lib/toast'

type DayType = 'review' | 'new' | 'practice'

const TYPE_META: Record<DayType, { label: string; tone: string; icon: typeof BookOpen }> = {
  new: { label: 'New', tone: 'text-cyan-400 bg-cyan-500/10 border-cyan-500/20', icon: Sparkles },
  review: { label: 'Review', tone: 'text-amber-400 bg-amber-500/10 border-amber-500/20', icon: RefreshCw },
  practice: { label: 'Practice', tone: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20', icon: Dumbbell },
}

function formatDate(iso: string): string {
  try {
    return new Date(iso + 'T00:00:00').toLocaleDateString(undefined, {
      weekday: 'short',
      month: 'short',
      day: 'numeric',
    })
  } catch {
    return iso
  }
}

export default function PlannerPage() {
  const { courseId } = useParams<{ courseId: string }>()

  const [plan, setPlan] = useState<StudyPlan | null>(null)
  const [loading, setLoading] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [daysAvailable, setDaysAvailable] = useState(10)
  const [hoursPerDay, setHoursPerDay] = useState(2)
  const [examDate, setExamDate] = useState('')

  useEffect(() => {
    if (!courseId) return
    void getStudyPlan(courseId).then((p) => {
      if (p) setPlan(p)
    })
  }, [courseId])

  const handleGenerate = async () => {
    if (!courseId) return
    setLoading(true)
    try {
      const result = await generateStudyPlan(courseId, {
        daysAvailable,
        hoursPerDay,
        examDate: examDate || undefined,
      })
      setPlan(result)
      showSuccess('Study plan generated')
    } catch (e) {
      showError(e instanceof Error ? e.message : 'Failed to generate study plan')
    } finally {
      setLoading(false)
    }
  }

  const handleExport = async () => {
    if (!courseId) return
    setExporting(true)
    try {
      const blob = await exportPlannerIcal(courseId)
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${courseId}_study_plan.ics`
      a.click()
      URL.revokeObjectURL(url)
    } catch (e) {
      showError(e instanceof Error ? e.message : 'Failed to export calendar')
    } finally {
      setExporting(false)
    }
  }

  return (
    <div className="max-w-4xl mx-auto px-6 py-8 space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-zinc-100">Study Planner</h1>
          <p className="text-sm text-zinc-500 mt-1">AI-powered study schedules with spaced repetition</p>
        </div>
        {plan && (
          <button
            onClick={() => void handleExport()}
            disabled={exporting}
            className="bg-zinc-800 border border-zinc-700 text-zinc-200 px-4 py-2 rounded-lg hover:bg-zinc-700 disabled:opacity-50 text-sm font-medium flex items-center gap-2 transition-colors flex-shrink-0"
          >
            <Download className="w-4 h-4" />
            {exporting ? 'Exporting...' : 'Export to iCal'}
          </button>
        )}
      </div>

      {/* Configuration form */}
      <div className="bg-zinc-800/60 border border-zinc-700/40 rounded-xl p-5">
        <h2 className="text-sm font-medium text-zinc-200 mb-4">Plan your revision</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
          <div>
            <label className="block text-xs font-medium text-zinc-500 mb-1.5">Days available</label>
            <input
              type="number"
              min={1}
              max={60}
              value={daysAvailable}
              onChange={(e) => setDaysAvailable(Math.max(1, Math.min(60, Number(e.target.value))))}
              className="w-full px-3 py-2 border border-zinc-700 rounded-lg bg-zinc-900 text-zinc-100 text-sm"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-zinc-500 mb-1.5">Hours per day</label>
            <input
              type="number"
              min={0.5}
              max={12}
              step={0.5}
              value={hoursPerDay}
              onChange={(e) => setHoursPerDay(Math.max(0.5, Math.min(12, Number(e.target.value))))}
              className="w-full px-3 py-2 border border-zinc-700 rounded-lg bg-zinc-900 text-zinc-100 text-sm"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-zinc-500 mb-1.5">Exam date (optional)</label>
            <input
              type="date"
              value={examDate}
              onChange={(e) => setExamDate(e.target.value)}
              className="w-full px-3 py-2 border border-zinc-700 rounded-lg bg-zinc-900 text-zinc-100 text-sm"
            />
          </div>
        </div>
        <button
          onClick={() => void handleGenerate()}
          disabled={loading || !courseId}
          className="bg-cyan-600 text-white px-4 py-2 rounded-lg hover:bg-cyan-500 disabled:opacity-50 disabled:cursor-not-allowed text-sm font-medium flex items-center gap-2 transition-colors"
        >
          {loading ? (
            <>
              <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              Generating...
            </>
          ) : (
            <>
              <Sparkles className="w-4 h-4" />
              {plan ? 'Regenerate Plan' : 'Generate Plan'}
            </>
          )}
        </button>
      </div>

      {/* Generated plan — day-by-day list */}
      {plan ? (
        <div className="space-y-2">
          {plan.days.map((day, i) => {
            const meta = TYPE_META[(day.type as DayType)] ?? TYPE_META.review
            const Icon = meta.icon
            return (
              <div
                key={i}
                className="bg-zinc-800/60 border border-zinc-700/40 rounded-xl p-4 flex items-start gap-4"
              >
                <div className="flex flex-col items-center justify-center w-16 flex-shrink-0">
                  <span className="text-xs text-zinc-500">Day {i + 1}</span>
                  <span className="text-sm font-medium text-zinc-200 text-center">{formatDate(day.date)}</span>
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1.5">
                    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full border text-[11px] font-medium ${meta.tone}`}>
                      <Icon className="w-3 h-3" />
                      {meta.label}
                    </span>
                    <span className="text-xs text-zinc-500">{day.duration_minutes} min</span>
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {day.topics.map((t, j) => (
                      <span key={j} className="text-xs text-zinc-300 bg-zinc-800 border border-zinc-700 rounded px-2 py-0.5">
                        {t}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      ) : (
        <div className="bg-zinc-800/60 border border-zinc-700/40 rounded-xl p-8 text-center">
          <div className="w-14 h-14 rounded-full bg-zinc-800 flex items-center justify-center mx-auto mb-4">
            <Calendar className="w-7 h-7 text-zinc-500" />
          </div>
          <h3 className="text-lg font-semibold text-zinc-200 mb-2">No plan yet</h3>
          <p className="text-sm text-zinc-500 max-w-md mx-auto">
            Set your available days and hours above, then generate a personalized study schedule
            with spaced-repetition reviews.
          </p>
        </div>
      )}
    </div>
  )
}
