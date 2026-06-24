import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { motion } from 'motion/react'
import {
  Calendar,
  Download,
  Sparkles,
  BookOpen,
  RefreshCw,
  Dumbbell,
  Clock,
  CheckCircle2,
  Circle,
  ChevronDown,
} from 'lucide-react'
import { BrandMark } from '@/components/ui/BrandMark'
import {
  generateStudyPlan,
  getStudyPlan,
  replanStudyPlan,
  exportPlannerIcal,
  type StudyPlan,
} from '@/lib/api'
import { useUser } from '@/hooks/useUser'
import { showError, showSuccess } from '@/lib/toast'
import { Button } from '@/components/ui/Button'
import { Card, PageHeader } from '@/components/ui/Card'

const inputClass =
  'w-full px-3 py-2 bg-white/[0.04] border border-white/10 rounded-lg text-zinc-100 placeholder-zinc-500 ' +
  'focus:border-cyan-400/50 focus:ring-2 focus:ring-cyan-400/20 outline-none text-sm transition-colors'

type DayType = 'review' | 'new' | 'practice'

const TYPE_META: Record<DayType, { label: string; tone: string; dot: string; icon: typeof BookOpen }> = {
  new: { label: 'New', tone: 'text-cyan-300 bg-cyan-500/12 border-cyan-400/20', dot: 'bg-cyan-400', icon: Sparkles },
  review: { label: 'Review', tone: 'text-amber-300 bg-amber-500/10 border-amber-500/20', dot: 'bg-amber-400', icon: RefreshCw },
  practice: { label: 'Practice', tone: 'text-emerald-300 bg-emerald-500/10 border-emerald-500/20', dot: 'bg-emerald-400', icon: Dumbbell },
}

function formatDate(iso: string): { weekday: string; date: string } {
  try {
    const d = new Date(iso + 'T00:00:00')
    return {
      weekday: d.toLocaleDateString(undefined, { weekday: 'short' }),
      date: d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }),
    }
  } catch {
    return { weekday: '', date: iso }
  }
}

function isToday(iso: string): boolean {
  try {
    return new Date(iso + 'T00:00:00').toDateString() === new Date().toDateString()
  } catch {
    return false
  }
}

function isPast(iso: string): boolean {
  try {
    const d = new Date(iso + 'T00:00:00')
    const today = new Date()
    today.setHours(0, 0, 0, 0)
    return d < today
  } catch {
    return false
  }
}

/** A single day in the agenda timeline. Today is highlighted; past days dim. */
function TimelineDay({
  day,
  index,
  total,
}: {
  day: StudyPlan['days'][number]
  index: number
  total: number
}) {
  const meta = TYPE_META[(day.type as DayType)] ?? TYPE_META.review
  const Icon = meta.icon
  const today = isToday(day.date)
  const past = isPast(day.date) && !today
  const { weekday, date } = formatDate(day.date)

  return (
    <motion.div
      initial={{ opacity: 0, x: -8 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.3, delay: Math.min(index * 0.04, 0.4), ease: [0.22, 1, 0.36, 1] }}
      className="relative flex gap-4 pl-1"
    >
      {/* Rail + node */}
      <div className="relative flex flex-col items-center flex-shrink-0 w-9">
        {/* connector line (skip after last) */}
        {index < total - 1 && (
          <span className="absolute top-9 bottom-[-1.25rem] w-px bg-white/[0.08]" />
        )}
        <div
          className={`relative z-10 flex h-9 w-9 items-center justify-center rounded-full border ${
            today
              ? 'border-cyan-400/50 bg-cyan-500/15 ring-2 ring-cyan-400/25'
              : past
                ? 'border-white/10 bg-white/[0.03]'
                : 'border-white/10 bg-[#19202f]'
          }`}
        >
          {today ? (
            <Circle className="h-3.5 w-3.5 text-cyan-300 fill-cyan-300" />
          ) : past ? (
            <CheckCircle2 className="h-4 w-4 text-zinc-500" />
          ) : (
            <span className={`h-2 w-2 rounded-full ${meta.dot}`} />
          )}
        </div>
      </div>

      {/* Day card */}
      <Card
        padding="md"
        accent={today}
        className={`mb-5 flex-1 transition-colors ${
          today ? 'ring-1 ring-cyan-400/30 border-cyan-400/30' : past ? 'opacity-70' : ''
        }`}
      >
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              {today && (
                <span className="rounded-full bg-cyan-500/15 border border-cyan-400/30 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-cyan-200">
                  Today
                </span>
              )}
              <span className={`text-sm font-semibold ${today ? 'text-cyan-200' : 'text-zinc-100'}`}>
                {weekday} · {date}
              </span>
              <span className="text-[11px] text-zinc-500">Day {index + 1}</span>
            </div>
          </div>
          <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full border text-[11px] font-medium flex-shrink-0 ${meta.tone}`}>
            <Icon className="w-3 h-3" />
            {meta.label}
          </span>
        </div>

        <div className="mt-2.5 flex items-center gap-1.5 text-xs text-zinc-400">
          <Clock className="w-3.5 h-3.5 text-zinc-500" />
          {day.duration_minutes} min focused study
        </div>

        {/* Checkable task affordance per topic */}
        <ul className="mt-3 space-y-1.5">
          {day.topics.map((t, j) => (
            <li
              key={j}
              className="flex items-center gap-2.5 rounded-lg border border-white/[0.06] bg-white/[0.02] px-2.5 py-1.5"
            >
              <Circle className="h-3.5 w-3.5 text-zinc-600 flex-shrink-0" />
              <span className="text-sm text-zinc-200 truncate">{t}</span>
            </li>
          ))}
        </ul>
      </Card>
    </motion.div>
  )
}

export default function PlannerPage() {
  const { courseId } = useParams<{ courseId: string }>()
  const userId = useUser()

  const [plan, setPlan] = useState<StudyPlan | null>(null)
  const [loading, setLoading] = useState(false)
  const [replanning, setReplanning] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [daysAvailable, setDaysAvailable] = useState(10)
  const [hoursPerDay, setHoursPerDay] = useState(2)
  const [examDate, setExamDate] = useState('')
  const [setupOpen, setSetupOpen] = useState(false)

  useEffect(() => {
    if (!courseId) return
    void getStudyPlan(courseId).then((p) => {
      if (p) setPlan(p)
    })
    // Prefill the exam date if a Canvas import detected one.
    const detected = localStorage.getItem(`vindexa_exam_date_${courseId}`)
    if (detected) setExamDate(detected)
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
      setSetupOpen(false)
      showSuccess('Study plan generated')
    } catch (e) {
      showError(e instanceof Error ? e.message : 'Failed to generate study plan')
    } finally {
      setLoading(false)
    }
  }

  const handleReplan = async () => {
    if (!courseId) return
    setReplanning(true)
    try {
      const result = await replanStudyPlan(courseId, userId, {
        daysAvailable,
        hoursPerDay,
        examDate: examDate || undefined,
      })
      setPlan(result)
      showSuccess('Replanned around your weak areas')
    } catch (e) {
      showError(e instanceof Error ? e.message : 'Failed to replan')
    } finally {
      setReplanning(false)
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

  const totalMinutes = plan ? plan.days.reduce((s, d) => s + d.duration_minutes, 0) : 0
  const totalHours = Math.round((totalMinutes / 60) * 10) / 10

  // The setup panel is collapsible once a plan exists, so the timeline leads.
  const showSetup = !plan || setupOpen

  return (
    <div className="max-w-4xl mx-auto px-6 py-8 space-y-6">
      <PageHeader
        eyebrow="Planner"
        title="Study Planner"
        subtitle="AI-powered study schedules with spaced repetition"
        actions={
          plan ? (
            <>
              <Button
                variant="secondary"
                onClick={() => void handleReplan()}
                loading={replanning}
                leftIcon={<Sparkles className="w-4 h-4" />}
                title="Rebuild the plan around your current weak areas and due reviews"
              >
                {replanning ? 'Replanning...' : 'Focus on weak areas'}
              </Button>
              <Button
                variant="secondary"
                onClick={() => void handleExport()}
                loading={exporting}
                leftIcon={<Download className="w-4 h-4" />}
              >
                {exporting ? 'Exporting...' : 'Export to iCal'}
              </Button>
            </>
          ) : undefined
        }
      />

      {/* ── Plan summary + collapsible setup ─────────────────────────── */}
      {plan && (
        <Card accent padding="md" className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-5">
            <div className="flex items-center gap-2.5">
              <div className="w-10 h-10 rounded-xl bg-cyan-500/12 border border-cyan-400/20 flex items-center justify-center">
                <Calendar className="w-5 h-5 text-cyan-300" />
              </div>
              <div>
                <p className="text-lg font-semibold text-zinc-50 leading-none tabular-nums">{plan.days.length}</p>
                <p className="text-xs text-zinc-400 mt-1">study days</p>
              </div>
            </div>
            <div className="h-9 w-px bg-white/[0.08]" />
            <div>
              <p className="text-lg font-semibold text-zinc-50 leading-none tabular-nums">{totalHours}h</p>
              <p className="text-xs text-zinc-400 mt-1">total planned</p>
            </div>
          </div>
          <Button
            variant="ghost"
            onClick={() => setSetupOpen((v) => !v)}
            rightIcon={<ChevronDown className={`w-4 h-4 transition-transform ${setupOpen ? 'rotate-180' : ''}`} />}
          >
            Adjust plan
          </Button>
        </Card>
      )}

      {/* ── Setup form ───────────────────────────────────────────────── */}
      {showSetup && (
        <Card accent padding="md">
          <h2 className="text-sm font-semibold text-zinc-100 mb-4">Plan your revision</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
            <div>
              <label className="block text-xs font-medium text-zinc-400 mb-1.5">Days available</label>
              <input
                type="number"
                min={1}
                max={60}
                value={daysAvailable}
                onChange={(e) => setDaysAvailable(Math.max(1, Math.min(60, Number(e.target.value))))}
                className={inputClass}
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-zinc-400 mb-1.5">Hours per day</label>
              <input
                type="number"
                min={0.5}
                max={12}
                step={0.5}
                value={hoursPerDay}
                onChange={(e) => setHoursPerDay(Math.max(0.5, Math.min(12, Number(e.target.value))))}
                className={inputClass}
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-zinc-400 mb-1.5">Exam date (optional)</label>
              <input
                type="date"
                value={examDate}
                onChange={(e) => setExamDate(e.target.value)}
                className={inputClass}
              />
            </div>
          </div>
          <Button
            onClick={() => void handleGenerate()}
            loading={loading}
            disabled={!courseId}
            leftIcon={<Sparkles className="w-4 h-4" />}
          >
            {loading ? 'Generating...' : plan ? 'Regenerate Plan' : 'Generate Plan'}
          </Button>
        </Card>
      )}

      {/* ── Agenda timeline ──────────────────────────────────────────── */}
      {plan ? (
        <div>
          {/* type legend */}
          <div className="flex flex-wrap items-center gap-3 mb-5 px-1">
            {(Object.keys(TYPE_META) as DayType[]).map((k) => (
              <span key={k} className="inline-flex items-center gap-1.5 text-[11px] text-zinc-400">
                <span className={`w-2 h-2 rounded-full ${TYPE_META[k].dot}`} />
                {TYPE_META[k].label}
              </span>
            ))}
          </div>
          <div>
            {plan.days.map((day, i) => (
              <TimelineDay key={i} day={day} index={i} total={plan.days.length} />
            ))}
          </div>
        </div>
      ) : (
        <Card accent padding="none" elevation={2} className="py-12 px-8 text-center">
          <BrandMark className="mx-auto mb-4 h-14 w-14" />
          <h3 className="text-lg font-semibold text-zinc-100 mb-2">No plan yet</h3>
          <p className="text-sm text-zinc-400 max-w-md mx-auto">
            Set your available days and hours above, then generate a personalized study schedule
            with spaced-repetition reviews.
          </p>
        </Card>
      )}
    </div>
  )
}
