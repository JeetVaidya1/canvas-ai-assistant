import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Calendar, Download, Sparkles, ChevronDown } from 'lucide-react'
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
import { Card } from '@/components/ui/Card'
import { Input } from '@/components/ui/Input'
import { EmptyState, ErrorState } from '@/components/ui/States'
import { PlanTimeline } from '@/components/progress/PlanTimeline'

const MIN_DAYS = 1
const MAX_DAYS = 60
const MIN_HOURS = 0.5
const MAX_HOURS = 12

/** Clamp numeric input safely (empty/garbage input falls back to the min). */
function clamp(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min
  return Math.max(min, Math.min(max, value))
}

function doneTasksStorageKey(courseId: string): string {
  return `vindexa_plan_done_${courseId}`
}

function loadDoneTasks(courseId: string | undefined): ReadonlySet<string> {
  if (!courseId) return new Set()
  try {
    const raw = localStorage.getItem(doneTasksStorageKey(courseId))
    const parsed: unknown = raw ? JSON.parse(raw) : []
    return new Set(Array.isArray(parsed) ? parsed.filter((k): k is string => typeof k === 'string') : [])
  } catch {
    return new Set()
  }
}

function persistDoneTasks(courseId: string, tasks: ReadonlySet<string>): void {
  try {
    localStorage.setItem(doneTasksStorageKey(courseId), JSON.stringify([...tasks]))
  } catch {
    // Storage full/blocked — ticks just won't persist; never break the UI.
  }
}

function PlannerSkeleton() {
  return (
    <div className="max-w-4xl mx-auto px-6 py-8 space-y-6 animate-pulse" aria-hidden>
      <div className="h-20 rounded-xl bg-paper-deep border border-line" />
      {[0, 1, 2].map((i) => (
        <div key={i} className="h-32 rounded-xl bg-paper-deep border border-line" />
      ))}
    </div>
  )
}

export default function PlannerPage() {
  const { courseId } = useParams<{ courseId: string }>()
  const userId = useUser()
  const queryClient = useQueryClient()

  const [daysAvailable, setDaysAvailable] = useState(10)
  const [hoursPerDay, setHoursPerDay] = useState(2)
  const [examDate, setExamDate] = useState('')
  const [setupOpen, setSetupOpen] = useState(false)
  const [doneTasks, setDoneTasks] = useState<ReadonlySet<string>>(() => loadDoneTasks(courseId))

  // Re-sync per-course state: detected exam date (Canvas import) + task ticks.
  useEffect(() => {
    if (!courseId) return
    const detected = localStorage.getItem(`vindexa_exam_date_${courseId}`)
    if (detected) setExamDate(detected)
    setDoneTasks(loadDoneTasks(courseId))
  }, [courseId])

  const planQuery = useQuery({
    queryKey: ['studyPlan', courseId],
    queryFn: () => getStudyPlan(courseId ?? ''),
    enabled: !!courseId,
  })
  const plan = planQuery.data ?? null

  const planParams = { daysAvailable, hoursPerDay, examDate: examDate || undefined }

  const generateMutation = useMutation({
    mutationFn: () => generateStudyPlan(courseId ?? '', planParams),
    onSuccess: (result: StudyPlan) => {
      queryClient.setQueryData(['studyPlan', courseId], result)
      setSetupOpen(false)
      showSuccess('Study plan generated')
    },
    onError: (e: unknown) => showError(e instanceof Error ? e.message : 'Failed to generate study plan'),
  })

  const replanMutation = useMutation({
    mutationFn: () => replanStudyPlan(courseId ?? '', userId, planParams),
    onSuccess: (result: StudyPlan) => {
      queryClient.setQueryData(['studyPlan', courseId], result)
      showSuccess('Replanned around your weak areas')
    },
    onError: (e: unknown) => showError(e instanceof Error ? e.message : 'Failed to replan'),
  })

  const exportMutation = useMutation({
    mutationFn: () => exportPlannerIcal(courseId ?? ''),
    onSuccess: (blob: Blob) => {
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${courseId}_study_plan.ics`
      a.click()
      URL.revokeObjectURL(url)
    },
    onError: (e: unknown) => showError(e instanceof Error ? e.message : 'Failed to export calendar'),
  })

  const toggleTask = (key: string) => {
    if (!courseId) return
    setDoneTasks((prev) => {
      const next = new Set(prev)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      persistDoneTasks(courseId, next)
      return next
    })
  }

  if (planQuery.isPending && !!courseId) return <PlannerSkeleton />

  if (planQuery.isError) {
    return (
      <div className="max-w-4xl mx-auto px-6 py-8">
        <ErrorState
          title="Couldn't load your study plan"
          description="Check your connection and try again."
          onRetry={() => void planQuery.refetch()}
          retrying={planQuery.isRefetching}
        />
      </div>
    )
  }

  const totalMinutes = plan ? plan.days.reduce((s, d) => s + d.duration_minutes, 0) : 0
  const totalHours = Math.round((totalMinutes / 60) * 10) / 10

  // The setup panel is collapsible once a plan exists, so the timeline leads.
  const showSetup = !plan || setupOpen

  return (
    <div className="max-w-4xl mx-auto px-6 py-8 space-y-6">
      {/* Slim toolbar — the Progress wrapper bar already names the page. */}
      {plan && (
        <div className="flex items-center justify-end gap-2 -mb-2">
          <Button
            variant="secondary"
            size="sm"
            onClick={() => replanMutation.mutate()}
            loading={replanMutation.isPending}
            leftIcon={<Sparkles className="w-4 h-4" />}
            title="Rebuild the plan around your current weak areas and due reviews"
          >
            {replanMutation.isPending ? 'Replanning...' : 'Focus on weak areas'}
          </Button>
          <Button
            variant="secondary"
            size="sm"
            onClick={() => exportMutation.mutate()}
            loading={exportMutation.isPending}
            leftIcon={<Download className="w-4 h-4" />}
          >
            {exportMutation.isPending ? 'Exporting...' : 'Export to iCal'}
          </Button>
        </div>
      )}

      {/* Plan summary + collapsible setup */}
      {plan && (
        <Card accent padding="md" className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-5">
            <div className="flex items-center gap-2.5">
              <div className="w-10 h-10 rounded-xl bg-accent-wash border border-accent-line flex items-center justify-center">
                <Calendar className="w-5 h-5 text-accent" />
              </div>
              <div>
                <p className="font-display text-xl font-semibold text-ink leading-none tnum">{plan.days.length}</p>
                <p className="text-xs text-ink-soft mt-1">study days</p>
              </div>
            </div>
            <div className="h-9 w-px bg-line" />
            <div>
              <p className="font-display text-xl font-semibold text-ink leading-none tnum">{totalHours}h</p>
              <p className="text-xs text-ink-soft mt-1">total planned</p>
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

      {/* Setup form */}
      {showSetup && (
        <Card accent padding="md">
          <h2 className="text-sm font-semibold text-ink mb-4">Plan your revision</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
            <Input
              label="Days available"
              type="number"
              min={MIN_DAYS}
              max={MAX_DAYS}
              value={daysAvailable}
              onChange={(e) => setDaysAvailable(clamp(Number(e.target.value), MIN_DAYS, MAX_DAYS))}
            />
            <Input
              label="Hours per day"
              type="number"
              min={MIN_HOURS}
              max={MAX_HOURS}
              step={0.5}
              value={hoursPerDay}
              onChange={(e) => setHoursPerDay(clamp(Number(e.target.value), MIN_HOURS, MAX_HOURS))}
            />
            <Input
              label="Exam date (optional)"
              type="date"
              value={examDate}
              onChange={(e) => setExamDate(e.target.value)}
            />
          </div>
          <Button
            onClick={() => generateMutation.mutate()}
            loading={generateMutation.isPending}
            disabled={!courseId}
            leftIcon={<Sparkles className="w-4 h-4" />}
          >
            {generateMutation.isPending ? 'Generating...' : plan ? 'Regenerate Plan' : 'Generate Plan'}
          </Button>
        </Card>
      )}

      {/* Agenda timeline */}
      {plan ? (
        <PlanTimeline plan={plan} doneTasks={doneTasks} onToggleTask={toggleTask} />
      ) : (
        <Card accent padding="lg" elevation={2}>
          <EmptyState
            icon={<Calendar />}
            title="Generate a study plan from your exam date"
            description="Set your available days and hours above — we’ll schedule new material, spaced-repetition reviews and practice blocks up to exam day."
          />
        </Card>
      )}
    </div>
  )
}
