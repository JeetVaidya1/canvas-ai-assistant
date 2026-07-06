import { motion } from 'motion/react'
import {
  RefreshCw,
  Dumbbell,
  Clock,
  CheckCircle2,
  Circle,
  type LucideIcon,
} from 'lucide-react'
import type { StudyPlan } from '@/lib/api'
import { Card } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'

type DayType = 'review' | 'new' | 'practice'
type PlanDay = StudyPlan['days'][number]

const TYPE_META: Record<DayType, { label: string; badge: 'accent' | 'warning' | 'success'; dot: string; icon: LucideIcon }> = {
  new: { label: 'New', badge: 'accent', dot: 'bg-accent', icon: Circle },
  review: { label: 'Review', badge: 'warning', dot: 'bg-warning', icon: RefreshCw },
  practice: { label: 'Practice', badge: 'success', dot: 'bg-success', icon: Dumbbell },
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

/** Stable identity for a task within a plan (day + topic). */
function taskKey(date: string, topic: string): string {
  return `${date}|${topic}`
}

interface TimelineDayProps {
  day: PlanDay
  index: number
  total: number
  doneTasks: ReadonlySet<string>
  onToggleTask: (key: string) => void
}

/** A single day in the agenda timeline. Today is highlighted; past days dim. */
function TimelineDay({ day, index, total, doneTasks, onToggleTask }: TimelineDayProps) {
  const meta = TYPE_META[day.type] ?? TYPE_META.review
  const Icon = meta.icon
  const today = isToday(day.date)
  const past = isPast(day.date) && !today
  const { weekday, date } = formatDate(day.date)

  return (
    <motion.div
      initial={{ opacity: 0, x: -8 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.3, delay: Math.min(index * 0.04, 0.3), ease: [0.22, 1, 0.36, 1] }}
      className="relative flex gap-4 pl-1"
    >
      {/* Rail + node */}
      <div className="relative flex flex-col items-center flex-shrink-0 w-9">
        {index < total - 1 && (
          <span className="absolute top-9 bottom-[-1.25rem] w-px bg-line-strong" />
        )}
        <div
          className={`relative z-10 flex h-9 w-9 items-center justify-center rounded-full border ${
            today
              ? 'border-accent-line bg-accent-wash ring-2 ring-accent/20'
              : past
                ? 'border-line bg-paper-deep'
                : 'border-line bg-surface'
          }`}
        >
          {today ? (
            <Circle className="h-3.5 w-3.5 text-accent fill-accent" />
          ) : past ? (
            <CheckCircle2 className="h-4 w-4 text-ink-faint" />
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
          today ? 'ring-1 ring-accent/25 border-accent-line' : past ? 'opacity-70' : ''
        }`}
      >
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0 flex items-center gap-2">
            {today && (
              <span className="rounded-full bg-accent-wash border border-accent-line px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-accent-deep">
                Today
              </span>
            )}
            <span className={`font-mono text-[13px] font-semibold tracking-tight ${today ? 'text-accent-deep' : 'text-ink'}`}>
              {weekday} · {date}
            </span>
            <span className="font-mono text-[11px] text-ink-faint">Day {index + 1}</span>
          </div>
          <Badge tone={meta.badge} icon={<Icon />} className="flex-shrink-0">
            {meta.label}
          </Badge>
        </div>

        <div className="mt-2.5 flex items-center gap-1.5 text-xs text-ink-soft tnum">
          <Clock className="w-3.5 h-3.5 text-ink-faint" />
          {day.duration_minutes} min focused study
        </div>

        {/* Checkable tasks — ticked topics persist per course */}
        <ul className="mt-3 space-y-1.5">
          {day.topics.map((topic) => {
            const key = taskKey(day.date, topic)
            const done = doneTasks.has(key)
            return (
              <li key={key}>
                <button
                  type="button"
                  onClick={() => onToggleTask(key)}
                  aria-pressed={done}
                  className={`w-full flex items-center gap-2.5 rounded-lg border px-2.5 py-1.5 text-left transition-colors ${
                    done
                      ? 'border-success/25 bg-success-wash'
                      : 'border-line bg-surface hover:border-line-strong hover:bg-surface-hover'
                  }`}
                >
                  {done ? (
                    <CheckCircle2 className="h-3.5 w-3.5 text-success flex-shrink-0" />
                  ) : (
                    <Circle className="h-3.5 w-3.5 text-ink-faint flex-shrink-0" />
                  )}
                  <span className={`text-sm truncate ${done ? 'text-ink-faint line-through' : 'text-ink'}`}>
                    {topic}
                  </span>
                </button>
              </li>
            )
          })}
        </ul>
      </Card>
    </motion.div>
  )
}

interface PlanTimelineProps {
  plan: StudyPlan
  doneTasks: ReadonlySet<string>
  onToggleTask: (key: string) => void
}

/** Vertical agenda timeline with a day-type legend. */
export function PlanTimeline({ plan, doneTasks, onToggleTask }: PlanTimelineProps) {
  return (
    <div>
      <div className="flex flex-wrap items-center gap-3 mb-5 px-1">
        {(Object.keys(TYPE_META) as DayType[]).map((k) => (
          <span key={k} className="inline-flex items-center gap-1.5 text-[11px] text-ink-soft">
            <span className={`w-2 h-2 rounded-full ${TYPE_META[k].dot}`} />
            {TYPE_META[k].label}
          </span>
        ))}
      </div>
      <div>
        {plan.days.map((day, i) => (
          <TimelineDay
            key={day.date + i}
            day={day}
            index={i}
            total={plan.days.length}
            doneTasks={doneTasks}
            onToggleTask={onToggleTask}
          />
        ))}
      </div>
    </div>
  )
}
