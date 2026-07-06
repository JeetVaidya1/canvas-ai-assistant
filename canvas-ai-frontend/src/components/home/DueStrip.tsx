import { useNavigate } from 'react-router-dom'
import { useQueries } from '@tanstack/react-query'
import { CalendarClock } from 'lucide-react'
import { Badge } from '@/components/ui/Badge'
import { reviewQueueOptions } from '@/hooks/useReviews'
import { useUser } from '@/hooks/useUser'

interface DueStripProps {
  courses: ReadonlyArray<{ course_id: string; title: string }>
}

/**
 * Slim cross-course "Due today" chips for the Dashboard.
 *
 * Design note: this is an enhancement layer, not a primary surface — it
 * appears only once review queues resolve with due items. While loading, or
 * when a queue errors, it renders nothing; the canonical loading/error/empty
 * treatments for reviews live on CourseHome and Progress. Queue queries are
 * shared with (and cached for) those pages via reviewQueueOptions.
 */
export default function DueStrip({ courses }: DueStripProps) {
  const navigate = useNavigate()
  const userId = useUser()

  const results = useQueries({
    queries: courses.map((c) => ({
      ...reviewQueueOptions(c.course_id, userId),
      enabled: !!userId,
    })),
  })

  const due = courses
    .map((course, i) => ({ course, dueCount: results[i]?.data?.due_count ?? 0 }))
    .filter((entry) => entry.dueCount > 0)

  if (due.length === 0) return null

  return (
    <div className="flex flex-wrap items-center gap-2.5 animate-fade-up">
      <span className="inline-flex items-center gap-1.5 text-xs font-medium text-ink-soft">
        <CalendarClock className="w-3.5 h-3.5 text-ink-faint" />
        Due today
      </span>
      {due.map(({ course, dueCount }) => (
        <button
          key={course.course_id}
          type="button"
          onClick={() => navigate(`/course/${course.course_id}/progress`)}
          className="focus-ring rounded-md"
          aria-label={`Review ${dueCount} due in ${course.title}`}
        >
          <Badge tone="accent" className="cursor-pointer max-w-[18rem] hover:border-accent transition-colors">
            <span className="truncate">{course.title}</span>
            <span className="tnum flex-shrink-0">{dueCount} due</span>
          </Badge>
        </button>
      ))}
    </div>
  )
}
