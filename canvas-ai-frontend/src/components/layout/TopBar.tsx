import { useLocation, useNavigate } from 'react-router-dom'
import { ChevronRight, Settings } from 'lucide-react'
import { useCourses } from '@/hooks/useCourses'
import { useProfile } from '@/hooks/useProfile'

const COURSE_PAGE_LABELS: Record<string, string> = {
  learn: 'Learn',
  practice: 'Practice',
  exam: 'Exam',
  kit: 'Study Kit',
  progress: 'Progress',
}

export default function TopBar() {
  const location = useLocation()
  const navigate = useNavigate()
  const { data: courses } = useCourses()
  const { displayName } = useProfile()

  const segments = location.pathname.split('/').filter(Boolean)

  let pageTitle = 'Dashboard'
  let courseCrumb: { title: string; path: string } | null = null

  if (segments[0] === 'course' && segments[1]) {
    const courseId = segments[1]
    const course = courses?.find((c) => c.course_id === courseId)
    const courseTitle = course?.title ?? courseId
    if (segments[2]) {
      // Course sub-destination — show the course name as a secondary crumb.
      pageTitle = COURSE_PAGE_LABELS[segments[2]] ?? segments[2].charAt(0).toUpperCase() + segments[2].slice(1)
      courseCrumb = { title: courseTitle, path: `/course/${courseId}` }
    } else {
      pageTitle = courseTitle
    }
  } else if (segments[0] === 'settings') {
    pageTitle = 'Settings'
  }

  const initial = (displayName || 'U').charAt(0).toUpperCase()

  return (
    <div className="h-14 flex items-center justify-between px-6 top-bar">
      <div className="flex items-center gap-1.5 min-w-0">
        {courseCrumb && (
          <>
            <button
              onClick={() => navigate(courseCrumb!.path)}
              className="max-w-[220px] truncate text-sm text-ink-soft transition-colors hover:text-ink"
            >
              {courseCrumb.title}
            </button>
            <ChevronRight className="w-3.5 h-3.5 flex-shrink-0 text-ink-faint" />
          </>
        )}
        <span className="truncate text-sm font-semibold text-ink tracking-tight">{pageTitle}</span>
      </div>
      <div className="flex items-center gap-2.5">
        <button
          onClick={() => navigate('/settings')}
          className="p-2 text-ink-faint hover:text-accent hover:bg-line/40 rounded-lg transition-colors"
          aria-label="Settings"
        >
          <Settings className="w-4 h-4" />
        </button>
        <div className="w-8 h-8 rounded-full bg-ink flex items-center justify-center text-xs font-semibold text-paper">
          {initial}
        </div>
      </div>
    </div>
  )
}
