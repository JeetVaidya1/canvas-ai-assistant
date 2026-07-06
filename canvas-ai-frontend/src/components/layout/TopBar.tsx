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
    <div className="h-14 flex items-center justify-between px-6 glass-bar">
      <div className="flex items-center gap-1.5 min-w-0">
        {courseCrumb && (
          <>
            <button
              onClick={() => navigate(courseCrumb!.path)}
              className="max-w-[220px] truncate text-sm text-zinc-400 transition-colors hover:text-zinc-100"
            >
              {courseCrumb.title}
            </button>
            <ChevronRight className="w-3.5 h-3.5 flex-shrink-0 text-zinc-500" />
          </>
        )}
        <span className="truncate text-sm font-semibold text-zinc-100 tracking-tight">{pageTitle}</span>
      </div>
      <div className="flex items-center gap-2.5">
        <button
          onClick={() => navigate('/settings')}
          className="p-2 text-zinc-500 hover:text-cyan-300 hover:bg-white/[0.05] rounded-lg transition-colors"
          aria-label="Settings"
        >
          <Settings className="w-4 h-4" />
        </button>
        <div className="w-8 h-8 rounded-full bg-gradient-brand flex items-center justify-center text-xs font-semibold text-white glow-brand-sm ring-1 ring-white/10">
          {initial}
        </div>
      </div>
    </div>
  )
}
