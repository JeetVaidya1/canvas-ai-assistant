import { useLocation, useNavigate } from 'react-router-dom'
import { Settings } from 'lucide-react'
import { useCourses } from '@/hooks/useCourses'
import { useProfile } from '@/hooks/useProfile'

export default function TopBar() {
  const location = useLocation()
  const navigate = useNavigate()
  const { data: courses } = useCourses()
  const { displayName } = useProfile()

  const segments = location.pathname.split('/').filter(Boolean)

  let pageTitle = 'Dashboard'
  if (segments[0] === 'course' && segments[2]) {
    pageTitle = segments[2].charAt(0).toUpperCase() + segments[2].slice(1)
  } else if (segments[0] === 'course' && segments[1]) {
    const courseId = segments[1]
    const course = courses?.find((c) => c.course_id === courseId)
    pageTitle = course?.title ?? courseId
  } else if (segments[0] === 'settings') {
    pageTitle = 'Settings'
  }

  const initial = (displayName || 'U').charAt(0).toUpperCase()

  return (
    <div className="h-14 flex items-center justify-between px-6 glass-bar sticky top-0 z-20">
      <span className="text-sm font-semibold text-zinc-100 tracking-tight">{pageTitle}</span>
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
