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
    <div className="h-12 flex items-center justify-between px-5 border-b border-zinc-800 bg-zinc-950/80 backdrop-blur-xl">
      <span className="text-sm font-medium text-zinc-100">{pageTitle}</span>
      <div className="flex items-center gap-2">
        <button
          onClick={() => navigate('/settings')}
          className="p-1.5 text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800 rounded-lg transition-colors"
        >
          <Settings className="w-4 h-4" />
        </button>
        <div className="w-7 h-7 rounded-full bg-cyan-600 flex items-center justify-center text-xs font-medium text-white">
          {initial}
        </div>
      </div>
    </div>
  )
}
