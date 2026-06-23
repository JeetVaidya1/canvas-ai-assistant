import { useState } from 'react'
import { NavLink, useParams, useLocation } from 'react-router-dom'
import {
  BookOpen,
  Settings,
  ChevronLeft,
  ChevronRight,
  GraduationCap,
  MessageCircle,
  Brain,
  Target,
  FileText,
  ClipboardList,
  BarChart3,
  Calendar,
  Headphones,
  Lightbulb,
} from 'lucide-react'
import { useCourses } from '@/hooks/useCourses'

interface SubNavItem {
  label: string
  path: string
  icon: typeof BookOpen
  group?: string
}

const courseSubNav: SubNavItem[] = [
  { label: 'Overview', path: '', icon: BookOpen },
  { label: 'Chat', path: '/chat', icon: MessageCircle, group: 'Study' },
  { label: 'Tutor', path: '/tutor', icon: Lightbulb },
  { label: 'Quiz', path: '/quiz', icon: Brain },
  { label: 'Practice', path: '/practice', icon: Target },
  { label: 'Notes', path: '/notes', icon: FileText, group: 'Create' },
  { label: 'Exams', path: '/exams', icon: ClipboardList },
  { label: 'Analytics', path: '/analytics', icon: BarChart3, group: 'Insights' },
  { label: 'Planner', path: '/planner', icon: Calendar },
  { label: 'Audio', path: '/audio', icon: Headphones },
]

export default function AppSidebar() {
  const [collapsed, setCollapsed] = useState(() => {
    return localStorage.getItem('vindexa_sidebar_collapsed') === 'true'
  })
  const { courseId } = useParams()
  const location = useLocation()
  const { data: courses } = useCourses()

  return (
    <aside
      className={`flex flex-col bg-zinc-950/80 backdrop-blur-xl border-r border-zinc-800 transition-all duration-300 ${
        collapsed ? 'w-16' : 'w-64'
      }`}
    >
      {/* Logo */}
      <div className="h-12 flex items-center px-4 border-b border-zinc-800 gap-3">
        <img src="/favicon-32x32.png" alt="Vindexa" className="w-8 h-8 rounded-lg flex-shrink-0" />
        {!collapsed && (
          <span className="text-lg font-semibold text-white">Vindexa</span>
        )}
      </div>

      {/* Course list */}
      <div className="flex-1 overflow-y-auto py-3">
        {!collapsed && (
          <div className="px-4 mb-2 text-xs font-semibold text-zinc-500 uppercase tracking-wider">
            Courses
          </div>
        )}
        <nav className="space-y-1 px-2">
          {courses?.map((course) => {
            const isActiveCourse = courseId === course.course_id
            return (
              <div key={course.course_id}>
                <NavLink
                  to={`/course/${course.course_id}`}
                  end
                  className={({ isActive }) =>
                    `flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors ${
                      isActive || isActiveCourse
                        ? 'bg-zinc-800 text-cyan-400'
                        : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50'
                    }`
                  }
                >
                  <GraduationCap className="w-4 h-4 flex-shrink-0" />
                  {!collapsed && (
                    <span className="truncate">{course.title}</span>
                  )}
                </NavLink>

                {/* Course sub-navigation with group labels */}
                {isActiveCourse && !collapsed && (
                  <div className="ml-4 mt-1 space-y-0.5">
                    {courseSubNav.map((item) => {
                      const fullPath = `/course/${course.course_id}${item.path}`
                      const isItemActive = item.path === ''
                        ? location.pathname === fullPath
                        : location.pathname.startsWith(fullPath)
                      return (
                        <div key={item.label}>
                          {item.group && (
                            <div className="text-[10px] font-medium text-zinc-600 uppercase tracking-wider pt-2 pb-0.5 pl-3">
                              {item.group}
                            </div>
                          )}
                          <NavLink
                            to={fullPath}
                            end={item.path === ''}
                            className={`flex items-center gap-2.5 pl-3 pr-2 py-1.5 rounded-lg text-xs transition-colors ${
                              isItemActive
                                ? 'text-cyan-400 bg-zinc-800/60'
                                : 'text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800/30'
                            }`}
                          >
                            <item.icon className="w-3.5 h-3.5 flex-shrink-0" />
                            <span>{item.label}</span>
                          </NavLink>
                        </div>
                      )
                    })}
                  </div>
                )}
              </div>
            )
          })}
          {(!courses || courses.length === 0) && !collapsed && (
            <div className="px-3 py-4 text-sm text-zinc-600 text-center">
              No courses yet
            </div>
          )}
        </nav>
      </div>

      {/* Bottom links */}
      <div className="border-t border-zinc-800 py-3 px-2 space-y-1">
        <NavLink
          to="/dashboard"
          end
          className={({ isActive }) =>
            `flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors ${
              isActive
                ? 'bg-zinc-800 text-zinc-100'
                : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50'
            }`
          }
        >
          <BookOpen className="w-4 h-4 flex-shrink-0" />
          {!collapsed && <span>Dashboard</span>}
        </NavLink>
        <NavLink
          to="/settings"
          className={({ isActive }) =>
            `flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors ${
              isActive
                ? 'bg-zinc-800 text-zinc-100'
                : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50'
            }`
          }
        >
          <Settings className="w-4 h-4 flex-shrink-0" />
          {!collapsed && <span>Settings</span>}
        </NavLink>
      </div>

      {/* Collapse toggle */}
      <button
        onClick={() => {
          const next = !collapsed
          setCollapsed(next)
          localStorage.setItem('vindexa_sidebar_collapsed', String(next))
        }}
        className="h-10 flex items-center justify-center border-t border-zinc-800 text-zinc-500 hover:text-zinc-300 transition-colors"
      >
        {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
      </button>
    </aside>
  )
}
