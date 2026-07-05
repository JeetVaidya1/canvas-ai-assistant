import { useState } from 'react'
import { NavLink, useParams, useLocation } from 'react-router-dom'
import {
  BookOpen,
  Home,
  Settings,
  ChevronLeft,
  ChevronRight,
  GraduationCap,
  MessageCircle,
  Target,
  ClipboardList,
  BarChart3,
  Layers,
  LogOut,
} from 'lucide-react'
import { useCourses } from '@/hooks/useCourses'
import { usePrefetch } from '@/hooks/usePrefetch'
import { useAuth } from '@/lib/auth'

interface SubNavItem {
  label: string
  path: string
  icon: typeof BookOpen
}

// Six intent-based destinations (was 10 overlapping tools). Learn = Chat+Tutor,
// Practice = Quiz+Practice, Study Kit = Notes+Flashcards+Audio, Progress =
// Analytics+Planner. See App.tsx for the old→new redirects.
const courseSubNav: SubNavItem[] = [
  { label: 'Home', path: '', icon: Home },
  { label: 'Learn', path: '/learn', icon: MessageCircle },
  { label: 'Practice', path: '/practice', icon: Target },
  { label: 'Exam', path: '/exam', icon: ClipboardList },
  { label: 'Study Kit', path: '/kit', icon: Layers },
  { label: 'Progress', path: '/progress', icon: BarChart3 },
]

export default function AppSidebar() {
  const [collapsed, setCollapsed] = useState(() => {
    return localStorage.getItem('vindexa_sidebar_collapsed') === 'true'
  })
  const { courseId } = useParams()
  const location = useLocation()
  const { data: courses } = useCourses()
  const { signOut } = useAuth()
  const { prefetchCourse, prefetchLearn, prefetchPractice, prefetchStudyKit, prefetchProgress } =
    usePrefetch()

  // Warm the destination's primary data while the cursor hovers a nav link.
  const prefetchSubNav = (targetCourseId: string, path: string) => {
    if (path === '') prefetchCourse(targetCourseId)
    else if (path === '/learn') prefetchLearn()
    else if (path === '/practice') prefetchPractice(targetCourseId)
    else if (path === '/kit') prefetchStudyKit(targetCourseId)
    else if (path === '/progress') prefetchProgress(targetCourseId)
  }

  return (
    <aside
      className={`flex flex-col bg-[#0a0a0d] border-r border-[#18181d] transition-all duration-300 ${
        collapsed ? 'w-16' : 'w-64'
      }`}
    >
      {/* Logo */}
      <div className="h-14 flex items-center px-4 gap-2.5">
        <img src="/favicon-32x32.png" alt="Vindexa" className="w-7 h-7 rounded-lg flex-shrink-0" />
        {!collapsed && (
          <span className="text-[17px] font-semibold text-gradient-brand tracking-tight">Vindexa</span>
        )}
      </div>

      {/* Course list */}
      <div className="flex-1 overflow-y-auto py-2">
        {!collapsed && (
          <div className="px-4 mb-1.5 text-[10px] font-semibold text-zinc-600 uppercase tracking-[0.16em]">
            Courses
          </div>
        )}
        <nav className="space-y-0.5 px-2">
          {courses?.map((course) => {
            const isActiveCourse = courseId === course.course_id
            return (
              <div key={course.course_id}>
                <NavLink
                  to={`/course/${course.course_id}`}
                  end
                  onMouseEnter={() => prefetchCourse(course.course_id)}
                  className={({ isActive }) =>
                    `flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm font-medium transition-all ${
                      isActive || isActiveCourse
                        ? 'bg-gradient-brand-soft text-cyan-100 ring-1 ring-inset ring-cyan-400/20'
                        : 'text-zinc-400 hover:text-zinc-100 hover:bg-white/[0.04]'
                    }`
                  }
                >
                  <GraduationCap className={`w-4 h-4 flex-shrink-0 ${isActiveCourse ? 'text-cyan-300' : ''}`} />
                  {!collapsed && (
                    <span className="truncate">{course.title}</span>
                  )}
                </NavLink>

                {/* Course sub-navigation — six intent-based destinations */}
                {isActiveCourse && !collapsed && (
                  <div className="mt-1 mb-1.5 ml-[1.35rem] pl-3 border-l border-[#1e1e24] space-y-0.5">
                    {courseSubNav.map((item) => {
                      const fullPath = `/course/${course.course_id}${item.path}`
                      const isItemActive = item.path === ''
                        ? location.pathname === fullPath
                        : location.pathname.startsWith(fullPath)
                      return (
                        <NavLink
                          key={item.label}
                          to={fullPath}
                          end={item.path === ''}
                          onMouseEnter={() => prefetchSubNav(course.course_id, item.path)}
                          className={`relative flex items-center gap-2.5 px-2.5 py-1.5 rounded-md text-[13px] font-medium transition-all ${
                            isItemActive
                              ? 'text-cyan-100 bg-cyan-500/10'
                              : 'text-zinc-500 hover:text-zinc-200 hover:bg-white/[0.03]'
                          }`}
                        >
                          {isItemActive && (
                            <span className="absolute -left-[calc(0.75rem+1px)] top-1/2 -translate-y-1/2 h-4 w-[2px] rounded-full bg-gradient-to-b from-cyan-400 to-blue-500" />
                          )}
                          <item.icon className={`w-3.5 h-3.5 flex-shrink-0 ${isItemActive ? 'text-cyan-300' : ''}`} />
                          <span>{item.label}</span>
                        </NavLink>
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
      <div className="border-t border-[#18181d] py-2.5 px-2 space-y-0.5">
        <NavLink
          to="/dashboard"
          end
          className={({ isActive }) =>
            `flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm transition-colors ${
              isActive
                ? 'bg-white/[0.06] text-zinc-100'
                : 'text-zinc-400 hover:text-zinc-100 hover:bg-white/[0.04]'
            }`
          }
        >
          <BookOpen className="w-4 h-4 flex-shrink-0" />
          {!collapsed && <span>Dashboard</span>}
        </NavLink>
        <NavLink
          to="/settings"
          className={({ isActive }) =>
            `flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm transition-colors ${
              isActive
                ? 'bg-white/[0.06] text-zinc-100'
                : 'text-zinc-400 hover:text-zinc-100 hover:bg-white/[0.04]'
            }`
          }
        >
          <Settings className="w-4 h-4 flex-shrink-0" />
          {!collapsed && <span>Settings</span>}
        </NavLink>
        <button
          onClick={() => { void signOut() }}
          className="w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm text-zinc-400 hover:text-zinc-100 hover:bg-white/[0.04] transition-colors"
        >
          <LogOut className="w-4 h-4 flex-shrink-0" />
          {!collapsed && <span>Sign out</span>}
        </button>
      </div>

      {/* Collapse toggle */}
      <button
        onClick={() => {
          const next = !collapsed
          setCollapsed(next)
          localStorage.setItem('vindexa_sidebar_collapsed', String(next))
        }}
        className="h-9 flex items-center justify-center border-t border-[#18181d] text-zinc-600 hover:text-zinc-300 transition-colors"
      >
        {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
      </button>
    </aside>
  )
}
