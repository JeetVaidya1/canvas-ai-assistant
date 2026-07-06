import { useState } from 'react'
import type { ReactNode } from 'react'
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
import { Tooltip } from '@/components/ui/Tooltip'
import { BrandMark } from '@/components/ui/BrandMark'
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

/**
 * Wraps a nav item in a right-side Tooltip only when the sidebar is collapsed
 * to icon rail. The `[&>span]:w-full` selector stretches the Tooltip's inline
 * wrapper so hover targets stay full-width.
 */
function CollapsedTip({ label, show, children }: { label: string; show: boolean; children: ReactNode }) {
  if (!show) return <>{children}</>
  return (
    <div className="[&>span]:w-full">
      <Tooltip content={label} side="right" delay={150}>
        {children}
      </Tooltip>
    </div>
  )
}

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

  // One active treatment everywhere: raised white sheet + pen-blue text.
  const activeClass = 'bg-surface text-accent-deep border border-line shadow-[0_1px_2px_rgba(33,31,26,0.05)]'
  const idleClass = 'text-ink-soft hover:text-ink hover:bg-line/40 border border-transparent'

  return (
    <aside
      className={`flex flex-col bg-paper-deep border-r border-line transition-all duration-300 ${
        collapsed ? 'w-16' : 'w-64'
      }`}
    >
      {/* Logo */}
      <div className="h-14 flex items-center px-4 gap-2.5">
        <BrandMark className="w-7 h-7 flex-shrink-0" />
        {!collapsed && (
          <span className="font-display text-[19px] font-semibold text-ink tracking-tight">Vindexa</span>
        )}
      </div>

      {/* Course list */}
      <div className="flex-1 overflow-y-auto py-2">
        {!collapsed && (
          <div className="px-4 mb-1.5 text-[10px] font-medium text-ink-faint uppercase tracking-[0.14em]">
            Courses
          </div>
        )}
        <nav className="space-y-0.5 px-2">
          {courses?.map((course) => {
            const isActiveCourse = courseId === course.course_id
            return (
              <div key={course.course_id}>
                <CollapsedTip label={course.title} show={collapsed}>
                  <NavLink
                    to={`/course/${course.course_id}`}
                    end
                    onMouseEnter={() => prefetchCourse(course.course_id)}
                    className={({ isActive }) =>
                      `w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm font-medium transition-all ${
                        isActive || isActiveCourse ? activeClass : idleClass
                      }`
                    }
                  >
                    <GraduationCap className="w-4 h-4 flex-shrink-0" />
                    {!collapsed && (
                      <span className="truncate">{course.title}</span>
                    )}
                  </NavLink>
                </CollapsedTip>

                {/* Course sub-navigation — six intent-based destinations */}
                {isActiveCourse && !collapsed && (
                  <div className="mt-1 mb-1.5 ml-[1.35rem] pl-3 border-l border-line-strong space-y-0.5">
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
                          className={`flex items-center gap-2.5 px-2.5 py-1.5 rounded-md text-[13px] font-medium transition-all ${
                            isItemActive
                              ? activeClass
                              : 'text-ink-faint hover:text-ink hover:bg-line/40 border border-transparent'
                          }`}
                        >
                          <item.icon className="w-3.5 h-3.5 flex-shrink-0" />
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
            <div className="px-3 py-4 text-sm text-ink-faint text-center">
              No courses yet
            </div>
          )}
        </nav>
      </div>

      {/* Bottom links */}
      <div className="border-t border-line py-2.5 px-2 space-y-0.5">
        <CollapsedTip label="Dashboard" show={collapsed}>
          <NavLink
            to="/dashboard"
            end
            className={({ isActive }) =>
              `w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm transition-colors ${
                isActive ? activeClass : idleClass
              }`
            }
          >
            <BookOpen className="w-4 h-4 flex-shrink-0" />
            {!collapsed && <span>Dashboard</span>}
          </NavLink>
        </CollapsedTip>
        <CollapsedTip label="Settings" show={collapsed}>
          <NavLink
            to="/settings"
            className={({ isActive }) =>
              `w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm transition-colors ${
                isActive ? activeClass : idleClass
              }`
            }
          >
            <Settings className="w-4 h-4 flex-shrink-0" />
            {!collapsed && <span>Settings</span>}
          </NavLink>
        </CollapsedTip>
        <CollapsedTip label="Sign out" show={collapsed}>
          <button
            onClick={() => { void signOut() }}
            className="w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm text-ink-soft hover:text-ink hover:bg-line/40 transition-colors"
          >
            <LogOut className="w-4 h-4 flex-shrink-0" />
            {!collapsed && <span>Sign out</span>}
          </button>
        </CollapsedTip>
      </div>

      {/* Collapse toggle */}
      <button
        onClick={() => {
          const next = !collapsed
          setCollapsed(next)
          localStorage.setItem('vindexa_sidebar_collapsed', String(next))
        }}
        aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        className="h-9 flex items-center justify-center border-t border-line text-ink-faint hover:text-ink transition-colors"
      >
        {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
      </button>
    </aside>
  )
}
