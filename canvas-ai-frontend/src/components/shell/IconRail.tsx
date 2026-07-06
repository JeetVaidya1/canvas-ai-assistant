import { NavLink, useLocation } from 'react-router-dom'
import { Tooltip } from '@/components/ui/Tooltip'
import { usePrefetch } from '@/hooks/usePrefetch'
import { COURSE_DESTINATIONS, DASHBOARD_ITEM, SETTINGS_ITEM, parseCoursePath } from './navModel'
import type { CourseDestination } from './navModel'

// One active treatment: raised white sheet pill + pen-blue glyph.
const ACTIVE = 'bg-surface text-accent-deep border border-line shadow-[0_1px_2px_rgba(33,31,26,0.05)]'
const IDLE = 'text-ink-soft hover:text-ink hover:bg-line/40 border border-transparent'

interface RailItem {
  label: string
  to: string
  icon: CourseDestination['icon']
  end: boolean
  onHover?: () => void
}

function useRailItems(): { top: RailItem[]; bottom: RailItem[] } {
  const location = useLocation()
  const course = parseCoursePath(location.pathname)
  const { prefetchCourse, prefetchLearn, prefetchPractice, prefetchStudyKit, prefetchProgress } =
    usePrefetch()

  const dashboard: RailItem = { ...DASHBOARD_ITEM, to: DASHBOARD_ITEM.path, end: true }
  const settings: RailItem = { ...SETTINGS_ITEM, to: SETTINGS_ITEM.path, end: false }

  if (!course) return { top: [dashboard], bottom: [settings] }

  const prefetchFor = (path: string) => {
    if (path === '') prefetchCourse(course.courseId)
    else if (path === '/learn') prefetchLearn()
    else if (path === '/practice') prefetchPractice(course.courseId)
    else if (path === '/kit') prefetchStudyKit(course.courseId)
    else if (path === '/progress') prefetchProgress(course.courseId)
  }

  const courseItems: RailItem[] = COURSE_DESTINATIONS.map((dest) => ({
    label: dest.label,
    to: `/course/${course.courseId}${dest.path}`,
    icon: dest.icon,
    end: dest.path === '',
    onHover: () => prefetchFor(dest.path),
  }))

  return { top: [dashboard, ...courseItems], bottom: [settings] }
}

function RailLink({ item, tooltipSide }: { item: RailItem; tooltipSide: 'right' | 'top' }) {
  return (
    <Tooltip content={item.label} side={tooltipSide} delay={150}>
      <NavLink
        to={item.to}
        end={item.end}
        onMouseEnter={item.onHover}
        aria-label={item.label}
        className={({ isActive }) =>
          `flex items-center justify-center w-9 h-9 rounded-lg transition-all focus-ring ${
            isActive ? ACTIVE : IDLE
          }`
        }
      >
        <item.icon className="w-4 h-4" aria-hidden="true" />
      </NavLink>
    </Tooltip>
  )
}

/**
 * Dense icon-only left rail (w-12, ≥md). Dashboard on top, the six course
 * destinations when a course is open, Settings pinned at the bottom.
 */
export default function IconRail() {
  const { top, bottom } = useRailItems()

  return (
    <nav
      aria-label="Primary"
      className="hidden md:flex flex-col items-center w-12 py-2 bg-paper-deep border-r border-line"
    >
      <div className="flex flex-col items-center gap-1">
        {top.map((item, i) => (
          <div key={item.to} className="flex flex-col items-center gap-1">
            {i === 1 && <div className="w-6 border-t border-line-strong my-1" aria-hidden="true" />}
            <RailLink item={item} tooltipSide="right" />
          </div>
        ))}
      </div>
      <div className="mt-auto flex flex-col items-center gap-1">
        {bottom.map((item) => (
          <RailLink key={item.to} item={item} tooltipSide="right" />
        ))}
      </div>
    </nav>
  )
}

/**
 * Mobile answer: below md the rail becomes a bottom icon bar with the same
 * items (Dashboard · course destinations · Settings) — part of the flex
 * column, so the single-scroll contract holds without fixed-position padding.
 */
export function MobileNavBar() {
  const { top, bottom } = useRailItems()
  const items = [...top, ...bottom]

  return (
    <nav
      aria-label="Primary"
      className="flex md:hidden items-center justify-around h-12 px-1 bg-paper-deep border-t border-line"
    >
      {items.map((item) => (
        <RailLink key={item.to} item={item} tooltipSide="top" />
      ))}
    </nav>
  )
}
