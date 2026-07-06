import { Link, useLocation } from 'react-router-dom'
import { ChevronRight, Search } from 'lucide-react'
import { BrandMark } from '@/components/ui/BrandMark'
import AvatarMenu from './AvatarMenu'
import CourseSwitcher from './CourseSwitcher'
import ReadinessChip from './ReadinessChip'
import { parseCoursePath, segmentLabel } from './navModel'

const IS_MAC = typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform)

interface TopBarProps {
  onOpenPalette: () => void
  onNewCourse: () => void
}

/**
 * Slim command-workspace top bar (h-12): brand → course switcher › destination
 * on the left; Cmd+K trigger, readiness chip, avatar menu on the right.
 */
export default function TopBar({ onOpenPalette, onNewCourse }: TopBarProps) {
  const location = useLocation()
  const course = parseCoursePath(location.pathname)
  const segments = location.pathname.split('/').filter(Boolean)

  // Non-course pages get a plain title where the switcher would sit.
  const plainTitle =
    segments[0] === 'settings' ? 'Settings' : segments[0] === 'dashboard' ? 'Dashboard' : null

  return (
    <header className="h-12 flex items-center justify-between gap-3 px-3 top-bar">
      {/* Brand + breadcrumb */}
      <div className="flex items-center gap-2 min-w-0">
        <Link
          to="/dashboard"
          className="flex items-center gap-2 flex-shrink-0 rounded-md focus-ring"
          aria-label="Vindexa — dashboard"
        >
          <BrandMark className="w-6 h-6" />
          <span className="font-display text-[17px] font-semibold text-ink tracking-tight max-sm:hidden">
            Vindexa
          </span>
        </Link>
        <span className="h-4 border-l border-line-strong mx-1 flex-shrink-0" aria-hidden="true" />

        {course ? (
          <div className="flex items-center gap-1 min-w-0">
            <CourseSwitcher courseId={course.courseId} subPath={course.subPath} onNewCourse={onNewCourse} />
            {course.subPath && (
              <>
                <ChevronRight className="w-3.5 h-3.5 flex-shrink-0 text-ink-faint" aria-hidden="true" />
                <span className="text-[13px] text-ink-faint truncate">
                  {segmentLabel(course.subPath.slice(1))}
                </span>
              </>
            )}
          </div>
        ) : (
          <span data-tour="course-switcher" className="text-sm font-semibold text-ink tracking-tight truncate">
            {plainTitle ?? ''}
          </span>
        )}
      </div>

      {/* Palette trigger + readiness + account */}
      <div className="flex items-center gap-2.5 flex-shrink-0">
        <button
          onClick={onOpenPalette}
          data-tour="command-k"
          aria-label="Open command palette"
          className="flex items-center gap-2 h-7 w-56 max-md:w-auto px-2.5 rounded-md bg-surface border border-line text-xs text-ink-faint hover:text-ink-soft hover:border-line-strong transition-colors focus-ring"
        >
          <Search className="w-3.5 h-3.5 flex-shrink-0" />
          <span className="truncate max-md:hidden">Search or jump to…</span>
          <kbd className="ml-auto font-mono text-[10px] text-ink-faint bg-paper-deep border border-line rounded px-1 py-px max-md:hidden">
            {IS_MAC ? '⌘K' : 'Ctrl K'}
          </kbd>
        </button>
        {course && <ReadinessChip courseId={course.courseId} />}
        <AvatarMenu />
      </div>
    </header>
  )
}
