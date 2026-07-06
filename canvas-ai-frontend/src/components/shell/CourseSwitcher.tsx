import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { ChevronDown, LayoutGrid, Plus } from 'lucide-react'
import { useCourses } from '@/hooks/useCourses'
import { usePrefetch } from '@/hooks/usePrefetch'
import { usePopover } from './usePopover'

interface SwitcherOption {
  key: string
  label: string
  kind: 'course' | 'dashboard' | 'new'
  courseId?: string
}

interface CourseSwitcherProps {
  courseId: string
  /** Current course sub-path ('' for home) — preserved when switching. */
  subPath: string
  onNewCourse: () => void
}

/**
 * TopBar course switcher: the current course name as a popover listbox.
 * Switching lands on the same destination in the other course.
 */
export default function CourseSwitcher({ courseId, subPath, onNewCourse }: CourseSwitcherProps) {
  const navigate = useNavigate()
  const { data: courses } = useCourses()
  const { prefetchCourse } = usePrefetch()
  const { open, setOpen, ref } = usePopover<HTMLDivElement>()
  const [highlighted, setHighlighted] = useState(0)

  const current = courses?.find((c) => c.course_id === courseId)

  const options: SwitcherOption[] = [
    ...(courses ?? []).map((c) => ({
      key: c.course_id,
      label: c.title,
      kind: 'course' as const,
      courseId: c.course_id,
    })),
    { key: '__dashboard', label: 'All courses / Dashboard', kind: 'dashboard' },
    { key: '__new', label: 'New course…', kind: 'new' },
  ]

  // Highlight the current course whenever the listbox opens.
  useEffect(() => {
    if (open) {
      const idx = options.findIndex((o) => o.courseId === courseId)
      setHighlighted(idx >= 0 ? idx : 0)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- options is rebuilt per render; keying on open + courseId is intentional
  }, [open, courseId])

  const select = (option: SwitcherOption) => {
    setOpen(false)
    if (option.kind === 'dashboard') navigate('/dashboard')
    else if (option.kind === 'new') onNewCourse()
    else if (option.courseId && option.courseId !== courseId) {
      navigate(`/course/${option.courseId}${subPath}`)
    }
  }

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (!open) {
      if (e.key === 'ArrowDown' || e.key === 'Enter' || e.key === ' ') {
        e.preventDefault()
        setOpen(true)
      }
      return
    }
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setHighlighted((h) => Math.min(h + 1, options.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setHighlighted((h) => Math.max(h - 1, 0))
    } else if (e.key === 'Enter') {
      e.preventDefault()
      const option = options[highlighted]
      if (option) select(option)
    }
  }

  return (
    <div ref={ref} data-tour="course-switcher" className="relative min-w-0" onKeyDown={onKeyDown}>
      <button
        onClick={() => setOpen(!open)}
        aria-haspopup="listbox"
        aria-expanded={open}
        className="flex items-center gap-1 max-w-[240px] px-1.5 py-1 rounded-md text-sm font-semibold text-ink tracking-tight hover:bg-line/40 transition-colors focus-ring"
      >
        <span className="truncate">{current?.title ?? courseId}</span>
        <ChevronDown className="w-3.5 h-3.5 flex-shrink-0 text-ink-faint" />
      </button>

      {open && (
        <div
          role="listbox"
          aria-label="Switch course"
          className="absolute left-0 top-full mt-1.5 w-64 max-h-80 overflow-y-auto py-1.5 bg-surface border border-line rounded-lg elev-3 z-50"
        >
          <div className="px-3 pt-1 pb-1.5 text-[10px] font-medium text-ink-faint uppercase tracking-[0.14em]">
            Courses
          </div>
          {options.map((option, i) => {
            const isCurrent = option.courseId === courseId
            const isDivided = option.kind !== 'course' && options[i - 1]?.kind === 'course'
            return (
              <div key={option.key}>
                {isDivided && <div className="my-1.5 border-t border-line" />}
                <button
                  role="option"
                  aria-selected={isCurrent}
                  onClick={() => select(option)}
                  onMouseEnter={() => {
                    setHighlighted(i)
                    if (option.courseId) prefetchCourse(option.courseId)
                  }}
                  className={`w-full flex items-center gap-2 px-3 py-1.5 text-left text-[13px] transition-colors ${
                    i === highlighted ? 'bg-accent-wash text-accent-deep' : 'text-ink-soft'
                  } ${isCurrent ? 'font-semibold' : 'font-medium'}`}
                >
                  {option.kind === 'dashboard' && <LayoutGrid className="w-3.5 h-3.5 flex-shrink-0" />}
                  {option.kind === 'new' && <Plus className="w-3.5 h-3.5 flex-shrink-0" />}
                  <span className="truncate">{option.label}</span>
                  {isCurrent && <span className="ml-auto text-[10px] text-ink-faint">current</span>}
                </button>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
