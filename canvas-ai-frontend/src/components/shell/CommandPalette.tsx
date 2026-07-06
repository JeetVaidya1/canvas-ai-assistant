import { useEffect, useMemo, useRef, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import {
  ClipboardList,
  CornerDownLeft,
  History,
  MessageCircle,
  PenLine,
  Plus,
  Search,
  Target,
} from 'lucide-react'
import { useCourses } from '@/hooks/useCourses'
import { useRecentActivity } from '@/hooks/useRecentActivity'
import { COURSE_DESTINATIONS, DASHBOARD_ITEM, SETTINGS_ITEM, parseCoursePath, segmentLabel } from './navModel'
import { fuzzyScore } from './fuzzy'

interface PaletteItem {
  key: string
  group: 'Navigation' | 'Actions' | 'Recent'
  label: string
  hint?: string
  icon: typeof Search
  run: () => void
}

interface CommandPaletteProps {
  open: boolean
  onClose: () => void
  onNewCourse: () => void
}

const GROUP_ORDER: PaletteItem['group'][] = ['Recent', 'Actions', 'Navigation']
const MAX_RESULTS = 12

/**
 * Cmd+K palette: client-side index of navigations (courses × destinations),
 * real actions (each one lands on the page that does the work), and recent
 * activity. Modal white sheet on an ink scrim — no external cmdk dependency.
 */
export default function CommandPalette({ open, onClose, onNewCourse }: CommandPaletteProps) {
  const navigate = useNavigate()
  const location = useLocation()
  const { data: courses } = useCourses()
  const recent = useRecentActivity()
  const [query, setQuery] = useState('')
  const [selected, setSelected] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)
  const listRef = useRef<HTMLDivElement>(null)

  const courseCtx = parseCoursePath(location.pathname)
  const contextCourseId = courseCtx?.courseId ?? null

  useEffect(() => {
    if (open) {
      setQuery('')
      setSelected(0)
      // Focus after the sheet mounts.
      requestAnimationFrame(() => inputRef.current?.focus())
    }
  }, [open])

  const items = useMemo<PaletteItem[]>(() => {
    if (!open) return []
    const go = (path: string) => () => {
      onClose()
      navigate(path)
    }
    const courseTitle = (id: string) => courses?.find((c) => c.course_id === id)?.title ?? id

    const navigation: PaletteItem[] = [
      { key: 'nav-dashboard', group: 'Navigation', label: 'Dashboard', icon: DASHBOARD_ITEM.icon, run: go('/dashboard') },
      { key: 'nav-settings', group: 'Navigation', label: 'Settings', icon: SETTINGS_ITEM.icon, run: go('/settings') },
      ...(courses ?? []).flatMap((course) =>
        COURSE_DESTINATIONS.map((dest) => ({
          key: `nav-${course.course_id}-${dest.path || 'home'}`,
          group: 'Navigation' as const,
          label: `${course.title} › ${dest.label}`,
          icon: dest.icon,
          run: go(`/course/${course.course_id}${dest.path}`),
        })),
      ),
    ]

    // Course-scoped actions only exist inside a course; each navigates to the
    // page where the action really lives (no fake side effects).
    const actions: PaletteItem[] = [
      ...(contextCourseId
        ? [
            { key: 'act-drill', group: 'Actions' as const, label: 'Start a quick drill', hint: courseTitle(contextCourseId), icon: Target, run: go(`/course/${contextCourseId}/practice`) },
            { key: 'act-note', group: 'Actions' as const, label: 'New note', hint: courseTitle(contextCourseId), icon: PenLine, run: go(`/course/${contextCourseId}/kit`) },
            { key: 'act-ask', group: 'Actions' as const, label: 'Ask a question', hint: courseTitle(contextCourseId), icon: MessageCircle, run: go(`/course/${contextCourseId}/learn`) },
            { key: 'act-topics', group: 'Actions' as const, label: 'Rebuild topics', hint: courseTitle(contextCourseId), icon: ClipboardList, run: go(`/course/${contextCourseId}`) },
          ]
        : []),
      {
        key: 'act-new-course',
        group: 'Actions',
        label: 'New course',
        icon: Plus,
        run: () => {
          onClose()
          onNewCourse()
        },
      },
    ]

    const recents: PaletteItem[] = recent
      .filter((entry) => courses?.some((c) => c.course_id === entry.courseId))
      .map((entry) => ({
        key: `recent-${entry.courseId}-${entry.page}`,
        group: 'Recent' as const,
        label: `${courseTitle(entry.courseId)} › ${segmentLabel(entry.page)}`,
        icon: History,
        run: go(`/course/${entry.courseId}${entry.page ? `/${entry.page}` : ''}`),
      }))

    return [...recents, ...actions, ...navigation]
  }, [open, courses, recent, contextCourseId, navigate, onClose, onNewCourse])

  const results = useMemo(() => {
    if (!query.trim()) return items
    const scored = items
      .map((item) => ({ item, score: fuzzyScore(query, item.label) }))
      .filter((r) => r.score > 0)
    return [...scored].sort((a, b) => b.score - a.score).slice(0, MAX_RESULTS).map((r) => r.item)
  }, [items, query])

  // Keep the selection in bounds as the result set changes.
  useEffect(() => {
    setSelected(0)
  }, [query])

  useEffect(() => {
    listRef.current
      ?.querySelector(`[data-index="${selected}"]`)
      ?.scrollIntoView({ block: 'nearest' })
  }, [selected])

  if (!open) return null

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setSelected((s) => Math.min(s + 1, results.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setSelected((s) => Math.max(s - 1, 0))
    } else if (e.key === 'Enter') {
      e.preventDefault()
      results[selected]?.run()
    } else if (e.key === 'Escape') {
      e.preventDefault()
      onClose()
    }
  }

  // Group results in a stable order while preserving rank inside each group.
  const grouped = GROUP_ORDER.map((group) => ({
    group,
    entries: results
      .map((item, index) => ({ item, index }))
      .filter(({ item }) => item.group === group),
  })).filter(({ entries }) => entries.length > 0)

  return (
    <div
      className="fixed inset-0 z-50 bg-ink/30 flex items-start justify-center px-4 pt-[12vh]"
      onPointerDown={(e) => {
        if (e.target === e.currentTarget) onClose()
      }}
      role="dialog"
      aria-modal="true"
      aria-label="Command palette"
    >
      <div className="w-full max-w-xl bg-surface border border-line rounded-xl elev-3 overflow-hidden animate-fade-up">
        <div className="flex items-center gap-2.5 px-4 h-12 border-b border-line">
          <Search className="w-4 h-4 text-ink-faint flex-shrink-0" aria-hidden="true" />
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Search or jump to…"
            aria-label="Search commands and destinations"
            className="flex-1 bg-transparent text-sm text-ink placeholder:text-ink-faint outline-none"
          />
          <kbd className="font-mono text-[10px] text-ink-faint bg-paper-deep border border-line rounded px-1 py-px">
            esc
          </kbd>
        </div>

        <div ref={listRef} className="max-h-[46vh] overflow-y-auto py-1.5">
          {grouped.length === 0 && (
            <div className="px-4 py-8 text-center text-sm text-ink-faint">
              Nothing matches “{query}”
            </div>
          )}
          {grouped.map(({ group, entries }) => (
            <div key={group}>
              <div className="section-num px-4 pt-2.5 pb-1 text-ink-faint">{group}</div>
              {entries.map(({ item, index }) => (
                <button
                  key={item.key}
                  data-index={index}
                  onClick={item.run}
                  onMouseEnter={() => setSelected(index)}
                  className={`w-full flex items-center gap-2.5 px-4 py-2 text-left text-[13px] transition-colors ${
                    index === selected ? 'bg-accent-wash text-accent-deep' : 'text-ink-soft'
                  }`}
                >
                  <item.icon className="w-3.5 h-3.5 flex-shrink-0" aria-hidden="true" />
                  <span className="truncate font-medium">{item.label}</span>
                  {item.hint && <span className="text-[11px] text-ink-faint truncate">{item.hint}</span>}
                  {index === selected && (
                    <CornerDownLeft className="w-3 h-3 ml-auto flex-shrink-0 text-ink-faint" aria-hidden="true" />
                  )}
                </button>
              ))}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
