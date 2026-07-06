import type { PracticeProblem } from '@/lib/api'

/** Backend may attach a grounding hint; read it defensively without altering the data layer. */
export type ProblemSource = 'materials' | 'general'

export type DifficultyBadgeTone = 'success' | 'warning' | 'danger' | 'accent'

/** mm:ss for session timers. */
export function formatTime(totalSeconds: number): string {
  const mins = Math.floor(totalSeconds / 60)
  const secs = totalSeconds % 60
  return `${mins}:${secs.toString().padStart(2, '0')}`
}

/** Normalizes the backend's free-form estimated-time string ("3", "~2 min", …). */
export function formatEstimatedTime(raw: string): string {
  const trimmed = (raw || '').trim()
  if (!trimmed) return '~1 min'
  if (/\d/.test(trimmed) && !/min|sec/i.test(trimmed)) return `~${trimmed} min`
  return trimmed.startsWith('~') ? trimmed : `~${trimmed}`
}

/** Maps a raw difficulty string onto the app-wide Badge tone system. */
export function resolveDifficultyBadge(raw: string): { label: string; tone: DifficultyBadgeTone } {
  const key = (raw || '').trim().toLowerCase()
  if (key === 'easy') return { label: 'Easy', tone: 'success' }
  if (key === 'medium') return { label: 'Medium', tone: 'warning' }
  if (key === 'hard') return { label: 'Hard', tone: 'danger' }
  return { label: raw || 'Adaptive', tone: 'accent' }
}

/** Infer whether a problem is grounded in course materials vs. general knowledge. */
export function resolveProblemSource(problem: PracticeProblem): ProblemSource {
  const probe = problem as PracticeProblem & {
    source?: string
    grounded?: boolean
    from_materials?: boolean
  }
  if (typeof probe.grounded === 'boolean') return probe.grounded ? 'materials' : 'general'
  if (typeof probe.from_materials === 'boolean') return probe.from_materials ? 'materials' : 'general'
  const s = (probe.source || '').toLowerCase()
  if (s.includes('general')) return 'general'
  if (s.includes('material') || s.includes('course') || s.includes('document') || s.includes('retriev')) {
    return 'materials'
  }
  // Default: the practice backend grounds in retrieved course content.
  return 'materials'
}
