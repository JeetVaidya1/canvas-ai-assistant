import {
  BarChart3,
  BookOpen,
  ClipboardList,
  Layers,
  LayoutGrid,
  MessageCircle,
  Settings,
  Target,
} from 'lucide-react'

/** A course-scoped destination. `path` is appended to `/course/:id`. */
export interface CourseDestination {
  label: string
  /** '' = course home, otherwise '/learn' etc. */
  path: string
  icon: typeof BookOpen
}

/** Six intent-based destinations (Learn = Chat+Tutor, Practice = Quiz+Practice,
 *  Study Kit = Notes+Flashcards+Audio, Progress = Analytics+Planner). */
export const COURSE_DESTINATIONS: CourseDestination[] = [
  { label: 'Home', path: '', icon: BookOpen },
  { label: 'Learn', path: '/learn', icon: MessageCircle },
  { label: 'Practice', path: '/practice', icon: Target },
  { label: 'Exam', path: '/exam', icon: ClipboardList },
  { label: 'Study Kit', path: '/kit', icon: Layers },
  { label: 'Progress', path: '/progress', icon: BarChart3 },
]

export const DASHBOARD_ITEM = { label: 'Dashboard', path: '/dashboard', icon: LayoutGrid }
export const SETTINGS_ITEM = { label: 'Settings', path: '/settings', icon: Settings }

/** Human labels for route segments (breadcrumb + recent activity). */
export const SEGMENT_LABELS: Record<string, string> = {
  learn: 'Learn',
  practice: 'Practice',
  exam: 'Exam',
  kit: 'Study Kit',
  progress: 'Progress',
  materials: 'Materials',
  // Legacy segments still present in stored recent-activity entries.
  chat: 'Learn',
  tutor: 'Learn',
  quiz: 'Practice',
  notes: 'Study Kit',
  audio: 'Study Kit',
  analytics: 'Progress',
  planner: 'Progress',
  exams: 'Exam',
}

export function segmentLabel(segment: string): string {
  if (!segment) return 'Home'
  return SEGMENT_LABELS[segment] ?? segment.charAt(0).toUpperCase() + segment.slice(1)
}

/** Extracts { courseId, subPath } from a pathname, or null outside courses. */
export function parseCoursePath(pathname: string): { courseId: string; subPath: string } | null {
  const segments = pathname.split('/').filter(Boolean)
  if (segments[0] !== 'course' || !segments[1]) return null
  return { courseId: segments[1], subPath: segments[2] ? `/${segments[2]}` : '' }
}
