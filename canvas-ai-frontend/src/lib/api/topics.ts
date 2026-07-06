import { apiFetch } from './client'

export interface CourseTopic {
  slug: string
  name: string
  description: string | null
  doc_coverage: Array<{ doc: string; pages: [number, number] }>
  prereq_slugs: string[]
  position: number
}

export interface CourseTopicsResponse {
  course_id: string
  topics: CourseTopic[]
  count: number
}

/** Course Brain taxonomy — auto-synthesizes server-side on first call. */
export async function getCourseTopics(courseId: string): Promise<CourseTopicsResponse> {
  return apiFetch(`/api/courses/${encodeURIComponent(courseId)}/topics`)
}

/** Force a fresh synthesis (e.g. after adding materials). */
export async function rebuildCourseTopics(courseId: string): Promise<CourseTopicsResponse> {
  return apiFetch(`/api/courses/${encodeURIComponent(courseId)}/topics/rebuild`, { method: 'POST' })
}
