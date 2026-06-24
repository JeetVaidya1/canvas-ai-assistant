// src/lib/api/sharing.ts
import { apiFetch } from './client'

/** ===== Shared class courses (Phase 5) ===== */
export interface SharedCourse {
  course_id: string
  share_code: string
  title: string
  subject?: string
  school?: string
  term?: string
  description?: string
  join_count: number
}

export async function publishCourse(
  courseId: string,
  userId: string,
  meta: { subject?: string; school?: string; term?: string; description?: string } = {}
): Promise<{ course_id: string; share_code: string; republished: boolean }> {
  const form = new FormData()
  form.append('course_id', courseId)
  form.append('user_id', userId)
  if (meta.subject) form.append('subject', meta.subject)
  if (meta.school) form.append('school', meta.school)
  if (meta.term) form.append('term', meta.term)
  if (meta.description) form.append('description', meta.description)
  return apiFetch('/api/courses/publish', { method: 'POST', body: form })
}

export async function getShareInfo(courseId: string): Promise<SharedCourse | null> {
  return apiFetch(`/api/courses/${encodeURIComponent(courseId)}/share`)
}

export async function browseSharedCourses(q = ''): Promise<SharedCourse[]> {
  const data = await apiFetch(`/api/shared-courses?q=${encodeURIComponent(q)}`)
  return data.courses || []
}

export async function joinCourse(shareCode: string, userId: string): Promise<{ course_id: string; title: string; newly_joined: boolean }> {
  const form = new FormData()
  form.append('share_code', shareCode)
  form.append('user_id', userId)
  return apiFetch('/api/courses/join', { method: 'POST', body: form })
}
