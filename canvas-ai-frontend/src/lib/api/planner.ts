// src/lib/api/planner.ts
import { BASE_URL, apiFetch } from './client'

/** ===== Study Planner ===== */
export interface StudyPlan {
  id: string
  course_id: string
  days: Array<{
    date: string
    topics: string[]
    duration_minutes: number
    type: 'review' | 'new' | 'practice'
  }>
  created_at: string
}

export async function generateStudyPlan(
  courseId: string,
  params: {
    daysAvailable?: number
    hoursPerDay?: number
    examDate?: string
  } = {}
): Promise<StudyPlan> {
  const form = new FormData()
  form.append('course_id', courseId)
  if (params.daysAvailable) form.append('days_available', String(params.daysAvailable))
  if (params.hoursPerDay) form.append('hours_per_day', String(params.hoursPerDay))
  if (params.examDate) form.append('exam_date', params.examDate)
  return apiFetch('/api/generate-study-plan', { method: 'POST', body: form })
}

export async function getStudyPlan(courseId: string): Promise<StudyPlan | null> {
  try {
    return await apiFetch(`/api/study-plan/${encodeURIComponent(courseId)}`)
  } catch {
    return null
  }
}

export async function replanStudyPlan(
  courseId: string,
  userId: string = 'anonymous',
  params: { daysAvailable?: number; hoursPerDay?: number; examDate?: string } = {}
): Promise<StudyPlan> {
  const form = new FormData()
  form.append('course_id', courseId)
  form.append('user_id', userId)
  if (params.daysAvailable) form.append('days_available', String(params.daysAvailable))
  if (params.hoursPerDay) form.append('hours_per_day', String(params.hoursPerDay))
  if (params.examDate) form.append('exam_date', params.examDate)
  return apiFetch('/api/replan', { method: 'POST', body: form }, 120_000)
}

export async function exportPlannerIcal(courseId: string): Promise<Blob> {
  const ctrl = new AbortController()
  const timer = setTimeout(() => ctrl.abort(), 120_000)
  try {
    const resp = await fetch(`${BASE_URL}/api/export-planner-ical/${encodeURIComponent(courseId)}`, {
      signal: ctrl.signal,
    })
    if (!resp.ok) throw new Error('Export failed')
    return await resp.blob()
  } finally {
    clearTimeout(timer)
  }
}
