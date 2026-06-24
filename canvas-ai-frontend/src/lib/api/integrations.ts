// src/lib/api/integrations.ts
import { BASE_URL, apiFetch } from './client'

/** ===== Export to AIs (Phase 4) ===== */
export async function getContextPack(courseId: string, userId: string = 'anonymous'): Promise<string> {
  const data = await apiFetch(`/api/context-pack/${encodeURIComponent(courseId)}/${encodeURIComponent(userId)}`, undefined, 120_000)
  return data.markdown || ''
}

/** ===== GitHub / Markdown interop (Phase 4) ===== */
export async function exportCourseMarkdown(courseId: string): Promise<Blob> {
  const ctrl = new AbortController()
  const timer = setTimeout(() => ctrl.abort(), 60_000)
  try {
    const resp = await fetch(`${BASE_URL}/api/export-markdown/${encodeURIComponent(courseId)}`, { signal: ctrl.signal })
    if (!resp.ok) throw new Error('Markdown export failed')
    return await resp.blob()
  } finally {
    clearTimeout(timer)
  }
}

export async function githubPush(courseId: string, repo: string, token: string, basePath = 'vindexa'): Promise<{ pushed: number; files: string[]; repo: string; branch: string }> {
  const form = new FormData()
  form.append('course_id', courseId)
  form.append('repo', repo)
  form.append('token', token)
  form.append('base_path', basePath)
  return apiFetch('/api/github/push', { method: 'POST', body: form }, 120_000)
}

export async function githubImport(courseId: string, repo: string, token?: string, subdir = ''): Promise<{ imported: number; skipped: number; files: string[]; message?: string }> {
  const form = new FormData()
  form.append('course_id', courseId)
  form.append('repo', repo)
  if (token) form.append('token', token)
  if (subdir) form.append('subdir', subdir)
  return apiFetch('/api/github/import', { method: 'POST', body: form }, 300_000)
}

/** ===== Canvas LMS Import ===== */
export interface CanvasImportResult {
  syllabus_imported: boolean
  assignments: Array<{ name: string; due_at: string; is_exam: boolean }>
  exam_dates: Array<{ name: string; due_at: string; is_exam: boolean }>
  next_exam_date: string | null
  materials_imported: number
  errors: string[]
}

export async function importCanvasLms(
  baseUrl: string,
  token: string,
  canvasCourseId: string,
  courseId: string
): Promise<CanvasImportResult> {
  const form = new FormData()
  form.append('canvas_base_url', baseUrl)
  form.append('canvas_token', token)
  form.append('canvas_course_id', canvasCourseId)
  form.append('course_id', courseId)
  return apiFetch('/api/import-canvas', { method: 'POST', body: form }, 300_000)
}
