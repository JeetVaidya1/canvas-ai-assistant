// src/lib/api/courses.ts
import { supabase } from '../supabaseClient'
import type { PostgrestError } from '@supabase/supabase-js'
import { apiFetch } from './client'

/** ===== Courses (Supabase direct) ===== */
export interface Course {
  course_id: string
  title: string
  created_at: string
}

export async function claimLegacyData(): Promise<{ claimed: number }> {
  return apiFetch('/api/claim-legacy-data', { method: 'POST' })
}

export async function fetchCourses(): Promise<Course[]> {
  const { data, error } = await supabase
    .from('courses')
    .select('*')
    .order('created_at', { ascending: false })

  if (error) throw error as PostgrestError
  return (data || []) as Course[]
}

export async function addCourse(course_id: string, title: string) {
  const { error } = await supabase
    .from('courses')
    .insert({ course_id, title })

  if (error) throw error as PostgrestError
}

/** ===== Files via backend ===== */
export type UploadedFile = string // just the filename

export async function uploadFiles(
  course_id: string,
  files: File[]
): Promise<UploadedFile[]> {
  const form = new FormData()
  form.append('course_id', course_id)
  files.forEach(f => form.append('files', f))

  // Backend returns: { status, message, files: [{ filename, url, ... }], chunks: [...] }
  const data = await apiFetch('/upload', { method: 'POST', body: form })
  const uploaded = (data.files || []) as Array<{ filename: string }>
  return uploaded.map((f) => f.filename)
}

export async function listFiles(course_id: string): Promise<string[]> {
  const data = await apiFetch(`/list-files?course_id=${encodeURIComponent(course_id)}`)
  return data.files || []
}

export async function deleteFile(course_id: string, filename: string): Promise<void> {
  const form = new FormData()
  form.append('course_id', course_id)
  form.append('filename', filename)
  await apiFetch('/delete-file', { method: 'POST', body: form })
}

/** ===== Courses via backend (create/delete) ===== */
export async function createCourse(courseId: string, title: string): Promise<void> {
  const form = new FormData()
  form.append('course_id', courseId)
  form.append('title', title)
  await apiFetch('/create-course', { method: 'POST', body: form })
}

export async function deleteCourse(courseId: string): Promise<void> {
  const form = new FormData()
  form.append('course_id', courseId)
  await apiFetch('/delete-course', { method: 'POST', body: form })
}
