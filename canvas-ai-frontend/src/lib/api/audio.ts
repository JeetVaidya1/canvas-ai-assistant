// src/lib/api/audio.ts
import { apiFetch } from './client'

/** ===== Audio Overview ===== */
export interface AudioOverview {
  id: string
  course_id: string
  title: string
  audio_url: string
  style: string
  duration_seconds: number
  created_at: string
}

export async function generateAudioOverview(
  courseId: string,
  style: 'summary' | 'lecture' | 'podcast' = 'summary'
): Promise<AudioOverview> {
  const form = new FormData()
  form.append('course_id', courseId)
  form.append('style', style)
  return apiFetch('/api/generate-audio-overview', { method: 'POST', body: form }, 300_000)
}

export async function getAudioOverviews(courseId: string): Promise<AudioOverview[]> {
  try {
    const data = await apiFetch(`/api/audio-overviews/${encodeURIComponent(courseId)}`)
    return data.overviews || []
  } catch {
    return []
  }
}
