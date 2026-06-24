// src/lib/api/tutor.ts
import { apiFetch } from './client'

/** ===== Tutor modes (Phase 5) ===== */
export interface TutorTurn {
  role: 'user' | 'assistant'
  content: string
}

export async function socraticTurn(
  courseId: string,
  message: string,
  history: TutorTurn[] = []
): Promise<{ reply: string }> {
  const form = new FormData()
  form.append('course_id', courseId)
  form.append('message', message)
  form.append('history', JSON.stringify(history))
  return apiFetch('/api/socratic', { method: 'POST', body: form }, 90_000)
}

export interface FeynmanResult {
  score_pct: number
  verdict: 'solid' | 'partial' | 'shaky'
  strengths: string[]
  gaps: string[]
  misconceptions: string[]
  summary: string
  review_items_added: number
}

export async function feynmanEvaluate(
  courseId: string,
  concept: string,
  explanation: string,
  userId: string = 'anonymous'
): Promise<FeynmanResult> {
  const form = new FormData()
  form.append('course_id', courseId)
  form.append('concept', concept)
  form.append('explanation', explanation)
  form.append('user_id', userId)
  return apiFetch('/api/feynman', { method: 'POST', body: form }, 90_000)
}
