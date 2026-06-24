// src/lib/api/reviews.ts
import { apiFetch } from './client'

/** ===== Mistake-driven review queue (Phase 4) ===== */
export interface ReviewItem {
  id: string
  concept: string
  prompt: string
  answer: string
  explanation: string
  source: string
  due: boolean
  due_date: string | null
}

export interface ReviewQueue {
  items: ReviewItem[]
  due_count: number
  total: number
}

export async function getReviewQueue(courseId: string, userId: string = 'anonymous'): Promise<ReviewQueue> {
  return apiFetch(`/api/reviews/${encodeURIComponent(courseId)}?user_id=${encodeURIComponent(userId)}`)
}

export async function gradeReview(itemId: string, grade: number, userId: string = 'anonymous'): Promise<{ item_id: string; due_date: string; interval: number }> {
  const form = new FormData()
  form.append('grade', String(grade))
  form.append('user_id', userId)
  return apiFetch(`/api/reviews/${encodeURIComponent(itemId)}/grade`, { method: 'POST', body: form })
}
