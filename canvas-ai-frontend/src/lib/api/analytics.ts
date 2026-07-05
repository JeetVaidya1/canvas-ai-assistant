// src/lib/api/analytics.ts
import { apiFetch } from './client'

/** ===== Concept prerequisite graph (Phase 4) ===== */
export interface ConceptBlocker {
  concept: string
  prerequisite: string
  concept_pct: number
  prerequisite_pct: number
}

export interface ConceptGraph {
  concepts: Array<{ concept: string; mastery_pct: number; has_data: boolean }>
  edges: Array<{ prerequisite: string; concept: string }>
  blockers: ConceptBlocker[]
  exists: boolean
}

export async function getConceptGraph(courseId: string, userId: string = 'anonymous'): Promise<ConceptGraph> {
  // Auto-builds server-side on first call; allow time for the LLM extraction.
  return apiFetch(`/api/concept-graph/${encodeURIComponent(courseId)}/${encodeURIComponent(userId)}`, undefined, 120_000)
}

/** ===== Exam readiness (Phase 4) ===== */
export interface Readiness {
  score_pct: number
  by_topic: Array<{ topic: string; mastery_pct: number; weight: number; has_data: boolean }>
  gaps: string[]
  confidence: 'low' | 'medium' | 'high'
  has_past_papers: boolean
  data_points?: number
  message?: string
}

export async function getReadiness(courseId: string, userId: string = 'anonymous'): Promise<Readiness> {
  return apiFetch(`/api/readiness/${encodeURIComponent(courseId)}/${encodeURIComponent(userId)}`)
}

/** ===== Analytics ===== */
export interface LearningAnalytics {
  topics_progress: Array<{
    topic: string
    mastery_level: number
    review_count: number
    last_reviewed: string
  }>
  study_streak: number
  weak_areas: string[]
  study_recommendations: string[]
  total_questions: number
  avg_confidence: number
  study_time_trend: Array<{
    date: string
    questions: number
    duration_minutes?: number
    avg_confidence?: number
  }>
}

export async function getLearningAnalytics(courseId: string, userId: string = 'anonymous'): Promise<LearningAnalytics> {
  const data = await apiFetch(`/analytics/${encodeURIComponent(courseId)}/${encodeURIComponent(userId)}`)
  return data.analytics
}

/** ===== Interaction tracking (fire-and-forget) ===== */
export async function trackInteraction(
  userId: string,
  courseId: string,
  question: string,
  answer: string,
  confidence: number,
  responseTime: number
): Promise<void> {
  const form = new FormData()
  form.append('user_id', userId)
  form.append('course_id', courseId)
  form.append('question', question)
  form.append('answer', answer)
  form.append('confidence', String(confidence))
  form.append('response_time', String(responseTime))
  try {
    await apiFetch('/track-interaction', { method: 'POST', body: form })
  } catch {
    // tracking should not break UX
  }
}

/** ===== Optional: system status/health ===== */
export async function getSystemStatus() {
  return apiFetch('/system-status')
}

export async function getRagHealth() {
  return apiFetch('/rag-health')
}
