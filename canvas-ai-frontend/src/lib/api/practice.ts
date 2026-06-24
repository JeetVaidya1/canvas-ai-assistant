// src/lib/api/practice.ts
import { apiFetch } from './client'

/** ===== Practice Mode ===== */
export interface PracticeProblem {
  question: string
  options: string[]
  correct_answer: string
  explanation: string
  estimated_time: string
  difficulty: string
  topic: string
}

export async function generatePracticeProblems(
  courseId: string,
  topic: string,
  difficulty: string = 'adaptive',
  count: number = 5,
  userId: string = 'anonymous'
): Promise<PracticeProblem[]> {
  const form = new FormData()
  form.append('course_id', courseId)
  form.append('topic', topic)
  form.append('difficulty', difficulty)
  form.append('count', String(count))
  form.append('user_id', userId)
  // Generation runs retrieval + reranker + LLM; allow a generous timeout.
  const data = await apiFetch('/generate-practice', { method: 'POST', body: form }, 120_000)
  return data.problems
}

export async function trackPracticeSession(
  userId: string,
  courseId: string,
  topic: string,
  problemsAttempted: number,
  problemsCorrect: number,
  durationMinutes: number,
  difficultyLevel: string
): Promise<void> {
  const form = new FormData()
  form.append('user_id', userId)
  form.append('course_id', courseId)
  form.append('topic', topic)
  form.append('problems_attempted', String(problemsAttempted))
  form.append('problems_correct', String(problemsCorrect))
  form.append('duration_minutes', String(durationMinutes))
  form.append('difficulty_level', difficultyLevel)

  await apiFetch('/track-practice-session', { method: 'POST', body: form })
}

/** ===== Topics from content ===== */
export async function getPracticeTopics(courseId: string): Promise<string[]> {
  try {
    const data = await apiFetch(`/practice-topics/${encodeURIComponent(courseId)}`)
    if (Array.isArray(data.topics) && data.topics.length > 0) return data.topics
    return ['Course Content', 'General Review']
  } catch {
    return ['Course Content', 'General Review']
  }
}
