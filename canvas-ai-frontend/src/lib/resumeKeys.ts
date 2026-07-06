// src/lib/resumeKeys.ts — single home for every "resume where you left off"
// localStorage key, plus small safe read/write helpers (stored data is never
// trusted: all parsing is guarded and shape-checked by callers).

/** JSON array of quiz ids the user dismissed from "Continue where you left off". */
export const DISMISSED_QUIZZES_KEY = 'vindexa_dismissed_quizzes'

/** Cap so the dismissed list can't grow unbounded across months of use. */
const MAX_DISMISSED = 50

/** Per-course Notes composer draft (topic, style, selected source files). */
export function noteDraftKey(courseId: string): string {
  return `vindexa_note_draft_${courseId}`
}

/** Per-course Problem Set session snapshot. */
export function practiceSnapshotKey(courseId: string): string {
  return `vindexa_practice_${courseId}`
}

/** Parse JSON from localStorage without ever throwing; null on any failure. */
export function readJson(key: string): unknown {
  try {
    const raw = localStorage.getItem(key)
    return raw ? (JSON.parse(raw) as unknown) : null
  } catch {
    return null
  }
}

/** Write JSON to localStorage, swallowing quota/serialization failures. */
export function writeJson(key: string, value: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify(value))
  } catch {
    // Storage full or unavailable — resume is best-effort, never blocking.
  }
}

/** Remove a key, swallowing storage failures. */
export function removeKey(key: string): void {
  try {
    localStorage.removeItem(key)
  } catch {
    // Ignore — nothing actionable for the user.
  }
}

/** Quiz ids the user dismissed; tolerates corrupt or foreign stored data. */
export function readDismissedQuizzes(): readonly string[] {
  const parsed = readJson(DISMISSED_QUIZZES_KEY)
  if (!Array.isArray(parsed)) return []
  return parsed.filter((id): id is string => typeof id === 'string')
}

/** Returns the new dismissed list (immutably) and persists it. */
export function addDismissedQuiz(quizId: string): readonly string[] {
  const current = readDismissedQuizzes()
  if (current.includes(quizId)) return current
  const next = [...current, quizId].slice(-MAX_DISMISSED)
  writeJson(DISMISSED_QUIZZES_KEY, next)
  return next
}
