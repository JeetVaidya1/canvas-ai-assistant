// src/lib/api/flashcards.ts
import { BASE_URL, apiFetch } from './client'

/** ===== Flashcards — spaced repetition (Phase 3) ===== */
export interface DeckCard {
  id: string
  q: string
  a: string
  due: boolean
  ease: number
  interval: number
  repetitions: number
  due_date: string | null
}

export interface FlashcardDeck {
  cards: DeckCard[]
  total: number
  due_count: number
}

export async function saveFlashcards(
  courseId: string,
  cards: { q: string; a: string }[],
  noteId?: string
): Promise<{ saved: number; skipped: number }> {
  const form = new FormData()
  form.append('course_id', courseId)
  form.append('cards', JSON.stringify(cards))
  if (noteId) form.append('note_id', noteId)
  return apiFetch('/flashcards/save', { method: 'POST', body: form })
}

export async function getFlashcardDeck(courseId: string, userId: string = 'anonymous'): Promise<FlashcardDeck> {
  return apiFetch(`/flashcards/${encodeURIComponent(courseId)}?user_id=${encodeURIComponent(userId)}`)
}

export async function reviewFlashcard(
  cardId: string,
  grade: number,
  userId: string = 'anonymous'
): Promise<{ card_id: string; due_date: string; interval: number; ease: number; repetitions: number }> {
  const form = new FormData()
  form.append('card_id', cardId)
  form.append('grade', String(grade))
  form.append('user_id', userId)
  return apiFetch('/flashcards/review', { method: 'POST', body: form })
}

export async function exportFlashcardsAnki(courseId: string, userId?: string): Promise<Blob> {
  const ctrl = new AbortController()
  const timer = setTimeout(() => ctrl.abort(), 120_000)
  try {
    const qs = userId ? `?user_id=${encodeURIComponent(userId)}` : ''
    const resp = await fetch(`${BASE_URL}/api/export-flashcards-anki/${encodeURIComponent(courseId)}${qs}`, {
      signal: ctrl.signal,
    })
    if (!resp.ok) throw new Error('Export failed')
    return await resp.blob()
  } finally {
    clearTimeout(timer)
  }
}
