// src/lib/api/chat.ts
import { BASE_URL, apiFetch } from './client'

/** ===== Chat + sessions ===== */
export interface ChatSession {
  id: string
  user_id: string
  course_id: string
  title: string
  created_at: string
}

export interface Source {
  file: string
  page?: number | null
}

export interface Message {
  id: string
  session_id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: string
  sources?: Source[]
}

export interface QuestionResponse {
  session_id: string
  question: string
  answer: string
}

export async function askQuestion(
  question: string,
  courseId: string,
  sessionId?: string,
  userId: string = 'anonymous'
): Promise<QuestionResponse> {
  const form = new FormData()
  form.append('question', question)
  form.append('course_id', courseId)
  form.append('user_id', userId)
  if (sessionId) form.append('session_id', sessionId)

  return apiFetch('/ask', { method: 'POST', body: form })
}

export interface AskStreamHandlers {
  onToken: (delta: string) => void
  onSession?: (sessionId: string) => void
  onSources?: (sources: Source[]) => void
  onDone?: (sessionId: string) => void
}

/**
 * Streaming version of askQuestion. Consumes Server-Sent Events from
 * /ask/stream and invokes handlers as answer text arrives.
 */
export async function askQuestionStream(
  question: string,
  courseId: string,
  sessionId: string | undefined,
  userId: string,
  handlers: AskStreamHandlers,
): Promise<void> {
  const form = new FormData()
  form.append('question', question)
  form.append('course_id', courseId)
  form.append('user_id', userId)
  if (sessionId) form.append('session_id', sessionId)

  const resp = await fetch(`${BASE_URL}/ask/stream`, { method: 'POST', body: form })
  if (!resp.ok || !resp.body) throw new Error('Stream failed')

  const reader = resp.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const events = buffer.split('\n\n')
    buffer = events.pop() ?? ''
    for (const evt of events) {
      const dataLine = evt.split('\n').find((l) => l.startsWith('data:'))
      if (!dataLine) continue
      const payload = dataLine.slice(5).trim()
      if (!payload) continue
      try {
        const obj = JSON.parse(payload) as {
          delta?: string
          session_id?: string
          sources?: Source[]
          done?: boolean
        }
        if (obj.session_id && !obj.done) handlers.onSession?.(obj.session_id)
        if (obj.sources) handlers.onSources?.(obj.sources)
        if (obj.delta) handlers.onToken(obj.delta)
        if (obj.done) handlers.onDone?.(obj.session_id ?? '')
      } catch {
        /* ignore malformed SSE frame */
      }
    }
  }
}

export async function getChatSessions(userId: string = 'anonymous'): Promise<ChatSession[]> {
  const data = await apiFetch(`/sessions?user_id=${encodeURIComponent(userId)}`)
  return data.sessions || []
}

export async function getSessionMessages(sessionId: string): Promise<Message[]> {
  const data = await apiFetch(`/sessions/${encodeURIComponent(sessionId)}/messages`)
  return data.messages || []
}

export async function deleteSession(sessionId: string): Promise<void> {
  await apiFetch(`/sessions/${encodeURIComponent(sessionId)}`, { method: 'DELETE' })
}
