// src/lib/api.ts
import { supabase } from './supabaseClient'
import type { PostgrestError } from '@supabase/supabase-js'

/** ===== Backend base + helper ===== */
const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

async function apiFetch(path: string, init?: RequestInit, timeoutMs = 60_000) {
  const ctrl = new AbortController()
  const timer = setTimeout(() => ctrl.abort(), timeoutMs)
  try {
    const resp = await fetch(`${BASE_URL}${path}`, { ...init, signal: ctrl.signal })
    // Most FastAPI errors include .detail; fall back to statusText
    if (!resp.ok) {
      let msg = resp.statusText
      try {
        const body = await resp.json()
        msg = body?.detail || body?.message || msg
      } catch { /* ignore json parse errors */ }
      throw new Error(msg || 'Request failed')
    }
    // Some endpoints return no body (void)
    const text = await resp.text()
    return text ? JSON.parse(text) : {}
  } finally {
    clearTimeout(timer)
  }
}

/** ===== Courses (Supabase direct) ===== */
export interface Course {
  course_id: string
  title: string
  created_at: string
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
  const names = (data.files || []).map((f: any) => f.filename)
  return names
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

/** ===== Quiz Assistant ===== */
export interface QuizResponse {
  status: 'success' | 'error'
  answer: string
  explanation: string
  confidence: number
  question_type: string
  study_tips: string[]
  similar_concepts: string[]
  estimated_time: string
  relevant_sources: string[]
  session_id?: string
  error?: string
}

export async function assistWithQuiz(
  question: string,
  courseId: string,
  sessionId?: string,
  userId: string = 'anonymous'
): Promise<QuizResponse> {
  const form = new FormData()
  form.append('question', question)
  form.append('course_id', courseId)
  form.append('user_id', userId)
  if (sessionId) form.append('session_id', sessionId)

  return apiFetch('/quiz-assist', { method: 'POST', body: form })
}

/** ===== Quiz Runner (Phase 3) ===== */
export interface QuizSource {
  doc_name: string | null
  page: number | null
}

export interface QuizQuestion {
  id: string
  question: string
  options: string[]
  concept: string
  difficulty: string
  source: QuizSource
}

export interface GeneratedQuiz {
  quiz_id: string
  difficulty: string
  topic: string | null
  num_questions: number
  questions: QuizQuestion[]
}

export interface QuizAnswerResult {
  is_correct: boolean
  correct_answer: string
  explanation: string
  concept: string
  source: QuizSource
}

export interface QuizTopicScore {
  topic: string
  correct: number
  total: number
  pct: number
}

export interface QuizResult {
  score: { correct: number; total: number; pct: number }
  by_topic: QuizTopicScore[]
  weak_areas: string[]
}

export async function generateQuiz(
  courseId: string,
  topic: string | null,
  difficulty: 'easy' | 'medium' | 'hard' = 'medium',
  numQuestions: number = 10
): Promise<GeneratedQuiz> {
  const form = new FormData()
  form.append('course_id', courseId)
  if (topic) form.append('topic', topic)
  form.append('difficulty', difficulty)
  form.append('num_questions', String(numQuestions))
  // Generation is slow (retrieval + reranker + LLM); allow a generous timeout.
  return apiFetch('/quiz/generate', { method: 'POST', body: form }, 120_000)
}

export async function submitQuizAnswer(
  quizId: string,
  questionId: string,
  selected: string,
  timeTaken: number,
  userId: string = 'anonymous'
): Promise<QuizAnswerResult> {
  const form = new FormData()
  form.append('question_id', questionId)
  form.append('selected', selected)
  form.append('time_taken', String(timeTaken))
  form.append('user_id', userId)
  return apiFetch(`/quiz/${encodeURIComponent(quizId)}/answer`, { method: 'POST', body: form })
}

export async function submitQuiz(
  quizId: string,
  userId: string = 'anonymous'
): Promise<QuizResult> {
  const form = new FormData()
  form.append('user_id', userId)
  return apiFetch(`/quiz/${encodeURIComponent(quizId)}/submit`, { method: 'POST', body: form })
}

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

/** ===== Notes ===== */

export interface Flashcard { q: string; a: string }

export interface NotesResponse {
  status: 'success' | 'error'
  notes: string
  suggested_title: string
  word_count: number
  reading_time: string
  topics: string[]
  source_files: string[]
  message?: string
  flashcards?: { q: string; a: string }[]
}

export interface SavedNote {
  id: string
  course_id: string
  title: string
  content: string
  source_files: string[]
  topic_focus: string
  topics: string[]
  word_count: number
  reading_time: string
  created_at: string
  updated_at: string
}

/**
 * Long-running endpoint — uses apiFetch with extended timeout.
 */
export async function generateNotes(
  courseId: string,
  fileNames: string[],
  topic: string = '',
  style: 'detailed' | 'summary' | 'outline' = 'detailed'
): Promise<NotesResponse> {
  const form = new FormData();
  form.append('course_id', courseId);
  form.append('file_names', JSON.stringify(fileNames));
  form.append('topic', topic);
  form.append('style', style);

  const data = await apiFetch('/generate-notes', { method: 'POST', body: form }, 300_000);
  if (data.status !== 'success') {
    throw new Error(data.message || 'Notes generation failed');
  }
  return data;
}

export async function saveNotes(
  courseId: string,
  title: string,
  content: string,
  sourceFiles: string[],
  topic: string = '',
  noteId?: string
): Promise<SavedNote> {
  const form = new FormData();
  form.append('course_id', courseId);
  form.append('title', title);
  form.append('content', content);
  form.append('source_files', JSON.stringify(sourceFiles));
  form.append('topic', topic);
  if (noteId) form.append('note_id', noteId);

  const data = await apiFetch('/save-notes', { method: 'POST', body: form });
  return data.note;
}

export async function updateNote(
  noteId: string,
  courseId: string,
  title: string,
  content: string,
  sourceFiles: string[] = [],
  topic: string = ''
): Promise<SavedNote> {
  const form = new FormData();
  form.append('course_id', courseId);
  form.append('title', title);
  form.append('content', content);
  form.append('source_files', JSON.stringify(sourceFiles));
  form.append('topic', topic);

  const data = await apiFetch(`/notes/${encodeURIComponent(noteId)}`, { method: 'PUT', body: form });
  return data.note;
}

export async function getNotes(courseId: string): Promise<SavedNote[]> {
  const data = await apiFetch(`/notes/${encodeURIComponent(courseId)}`);
  return data.notes || [];
}

export async function deleteNotes(noteId: string): Promise<void> {
  await apiFetch(`/notes/${encodeURIComponent(noteId)}`, { method: 'DELETE' });
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
  }>
}

export async function getLearningAnalytics(courseId: string, userId: string = 'anonymous'): Promise<LearningAnalytics> {
  const data = await apiFetch(`/analytics/${encodeURIComponent(courseId)}/${encodeURIComponent(userId)}`)
  return data.analytics
}

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

/** ===== Optional: system status/health ===== */
export async function getSystemStatus() {
  return apiFetch('/system-status')
}

export async function getRagHealth() {
  return apiFetch('/rag-health')
}

export async function generatePracticeExam(params: {
  courseId: string
  examType?: string
  questionCount?: number
  timeLimit?: number
  difficulty?: 'easy'|'medium'|'hard'|'mixed'
  questionTypes?: string[]
  topicFocus?: string
  userId?: string
}) {
  const form = new FormData()
  form.append('course_id', params.courseId)
  form.append('exam_type', params.examType ?? 'practice')
  form.append('question_count', String(params.questionCount ?? 10))
  form.append('time_limit', String(params.timeLimit ?? 120))
  form.append('difficulty', params.difficulty ?? 'mixed')
  form.append('question_types', JSON.stringify(params.questionTypes ?? ["multiple_choice","calculation","short_answer"]))
  form.append('topic_focus', params.topicFocus ?? '')
  form.append('user_id', params.userId ?? 'anonymous')

  return apiFetch('/api/generate-practice-exam', { method: 'POST', body: form })
}

export async function createExamSession(courseId: string, userId = 'anonymous', exam: unknown = null) {
  const form = new FormData()
  form.append('exam_data', JSON.stringify(exam))
  form.append('user_id', userId)
  form.append('course_id', courseId)
  return apiFetch('/api/create-exam-session', { method: 'POST', body: form })
}

export async function startExamSession(sessionId: string) {
  return apiFetch(`/api/start-exam-session/${encodeURIComponent(sessionId)}`, { method: 'POST' })
}

export async function pauseExamSession(sessionId: string) {
  return apiFetch(`/api/pause-exam-session/${encodeURIComponent(sessionId)}`, { method: 'POST' })
}

export async function saveExamAnswer(sessionId: string, questionId: string, answer: string) {
  const form = new FormData()
  form.append('session_id', sessionId)
  form.append('question_id', questionId)
  form.append('answer', answer)
  return apiFetch('/api/save-exam-answer', { method: 'POST', body: form })
}

export async function navigateExamQuestion(sessionId: string, index: number) {
  const form = new FormData()
  form.append('session_id', sessionId)
  form.append('question_index', String(index))
  return apiFetch('/api/navigate-exam-question', { method: 'POST', body: form })
}

export async function submitExamSession(sessionId: string) {
  return apiFetch(`/api/submit-exam/${encodeURIComponent(sessionId)}`, { method: 'POST' })
}

export async function solveExamQuestion(params: {
  courseId: string,
  questionText: string,
  wantHint?: boolean,
  pastPaperId?: string,
  pages?: number[],
  file?: File | null
}) {
  const form = new FormData()
  form.append('course_id', params.courseId)
  form.append('question_text', params.questionText)
  form.append('want_hint', String(Boolean(params.wantHint)))
  form.append('pages', JSON.stringify(params.pages || []))
  if (params.pastPaperId) form.append('past_paper_id', params.pastPaperId)
  if (params.file) form.append('pdf_file', params.file)
  return apiFetch('/api/solve-exam-question', { method: 'POST', body: form })
}

export async function uploadPastPaper(courseId: string, file: File, userId = 'anonymous') {
  const form = new FormData()
  form.append('course_id', courseId)
  form.append('file', file)
  form.append('user_id', userId)
  return apiFetch('/api/upload-past-paper', { method: 'POST', body: form })
}

export async function getExamHistory(userId: string, courseId?: string) {
  const url = courseId 
    ? `/api/exam-history/${encodeURIComponent(userId)}?course_id=${encodeURIComponent(courseId)}`
    : `/api/exam-history/${encodeURIComponent(userId)}`
  return apiFetch(url)
}

export async function getPastPapers(courseId: string) {
  return apiFetch(`/api/past-papers/${encodeURIComponent(courseId)}`)
}

/** ===== Study Planner ===== */
export interface StudyPlan {
  id: string
  course_id: string
  days: Array<{
    date: string
    topics: string[]
    duration_minutes: number
    type: 'review' | 'new' | 'practice'
  }>
  created_at: string
}

export async function generateStudyPlan(
  courseId: string,
  params: {
    daysAvailable?: number
    hoursPerDay?: number
    examDate?: string
  } = {}
): Promise<StudyPlan> {
  const form = new FormData()
  form.append('course_id', courseId)
  if (params.daysAvailable) form.append('days_available', String(params.daysAvailable))
  if (params.hoursPerDay) form.append('hours_per_day', String(params.hoursPerDay))
  if (params.examDate) form.append('exam_date', params.examDate)
  return apiFetch('/api/generate-study-plan', { method: 'POST', body: form })
}

export async function getStudyPlan(courseId: string): Promise<StudyPlan | null> {
  try {
    return await apiFetch(`/api/study-plan/${encodeURIComponent(courseId)}`)
  } catch {
    return null
  }
}

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

/** ===== Export ===== */
export async function exportNotesPdf(courseId: string): Promise<Blob> {
  const ctrl = new AbortController()
  const timer = setTimeout(() => ctrl.abort(), 120_000)
  try {
    const resp = await fetch(`${BASE_URL}/api/export-notes-pdf/${encodeURIComponent(courseId)}`, {
      signal: ctrl.signal,
    })
    if (!resp.ok) throw new Error('Export failed')
    return await resp.blob()
  } finally {
    clearTimeout(timer)
  }
}

export async function exportFlashcardsAnki(courseId: string): Promise<Blob> {
  const ctrl = new AbortController()
  const timer = setTimeout(() => ctrl.abort(), 120_000)
  try {
    const resp = await fetch(`${BASE_URL}/api/export-flashcards-anki/${encodeURIComponent(courseId)}`, {
      signal: ctrl.signal,
    })
    if (!resp.ok) throw new Error('Export failed')
    return await resp.blob()
  } finally {
    clearTimeout(timer)
  }
}

export async function exportPlannerIcal(courseId: string): Promise<Blob> {
  const ctrl = new AbortController()
  const timer = setTimeout(() => ctrl.abort(), 120_000)
  try {
    const resp = await fetch(`${BASE_URL}/api/export-planner-ical/${encodeURIComponent(courseId)}`, {
      signal: ctrl.signal,
    })
    if (!resp.ok) throw new Error('Export failed')
    return await resp.blob()
  } finally {
    clearTimeout(timer)
  }
}

/** ===== Canvas LMS Import ===== */
export async function importCanvasLms(
  token: string,
  canvasCourseId: string
): Promise<{ status: string; message: string }> {
  const form = new FormData()
  form.append('canvas_token', token)
  form.append('canvas_course_id', canvasCourseId)
  return apiFetch('/api/import-canvas', { method: 'POST', body: form })
}

