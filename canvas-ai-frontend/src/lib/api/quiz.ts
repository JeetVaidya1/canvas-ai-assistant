// src/lib/api/quiz.ts
import { apiFetch } from './client'

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

/** Lifecycle of background question generation for a quiz. */
export type QuizGenerationStatus = 'ready' | 'generating' | 'partial'

/** Self-reported confidence tapped before submitting an answer. */
export type QuizConfidence = 'sure' | 'thinkso' | 'guessing'

export interface GeneratedQuiz {
  quiz_id: string
  difficulty: string
  topic: string | null
  /** How many questions the user asked for. */
  num_requested: number
  /** How many questions are available right now (grows while generating). */
  num_questions: number
  generation_status: QuizGenerationStatus
  questions: QuizQuestion[]
}

/** GET /quiz/{id}/questions — everything generated so far, ordered q1..qN. */
export interface QuizQuestionsResponse {
  quiz_id: string
  generation_status: QuizGenerationStatus
  num_requested: number
  num_questions: number
  questions: QuizQuestion[]
}

export interface QuizAnswerResult {
  is_correct: boolean
  correct_answer: string
  explanation: string
  concept: string
  source: QuizSource
  mistake_explanation?: string
  mistake_source?: QuizSource
}

export interface QuizTopicScore {
  topic: string
  correct: number
  total: number
  pct: number
}

/** One confidence bucket in the calibration read-out. */
export interface QuizCalibrationBucket {
  n: number
  correct: number
}

export interface QuizCalibration {
  sure: QuizCalibrationBucket
  thinkso: QuizCalibrationBucket
  guessing: QuizCalibrationBucket
  /** Answers marked "sure" that were wrong — priority reviews. */
  confident_wrong: number
}

export interface QuizResult {
  score: { correct: number; total: number; pct: number }
  by_topic: QuizTopicScore[]
  weak_areas: string[]
  /** Present on v3 backends; older responses may omit it. */
  calibration?: QuizCalibration
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
  // Fast-start: the backend returns immediately with the first ~3 questions and
  // keeps writing the rest in the background (poll getQuizQuestions for those).
  return apiFetch('/quiz/generate', { method: 'POST', body: form }, 120_000)
}

/** Everything generated so far for a quiz — poll while generation_status === 'generating'. */
export async function getQuizQuestions(quizId: string): Promise<QuizQuestionsResponse> {
  return apiFetch(`/quiz/${encodeURIComponent(quizId)}/questions`, { method: 'GET' })
}

export async function submitQuizAnswer(
  quizId: string,
  questionId: string,
  selected: string,
  timeTaken: number,
  userId: string = 'anonymous',
  confidence?: QuizConfidence
): Promise<QuizAnswerResult> {
  const form = new FormData()
  form.append('question_id', questionId)
  form.append('selected', selected)
  form.append('time_taken', String(timeTaken))
  form.append('user_id', userId)
  if (confidence) form.append('confidence', confidence)
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
