// src/lib/api/exams.ts
import { apiFetch } from './client'

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
