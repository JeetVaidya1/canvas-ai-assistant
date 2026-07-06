import type { DifficultyOption, PracticeDifficulty, QuizDifficulty } from './types'

// Sentinel for "quiz the entire course" — sends a null topic so the backend does
// broad whole-course retrieval (core concepts) instead of one narrow topic.
export const WHOLE_COURSE = 'Whole course'

export const LETTERS = ['A', 'B', 'C', 'D'] as const

export const QUIZ_DIFFICULTIES: readonly DifficultyOption<QuizDifficulty>[] = [
  { value: 'easy', label: 'Easy', hint: 'Warm-up' },
  { value: 'medium', label: 'Medium', hint: 'Balanced' },
  { value: 'hard', label: 'Hard', hint: 'Exam-grade' },
]

export const QUIZ_COUNTS: readonly number[] = [5, 10, 15, 20]

export const PRACTICE_DIFFICULTIES: readonly DifficultyOption<PracticeDifficulty>[] = [
  { value: 'adaptive', label: 'Adaptive', hint: 'Matches your mastery' },
  { value: 'easy', label: 'Easy', hint: 'Warm-up' },
  { value: 'medium', label: 'Medium', hint: 'Balanced' },
  { value: 'hard', label: 'Hard', hint: 'Push yourself' },
]

export const PRACTICE_COUNTS: readonly number[] = [3, 5, 10, 15]

/** Pen-blue accent for "how far through the session" progress (not a score). */
export const PROGRESS_ACCENT = '#2b4acb'
