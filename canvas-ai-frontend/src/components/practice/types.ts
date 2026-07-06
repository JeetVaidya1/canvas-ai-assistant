import type { PracticeProblem, QuizAnswerResult, QuizQuestion } from '@/lib/api'
import type { SelectOption } from '@/components/ui/Select'

/** Navigation callback shared by both practice surfaces. */
export type ModeChangeHandler = (
  mode: 'chat' | 'quiz' | 'notes' | 'practice' | 'analytics',
) => void

export type QuizDifficulty = 'easy' | 'medium' | 'hard'
export type PracticeDifficulty = 'adaptive' | 'easy' | 'medium' | 'hard'

export interface DifficultyOption<T extends string = string> {
  value: T
  label: string
  hint: string
}

/** In-flight quiz drill (one generated quiz, answered question by question). */
export interface QuizRunState {
  quizId: string
  questions: QuizQuestion[]
  currentIndex: number
  selectedLetter: string
  feedback: QuizAnswerResult | null
  questionStart: number
  correctCount: number
}

/** In-flight problem-set session (graded client-side at the end). */
export interface PracticeSessionState {
  problems: PracticeProblem[]
  currentProblemIndex: number
  userAnswers: string[]
  startTime: Date
  isComplete: boolean
  score: number
}

/** Visual state of one answer row (caller derives it from selection + reveal). */
export type OptionRowState = 'idle' | 'selected' | 'correct' | 'incorrect' | 'dimmed'

/** Normalized view of the topics query each setup screen consumes. */
export interface TopicListState {
  options: SelectOption[]
  /** True while (re)fetching — also drives the Refresh spinner. */
  loading: boolean
  /** True only on the very first load (no data yet) — show a skeleton. */
  pending: boolean
  /** Query succeeded but the course has no indexed topics. */
  empty: boolean
  /** User-friendly failure message, or null. */
  error: string | null
  refetch: () => void
}
