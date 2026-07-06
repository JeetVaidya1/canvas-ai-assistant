import type {
  PracticeProblem,
  QuizAnswerResult,
  QuizConfidence,
  QuizGenerationStatus,
  QuizQuestion,
} from '@/lib/api'
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

/** One graded answer, kept client-side so the debrief can replay mistakes. */
export interface AnsweredQuestion {
  question: QuizQuestion
  selectedLetter: string
  confidence: QuizConfidence | null
  result: QuizAnswerResult
}

/**
 * In-flight quiz drill. The run starts as soon as the first ~3 questions land;
 * `questions` grows in the background while `generationStatus === 'generating'`.
 */
export interface QuizRunState {
  quizId: string
  /** Every question available so far, in server order (q1..qN, stable ids). */
  questions: QuizQuestion[]
  /** How many the user asked for — the honest denominator while generating. */
  numRequested: number
  generationStatus: QuizGenerationStatus
  /** What this run targets (snapshotted at start; survives setup edits). */
  topicLabel: string
  currentIndex: number
  selectedLetter: string
  /** Optional confidence tap for the current question (omitted if null). */
  confidence: QuizConfidence | null
  feedback: QuizAnswerResult | null
  questionStart: number
  correctCount: number
  answers: AnsweredQuestion[]
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
