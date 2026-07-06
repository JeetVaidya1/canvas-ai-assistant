// Shared types for the Exam destination. The state machine lives in
// useExamSession; every screen component consumes these shapes.

export type QType = 'multiple_choice' | 'calculation' | 'short_answer' | 'essay' | 'diagram' | 'proof'
export type Diff = 'easy' | 'medium' | 'hard'
export type ExamDifficulty = Diff | 'mixed'

export interface ExamQuestion {
  id: string
  type: QType
  question: string
  options?: string[]
  points: number
  topic: string
  difficulty: Diff
  answer?: string
  solution?: string
  timeEstimate: number
}

export interface ExamSession {
  id: string
  examName: string
  questions: ExamQuestion[]
  timeLimit: number
  startTime?: Date
  endTime?: Date
  userAnswers: Record<string, string>
  currentQuestion: number
  isActive: boolean
  isPaused: boolean
}

export type SolveJSON = {
  final_answer: string
  steps: string[]
  choice?: string | null
  units?: string | null
}

export type Verdict = 'correct' | 'partial' | 'incorrect'

export interface BreakdownItem {
  question: string
  userAnswer?: string
  correctAnswer?: string
  points?: number
  pointsEarned?: number
  verdict?: string
  gradeReason?: string
  mistakeExplanation?: string
  timeSpent?: number
  topic?: string
}

export interface ExamResults {
  totalQuestions: number
  correctAnswers: number
  totalPoints: number
  earnedPoints: number
  percentage: number
  letterGrade?: string | null
  topicPerformance?: Record<string, { earned?: number; possible?: number; percentage?: number }> | null
  timeEfficiency?: string | null
  timeSpent: number
  breakdown: BreakdownItem[]
}

/** Shape of a generated exam question as returned by the backend. */
export interface RawExamQuestion {
  id: string
  type: QType
  question: string
  options?: string[]
  points?: number
  topic?: string
  difficulty?: Diff
  correct_answer?: string
  explanation?: string
  time_estimate?: number
}

/** Shape of a graded question result as returned by the backend. */
export interface RawQuestionResult {
  question: string
  user_answer?: string
  correct_answer?: string
  points_possible?: number
  points_earned?: number
  verdict?: string
  grade_reason?: string
  mistake_explanation?: string
  time_spent?: number
  topic?: string
}

/** Past-paper upload analysis summary from the backend. */
export interface PastPaperAnalysis {
  status?: string
  questions_found?: number
  message?: string
}
