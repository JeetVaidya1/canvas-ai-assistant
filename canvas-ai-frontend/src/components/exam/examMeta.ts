// Presentation metadata + pure helpers shared across the exam screens.
import { CheckCircle, CircleDot, XCircle } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import type { ExamDifficulty, ExamSession, Verdict } from './types'

/** Human-readable message from an unknown thrown value. */
export function errText(e: unknown): string {
  return e instanceof Error ? e.message : String(e)
}

// Centered setup choices — tactile, mirrors QuizMode's center-first flow.
export const DIFFICULTIES: { value: ExamDifficulty; label: string; hint: string }[] = [
  { value: 'easy', label: 'Easy', hint: 'Warm-up' },
  { value: 'medium', label: 'Medium', hint: 'Balanced' },
  { value: 'hard', label: 'Hard', hint: 'Exam-grade' },
  { value: 'mixed', label: 'Mixed', hint: 'Like the real thing' },
]

export const COUNTS = [5, 8, 12, 15] as const

export function formatTime(seconds: number): string {
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const secs = seconds % 60
  if (hours > 0) return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
  return `${minutes}:${secs.toString().padStart(2, '0')}`
}

export function readinessLabel(pct: number): string {
  if (pct >= 85) return 'Exam ready'
  if (pct >= 70) return 'Nearly ready'
  if (pct >= 50) return 'Getting there'
  return 'Needs work'
}

export interface VerdictMeta {
  label: string
  /** Card tone for the per-question review row. */
  tone: string
  text: string
  /** Badge tone matching the primitive's tone system. */
  badgeTone: 'success' | 'warning' | 'danger'
  Icon: LucideIcon
}

export function verdictMeta(v: Verdict): VerdictMeta {
  switch (v) {
    case 'correct':
      return {
        label: 'Correct',
        tone: 'border-emerald-500/25 bg-emerald-500/[0.06]',
        text: 'text-emerald-400',
        badgeTone: 'success',
        Icon: CheckCircle,
      }
    case 'partial':
      return {
        label: 'Partial',
        tone: 'border-amber-500/25 bg-amber-500/[0.06]',
        text: 'text-amber-400',
        badgeTone: 'warning',
        Icon: CircleDot,
      }
    default:
      return {
        label: 'Incorrect',
        tone: 'border-rose-500/25 bg-rose-500/[0.06]',
        text: 'text-rose-400',
        badgeTone: 'danger',
        Icon: XCircle,
      }
  }
}

export const SAMPLE_EXAM: ExamSession = {
  id: 'sample',
  examName: 'Physics 122 Practice Exam',
  timeLimit: 120,
  currentQuestion: 0,
  isActive: false,
  isPaused: false,
  userAnswers: {},
  questions: [
    {
      id: '1', type: 'multiple_choice',
      question: 'If your heart is beating at 76.0 beats per minute, what is the frequency in hertz?',
      options: ['1.27 Hz', '0.79 Hz', '1.33 Hz', '0.76 Hz'],
      points: 2, topic: 'Oscillations', difficulty: 'easy', answer: 'A', timeEstimate: 3,
    },
    {
      id: '2', type: 'calculation',
      question: 'A 4.0-g string is 0.36 m long. It vibrates at 500 Hz in its third harmonic. What is the wavelength?',
      points: 3, topic: 'Waves', difficulty: 'medium', answer: '0.24 m',
      solution: 'Third harmonic: L = 3λ/2 → λ = 2L/3 = 0.24 m', timeEstimate: 5,
    },
    {
      id: '3', type: 'short_answer',
      question: 'Three point charges are located on the x-axis. Calculate the magnitude of the electric force on the middle charge.',
      points: 4, topic: 'Electrostatics', difficulty: 'hard', timeEstimate: 8,
    },
  ],
}
