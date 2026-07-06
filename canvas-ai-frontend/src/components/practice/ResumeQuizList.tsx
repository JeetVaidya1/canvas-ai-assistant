import { X, RotateCcw } from 'lucide-react'
import { Card } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { timeAgoIso } from '@/lib/relativeTime'
import type { QuizInProgressSession } from '@/hooks/useQuizInProgress'

/** Fallback label for a resumable drill that targeted the whole course. */
export const GENERAL_DRILL_LABEL = 'General drill'

interface ResumeQuizListProps {
  sessions: readonly QuizInProgressSession[]
  /** Restore is in flight (disables every Resume button — one run at a time). */
  resuming: boolean
  onResume: (session: QuizInProgressSession) => void
  onDismiss: (quizId: string) => void
}

/**
 * "Continue where you left off" — compact list of resumable drills shown above
 * the quiz setup form. Renders nothing when there is nothing to resume.
 */
export function ResumeQuizList({ sessions, resuming, onResume, onDismiss }: ResumeQuizListProps) {
  if (sessions.length === 0) return null
  return (
    <div className="mb-7 animate-fade-up">
      <p className="mb-2.5 text-[11px] font-semibold uppercase tracking-[0.14em] text-ink-faint">
        Continue where you left off
      </p>
      <ul className="space-y-2">
        {sessions.map((session) => {
          const when = timeAgoIso(session.created_at)
          return (
            <li key={session.quiz_id}>
              <Card padding="none" className="flex items-center gap-3 pl-4 pr-2 py-2.5">
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-medium text-ink">
                    {session.topic ?? GENERAL_DRILL_LABEL}
                  </p>
                  <p className="mt-0.5 text-xs text-ink-faint">
                    <span className="tnum">
                      {session.num_answered}/{session.num_available}
                    </span>{' '}
                    answered{when ? ` · ${when}` : ''}
                  </p>
                </div>
                <Button
                  size="sm"
                  variant="secondary"
                  disabled={resuming}
                  leftIcon={<RotateCcw className="h-3.5 w-3.5" />}
                  onClick={() => onResume(session)}
                  className="flex-shrink-0"
                >
                  Resume
                </Button>
                <button
                  type="button"
                  aria-label={`Dismiss ${session.topic ?? GENERAL_DRILL_LABEL}`}
                  onClick={() => onDismiss(session.quiz_id)}
                  className="focus-ring flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-lg text-ink-faint transition-colors hover:bg-paper-deep hover:text-ink"
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              </Card>
            </li>
          )
        })}
      </ul>
    </div>
  )
}
