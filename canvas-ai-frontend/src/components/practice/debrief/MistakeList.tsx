import { Badge } from '@/components/ui/Badge'
import { Markdown } from '@/components/ui/Markdown'
import { SourceChip } from '../SourceChip'
import type { AnsweredQuestion } from '../types'
import type { QuizQuestion } from '@/lib/api'

const EXCERPT_MAX = 260

/** Option text for a letter, with the backend's "A) " prefix stripped. */
function optionText(question: QuizQuestion, letter: string): string {
  const index = letter.charCodeAt(0) - 65
  const raw = question.options[index] ?? letter
  return raw.replace(/^[A-D]\)\s*/, '')
}

/** First ~260 chars of an explanation, cut on a word boundary. */
function excerpt(text: string): string {
  if (text.length <= EXCERPT_MAX) return text
  const cut = text.slice(0, EXCERPT_MAX)
  const lastSpace = cut.lastIndexOf(' ')
  return `${cut.slice(0, lastSpace > EXCERPT_MAX * 0.6 ? lastSpace : EXCERPT_MAX)}…`
}

interface MistakeListProps {
  mistakes: AnsweredQuestion[]
}

/** Every wrong answer replayed: your pick vs the right one, why, and the source. */
export function MistakeList({ mistakes }: MistakeListProps) {
  return (
    <div className="space-y-4">
      {mistakes.map((m) => (
        <div key={m.question.id} className="rounded-lg border border-line bg-paper-deep/60 p-4">
          <div className="mb-2.5 flex items-start justify-between gap-3">
            <div className="min-w-0 flex-1 text-sm font-medium text-ink">
              <Markdown content={m.question.question} />
            </div>
            {m.confidence === 'sure' && (
              <Badge tone="warning" className="flex-shrink-0">
                You were sure
              </Badge>
            )}
          </div>

          <div className="mb-2.5 space-y-1.5 text-sm">
            <div className="flex items-baseline gap-2">
              <span className="w-24 flex-shrink-0 text-xs font-medium text-danger">
                Your answer &middot; {m.selectedLetter}
              </span>
              <span className="min-w-0 text-ink-soft">{optionText(m.question, m.selectedLetter)}</span>
            </div>
            {/* Restored (resumed-quiz) answers may not know the correct letter. */}
            {m.result.correct_answer && (
              <div className="flex items-baseline gap-2">
                <span className="w-24 flex-shrink-0 text-xs font-medium text-success">
                  Correct &middot; {m.result.correct_answer}
                </span>
                <span className="min-w-0 text-ink-soft">
                  {optionText(m.question, m.result.correct_answer)}
                </span>
              </div>
            )}
          </div>

          {m.result.explanation && (
            <div className="border-t border-line pt-2.5 text-sm text-ink-soft">
              <Markdown content={excerpt(m.result.explanation)} />
            </div>
          )}
          {(m.result.source?.doc_name || m.question.source?.doc_name) && (
            <div className="mt-2">
              <SourceChip source={m.result.source?.doc_name ? m.result.source : m.question.source} />
            </div>
          )}
        </div>
      ))}
    </div>
  )
}
