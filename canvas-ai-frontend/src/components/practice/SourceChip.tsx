import { BookOpen } from 'lucide-react'
import { Tooltip } from '@/components/ui/Tooltip'
import type { QuizSource } from '@/lib/api'

/** Basename of a source path — `slides/week3.pdf` → `week3.pdf`. */
function docLabel(doc: string): string {
  const parts = doc.split(/[/\\]/)
  return parts[parts.length - 1] || doc
}

interface SourceChipProps {
  source: QuizSource | null | undefined
  className?: string
}

/**
 * Footnote-style citation chip for quiz sources — the practice-side twin of
 * Learn's CitationChip (same `.footnote-ref` textbook treatment), kept local
 * so the practice folder doesn't reach into learn/.
 */
export function SourceChip({ source, className }: SourceChipProps) {
  if (!source?.doc_name) return null
  return (
    <Tooltip content={source.doc_name}>
      <span className={`footnote-ref max-w-[240px] align-middle ${className ?? ''}`}>
        <BookOpen className="h-3 w-3 flex-shrink-0" />
        <span className="min-w-0 truncate">{docLabel(source.doc_name)}</span>
        {source.page !== null && source.page !== undefined && (
          <span className="font-mono text-[10px] font-medium opacity-75 tnum">p.{source.page}</span>
        )}
      </span>
    </Tooltip>
  )
}
