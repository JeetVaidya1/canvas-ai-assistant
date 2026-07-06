import { BookOpen } from 'lucide-react'
import { Tooltip } from '@/components/ui/Tooltip'
import { fileLabel } from '@/components/learn/citation-utils'

interface CitationChipProps {
  file: string
  page?: number | null
}

/**
 * Inline citation chip — the single way a source reference renders in Learn.
 * Paper & Ink: the `.footnote-ref` textbook treatment (tinted pen-blue chip),
 * with a Tooltip revealing the full source file path. Used both in the sources
 * disclosure and for inline citations inside answer prose.
 */
export function CitationChip({ file, page }: CitationChipProps) {
  return (
    <Tooltip content={file}>
      <span className="footnote-ref max-w-[240px] align-middle">
        <BookOpen className="h-3 w-3 flex-shrink-0" />
        <span className="min-w-0 truncate">{fileLabel(file)}</span>
        {page !== null && page !== undefined && (
          <span className="font-mono text-[10px] font-medium opacity-75 tnum">p.{page}</span>
        )}
      </span>
    </Tooltip>
  )
}
