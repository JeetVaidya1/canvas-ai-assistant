import { BookOpen } from 'lucide-react'
import { Badge } from '@/components/ui/Badge'
import { Tooltip } from '@/components/ui/Tooltip'
import { fileLabel } from '@/components/learn/citation-utils'

interface CitationChipProps {
  file: string
  page?: number | null
}

/**
 * Inline citation chip — the single way a source reference renders in Learn.
 * Badge (accent tone, BookOpen icon) with a Tooltip revealing the full source
 * file path. Used both in the sources disclosure and for inline citations
 * inside answer prose.
 */
export function CitationChip({ file, page }: CitationChipProps) {
  return (
    <Tooltip content={file}>
      <Badge tone="accent" icon={<BookOpen />} className="max-w-[240px] align-middle">
        <span className="min-w-0 truncate">{fileLabel(file)}</span>
        {page !== null && page !== undefined && (
          <span className="rounded bg-cyan-500/15 px-1 text-[10px] font-semibold text-cyan-200">
            p.{page}
          </span>
        )}
      </Badge>
    </Tooltip>
  )
}
