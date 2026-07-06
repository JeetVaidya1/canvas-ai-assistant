import { FileText, Sparkles } from 'lucide-react'
import { Badge } from '@/components/ui/Badge'
import type { QuizSource } from '@/lib/api'

/** Chip that cites a source document + page from the user's own materials. */
export function SourceTag({ source, label }: { source: QuizSource; label?: string }) {
  if (!source?.doc_name) return null
  return (
    <Badge tone="neutral" icon={<FileText />}>
      {label && <span className="text-ink-faint">{label}</span>}
      <span className="truncate max-w-[16rem]">{source.doc_name}</span>
      {source.page ? <span className="text-ink-faint">p.{source.page}</span> : null}
    </Badge>
  )
}

/** Chip that surfaces the concept a grounded question is testing. */
export function ConceptTag({ concept }: { concept?: string }) {
  if (!concept) return null
  return (
    <Badge tone="accent" icon={<Sparkles />}>
      <span className="truncate max-w-[14rem]">{concept}</span>
    </Badge>
  )
}
