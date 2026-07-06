import { cn } from '@/lib/utils'

interface SectionHeadProps {
  /** Mono index for the numbered-syllabus motif, e.g. "02" → "02 — Concept map". */
  num: string
  title: string
  hint?: string
  className?: string
}

/**
 * Numbered syllabus section header used across the Progress tiles — the
 * Paper & Ink `.section-head` / `.section-num` motif for a consistent,
 * report-card hierarchy.
 */
export function SectionHead({ num, title, hint, className }: SectionHeadProps) {
  return (
    <div className={cn('mb-5', className)}>
      <div className="section-head">
        <span className="section-num">{num}</span>
        <h2 className="text-base font-semibold text-ink tracking-tight leading-tight">{title}</h2>
      </div>
      {hint && <p className="text-xs text-ink-soft mt-1.5">{hint}</p>}
    </div>
  )
}
