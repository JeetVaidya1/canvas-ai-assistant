interface SectionHeadProps {
  num: string
  title: string
}

/**
 * Numbered syllabus header — the editorial section treatment shared by
 * every public/marketing page ("01 — From file dump to study plan").
 */
export function SectionHead({ num, title }: SectionHeadProps) {
  return (
    <div className="section-head mb-10">
      <span className="section-num">{num}</span>
      <h2 className="font-display text-2xl md:text-3xl font-semibold text-ink tracking-tight">{title}</h2>
    </div>
  )
}
