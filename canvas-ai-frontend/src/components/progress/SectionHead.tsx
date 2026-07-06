import type { LucideIcon } from 'lucide-react'

interface SectionHeadProps {
  icon: LucideIcon
  title: string
  /** Icon chip surface classes — defaults to the cyan accent chip. */
  chip?: string
  /** Icon tint class — defaults to cyan. */
  tint?: string
  hint?: string
}

/** Section header used across the Progress tiles for a consistent hierarchy. */
export function SectionHead({
  icon: Icon,
  title,
  chip = 'bg-cyan-500/12 border-cyan-400/20',
  tint = 'text-cyan-300',
  hint,
}: SectionHeadProps) {
  return (
    <div className="mb-5 flex items-start gap-2.5">
      <div className={`w-9 h-9 rounded-xl border flex items-center justify-center flex-shrink-0 ${chip}`}>
        <Icon className={`w-5 h-5 ${tint}`} />
      </div>
      <div className="min-w-0">
        <h2 className="text-base font-semibold text-zinc-50 tracking-tight leading-tight">{title}</h2>
        {hint && <p className="text-xs text-zinc-400 mt-0.5">{hint}</p>}
      </div>
    </div>
  )
}
