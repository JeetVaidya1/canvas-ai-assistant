// src/components/studykit/SourceCheckbox.tsx — the one styled checkbox visual for Study Kit pickers
import { Check } from 'lucide-react'
import { cn } from '@/lib/utils'

interface SourceCheckboxProps {
  checked: boolean
  className?: string
}

/**
 * Presentation-only checkbox tick used inside selectable rows. The parent
 * (usually a <button aria-pressed>) owns the click target and a11y state —
 * this just renders the shared visual so every picker in studykit/ matches.
 */
export function SourceCheckbox({ checked, className }: SourceCheckboxProps) {
  return (
    <span
      aria-hidden
      className={cn(
        'flex h-[18px] w-[18px] flex-shrink-0 items-center justify-center rounded-[5px] border transition-colors',
        checked ? 'border-transparent bg-gradient-brand' : 'border-white/20 group-hover:border-white/40',
        className,
      )}
    >
      {checked && <Check className="h-3 w-3 text-white" />}
    </span>
  )
}
