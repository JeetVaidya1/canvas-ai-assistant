import { cn } from '@/lib/utils'

interface BrandMarkProps {
  /** Tailwind size + layout classes (e.g. "mx-auto mb-5 h-14 w-14"). */
  className?: string
}

/**
 * The Vindexa "V" logo, used as the brand badge that heads the center-first
 * intro / empty states (replacing generic icon tiles). The asset already has
 * its own dark ground + rounded look, so it drops in as a self-contained mark.
 */
export function BrandMark({ className }: BrandMarkProps) {
  return (
    <img
      src="/android-chrome-512x512.png"
      alt="Vindexa"
      className={cn('rounded-2xl object-cover ring-1 ring-white/10', className)}
    />
  )
}
