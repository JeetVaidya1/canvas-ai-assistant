import { cn } from '@/lib/utils'

interface BrandMarkProps {
  /** Tailwind size + layout classes (e.g. "mx-auto mb-5 h-14 w-14"). */
  className?: string
}

/**
 * The Vindexa mark, Paper & Ink edition: a serif "V" set white on the
 * pen-blue square — typographic, printable, no gradients. The glyph scales
 * with the box via an SVG viewBox, so any height/width classes just work.
 */
export function BrandMark({ className }: BrandMarkProps) {
  return (
    <span
      aria-label="Vindexa"
      role="img"
      className={cn('inline-flex items-center justify-center rounded-[22%] bg-accent select-none overflow-hidden', className)}
    >
      <svg viewBox="0 0 32 32" className="w-full h-full" aria-hidden="true">
        <text
          x="16"
          y="23.5"
          textAnchor="middle"
          fontFamily="Newsreader, Georgia, serif"
          fontWeight="600"
          fontSize="21"
          fill="#ffffff"
        >
          V
        </text>
      </svg>
    </span>
  )
}
