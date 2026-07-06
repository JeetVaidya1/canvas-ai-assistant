import type { ReactNode } from 'react'
import { motion } from 'motion/react'
import { BrandMark } from '@/components/ui/BrandMark'
import { cn } from '@/lib/utils'

interface SetupShellProps {
  title: string
  subtitle: string
  children: ReactNode
}

/**
 * Center-first setup layout shared by both practice surfaces (mirrors the
 * Chat/Notes "composer first" pattern): brand mark, confident title, then a
 * short stack of tactile controls ending in one primary CTA.
 */
export function SetupShell({ title, subtitle, children }: SetupShellProps) {
  return (
    <div className="flex min-h-full flex-col items-center justify-center px-4 py-10">
      <motion.div
        initial={{ opacity: 0, y: 14 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
        className="w-full max-w-xl"
      >
        <div className="mb-8 text-center">
          <BrandMark className="mx-auto mb-5 h-14 w-14" />
          <h1 className="font-display text-[28px] font-semibold text-ink">{title}</h1>
          <p className="mx-auto mt-2 max-w-md text-sm text-ink-soft">{subtitle}</p>
        </div>
        {children}
      </motion.div>
    </div>
  )
}

interface FieldLabelProps {
  children: ReactNode
  /** Optional right-aligned action (e.g. a Refresh link). */
  action?: ReactNode
  center?: boolean
}

/** Small setup-field label (normal case — the page keeps a single eyebrow). */
export function FieldLabel({ children, action, center = false }: FieldLabelProps) {
  if (action) {
    return (
      <div className="mb-2.5 flex items-center justify-between">
        <span className="text-xs font-semibold text-ink-soft">{children}</span>
        {action}
      </div>
    )
  }
  return (
    <div className={cn('mb-2.5 text-xs font-semibold text-ink-soft', center && 'text-center')}>
      {children}
    </div>
  )
}
