import type { ReactNode } from 'react'
import { cn } from '@/lib/utils'

export interface SubTab {
  key: string
  label: string
  icon?: ReactNode
  hint?: string
}

interface SubTabsProps {
  tabs: SubTab[]
  active: string
  onChange: (key: string) => void
  className?: string
}

/** Segmented control used inside the consolidated destinations (Learn / Practice
 *  / Study Kit / Progress). Paper & Ink: inset paper well, active = raised
 *  white sheet with ink text. */
export function SubTabs({ tabs, active, onChange, className }: SubTabsProps) {
  return (
    <div className={cn('inline-flex items-center gap-0.5 p-0.5 rounded-lg bg-paper-deep border border-line', className)}>
      {tabs.map((t) => {
        const isActive = t.key === active
        return (
          <button
            key={t.key}
            onClick={() => onChange(t.key)}
            className={cn(
              'inline-flex items-center gap-2 px-3.5 py-1.5 rounded-[7px] text-sm font-medium transition-all focus-ring',
              isActive
                ? 'bg-surface text-ink border border-line shadow-[0_1px_2px_rgba(33,31,26,0.06)]'
                : 'text-ink-soft hover:text-ink border border-transparent',
            )}
            title={t.hint}
            aria-pressed={isActive}
          >
            {t.icon}
            {t.label}
          </button>
        )
      })}
    </div>
  )
}
