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
 *  / Study Kit / Progress) to switch between their sub-modes. */
export function SubTabs({ tabs, active, onChange, className }: SubTabsProps) {
  return (
    <div className={cn('inline-flex items-center gap-1 p-1 rounded-xl bg-zinc-900 border border-zinc-800', className)}>
      {tabs.map((t) => {
        const isActive = t.key === active
        return (
          <button
            key={t.key}
            onClick={() => onChange(t.key)}
            className={cn(
              'inline-flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-sm font-medium transition-all',
              isActive
                ? 'bg-gradient-brand-soft text-cyan-200 ring-1 ring-inset ring-cyan-500/20'
                : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/60',
            )}
            title={t.hint}
          >
            {t.icon}
            {t.label}
          </button>
        )
      })}
    </div>
  )
}
