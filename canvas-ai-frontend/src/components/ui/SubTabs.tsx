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
    <div className={cn('inline-flex items-center gap-1 p-1 rounded-xl bg-[#101014] border border-[#1f1f26]', className)}>
      {tabs.map((t) => {
        const isActive = t.key === active
        return (
          <button
            key={t.key}
            onClick={() => onChange(t.key)}
            className={cn(
              'inline-flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-sm font-medium transition-all',
              isActive
                ? 'bg-gradient-brand-soft text-cyan-100 ring-1 ring-inset ring-cyan-400/25 shadow-[0_2px_10px_-4px_rgba(34,211,238,0.5)]'
                : 'text-zinc-400 hover:text-zinc-100 hover:bg-white/[0.05]',
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
