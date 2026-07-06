import type { ReactNode } from 'react'
import { PublicNav } from '@/components/marketing/PublicNav'
import { PublicFooter } from '@/components/marketing/PublicFooter'

/** Amber draft notice pinned above every legal document. */
function DraftBanner() {
  return (
    <div className="rounded-lg border border-warning/30 bg-warning-wash px-4 py-3 mb-10">
      <p className="text-sm text-warning font-medium">
        Draft — not yet reviewed by counsel. This document is a working draft and may change before launch.
      </p>
    </div>
  )
}

interface LegalShellProps {
  title: string
  updated: string
  children: ReactNode
}

/**
 * Shared typeset shell for Terms / Privacy: public nav, serif document
 * title, draft banner, prose column, footer.
 */
export function LegalShell({ title, updated, children }: LegalShellProps) {
  return (
    <div className="relative w-full min-h-screen bg-paper text-ink">
      <PublicNav />
      <main className="px-6 pt-28 pb-20 sm:pt-32">
        <article className="max-w-2xl mx-auto animate-fade-up">
          <h1 className="font-display text-3xl sm:text-4xl font-semibold tracking-tight text-ink mb-2">{title}</h1>
          <p className="text-xs text-ink-faint mb-8">Last updated: {updated}</p>
          <DraftBanner />
          <div className="space-y-10">{children}</div>
        </article>
      </main>
      <PublicFooter />
    </div>
  )
}

interface LegalSectionProps {
  num: string
  title: string
  children: ReactNode
}

/** One numbered clause: mono section number, ink heading, soft prose. */
export function LegalSection({ num, title, children }: LegalSectionProps) {
  return (
    <section>
      <div className="flex items-baseline gap-3 mb-3">
        <span className="font-mono text-xs text-ink-faint">{num}</span>
        <h2 className="text-lg font-semibold text-ink tracking-tight">{title}</h2>
      </div>
      <div className="space-y-3 text-sm text-ink-soft leading-relaxed [&_strong]:text-ink [&_strong]:font-medium">
        {children}
      </div>
    </section>
  )
}
